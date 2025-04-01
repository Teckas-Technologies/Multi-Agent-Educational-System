import json
import requests
import random
from typing import Dict, TypedDict, Annotated
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, MessagesState, END, add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel 
from nearai.agents.environment import Environment
import re
api_key = env.env_vars.get("OPEN_API_KEY", "default_value")
print(api_key)  # This will print your updated API key
llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
STATE_FILE = "teacher_file.json"
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    last_agent: str
    selected_students: list[str]

class State(BaseModel):
    student_1_answers: str = ""
    student_2_answers: str = ""
    student_3_answers: str = ""
    user_answer: str = ""
    questions: str = ""
    last_agent: str = ""
    selected_agents: str = ""
    
student_names = {
    "student_agent_1": "Jack",
    "student_agent_2": "Olivia",
    "student_agent_3": "Alden",
    "user_node": "Human Student",  
}

def _load_state() -> State:
    """Helper to safely load state from file."""
    try:
        saved_state = env.read_file(STATE_FILE)
        return State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        return State()  # Returns fresh state if error occurs

def _save_state(state_model: State) -> None:
    """Helper to safely save state to file."""
    try:
        env.write_file(STATE_FILE, state_model.model_dump_json())
    except Exception as e:
        print(f"Error saving state file: {e}")

def teacher_agent_node(state: AgentState):
    """Handles topic selection, question generation, and summarizing student responses."""
    state_model = _load_state()
    previous_messages = env.list_messages()

    print("\nPrevious Messages:", previous_messages if previous_messages else "No previous messages.")
    print("Conversation History:", previous_messages)

    last_agent = state_model.last_agent.strip() if state_model.last_agent else None
    print(f"[DEBUG] Last Agent: '{last_agent}'")

    # If last agent was user, store their response from messages
    if last_agent == "user_node":
        user_response = previous_messages[-1]['content']
        
        print(f"Storing user response: {user_response}")
        state_model.user_answer = user_response  # Or use dedicated user_answers field
        _save_state(state_model)

    if last_agent in {"user_node", "student_agent_1", "student_agent_2", "student_agent_3"}:
        # Parse selected agents
        selected_agents = [a.strip() for a in state_model.selected_agents.split(",")] if state_model.selected_agents else []
        print("Selected Agents from State:", selected_agents)
        
        # Separate student and user responses
        student_responses = []
        user_response = None
        
        # Collect student responses
        student_answers = {
            "student_agent_1": state_model.student_1_answers,
            "student_agent_2": state_model.student_2_answers,
            "student_agent_3": state_model.student_3_answers
        }
        
        for agent_id, answer in student_answers.items():
            if agent_id in selected_agents and answer:
                student_num = agent_id.split("_")[-1]  # Gets 1, 2, or 3
                student_responses.append(f"Student {student_num}: {answer}")
        
        # Collect user response if selected and available
        if "user_node" in selected_agents and last_agent == "user_node" and state_model.user_answer:
            user_response = f"User: {state_model.user_answer}"
        
        # Combine all responses for prompt
        all_responses = student_responses.copy()
        if user_response:
            all_responses.append(user_response)
            print("All -----------------------------------------:", all_responses)
        # Generate appropriate prompt
        if all_responses and state_model.questions:
            responses_text = "\n".join(all_responses)
            prompt_content = (
                f"You are the teacher. Here are the student and user responses:\n\n{responses_text}\n\n"
                "Summarize these responses and provide a brief conclusion. Address the user directly using 'you' when referring to their response. Then, introduce a new topic and ask a new question to continue the lesson.\n"
                "Your response must start with 'Teacher:' on one line, and the actual response should begin on the next line. For example:\n\n"
                "'Teacher:'\n"
                "For example, phrase it like this:'Thank you, everyone, for your insightful responses. It seems we all agree that mathematics is essential in our daily lives. "
    "           Now, let's move on to our next topic: Physics. What do you think is the most fundamental concept in physics?'"
            )
        else:
            print(" No relevant responses found. Starting a new topic.")
            prompt_content = (
                "You are the teacher. There are no student responses yet.\n\n"
                "Introduce a new topic and ask a question to begin the lesson.\n"
                "Your response must start with 'Teacher:' on one line, and the actual response should begin on the next line. For example:\n\n"
                "'Teacher:'\n"
                "For example, phrase it like this:'Let's begin with Mathematics. What role do you think mathematics plays in our everyday lives? Student 1, could you start us off?'"
            )
    else:
        prompt_content = (
            "You are the teacher. Start a new discussion with a relevant topic and question.\n"
            "Your response must start with 'Teacher:'."
        )

    response = llm.invoke([SystemMessage(content=prompt_content)])

    if not response or not response.content:
        print("[ERROR] LLM did not return a valid response.")
        return  

    state_model.last_agent = "teacher_agent"
    state_model.questions = response.content

    # Clear student answers when moving to new topic
    state_model.student_1_answers = ""
    state_model.student_2_answers = ""
    state_model.student_3_answers = ""
    state_model.user_answer = ""
    print("Teacher's Response:", response.content)

    all_students = list(student_names.keys()) + ["user_node"]
    selected_students = random.sample(all_students, random.randint(1, 3))
    
    # Ensure user_node is always last if present
    if "user_node" in selected_students:
        selected_students.remove("user_node")
        selected_students.append("user_node")

    state_model.selected_agents = ", ".join(selected_students)
    selected_student_id = selected_students[0]
    selected_student_name = student_names.get(selected_student_id, selected_student_id)

    if selected_student_name == "Human Student":
        response_with_student = f"{response.content} What do you think?"
    else:
        response_with_student = f"{response.content} {selected_student_name}, what do you think?"
        env.add_reply(response_with_student)
        _save_state(state_model)

    return {
        "messages": previous_messages + [SystemMessage(content=response_with_student)],
        "next": selected_student_id,
        "last_agent": "teacher_agent",
        "selected_students": selected_students,
    }

def user_node(state: AgentState):
    state_model = _load_state()
    state_model.last_agent = "user_node"  # Set in file state
    _save_state(state_model)
    
    return {
        "messages": state["messages"],
        "next": "END",
        "last_agent": "user_node",  # Also set in runtime state
    }


def student_agent_1(state: AgentState):
    state_model = _load_state()
    conversation_context = state["messages"][-1].content
    next_student_name = get_next_student_name(state, "student_agent_1")
    state_model.last_agent = "student_agent_1"
    
    answer_prompt = (
        f"Conversation Context: '{conversation_context}'\n\n"
        "You are Jack, an AI assistant known for being precise and direct. "
        "Your responses should be short, clear, and strictly factual. Do not provide additional context "
        "or explanations unless specifically requested.\n\n"
        "Your response must start with 'Jack:' on one line, followed by your actual response on the next line."
    )

    if next_student_name == "Human Student":
        answer_prompt += (
        "\n\nAt the end of your response, ask the user directly by addressing them as 'you'. "
        "For example, phrase it like this: 'What do you think about this, do you agree?'."
        )
    else:
        answer_prompt += (
        "\n\nAt the end of your response, call on the following student by name for their input: "
        f"{next_student_name}."
        )

    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state_model.student_1_answers = response.content
    _save_state(state_model)
    env.add_reply(response.content)
    
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),
        "last_agent": "student_agent_1",
    }

def student_agent_2(state: AgentState):
    """Student Agent 2: Provides a more detailed response."""
    state_model = _load_state()
    state_model.last_agent = "student_agent_2"
    conversation_context = state["messages"][-1].content
    next_student_name = get_next_student_name(state, "student_agent_2")

    answer_prompt = (
        f"Conversation Context: '{conversation_context}'\n\n"
        "You are Olivia, an AI assistant who provides concise yet clear explanations. "
        "While keeping responses brief, include a short explanation or reasoning when necessary to ensure understanding.\n\n"
        "Your response must start with 'Olivia:' on one line, followed by your actual response on the next line."
    )

    if next_student_name == "Human Student":
        answer_prompt += (
        "\n\nAt the end of your response, ask the user directly by addressing them as 'you'. "
        "For example, phrase it like this: 'What do you think about this, do you agree?'."
        )
    else:
        answer_prompt += (
        "\n\nAt the end of your response, call on the following student by name for their input: "
        f"{next_student_name}."
        )

    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state_model.student_2_answers = response.content
    _save_state(state_model)
    env.add_reply(response.content)
    
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),
        "last_agent": "student_agent_2",
    }

def student_agent_3(state: AgentState):
    state_model = _load_state()
    state_model.last_agent = "student_agent_3"
    conversation_context = state["messages"][-1].content
    next_student_name = get_next_student_name(state, "student_agent_3")

    answer_prompt = (
        f"Conversation Context: '{conversation_context}'\n\n"
        "You are Alden, an AI assistant who answers questions in a conversational and engaging way. "
        "Your responses should be informative yet natural and friendly, incorporating examples or analogies when helpful.\n\n"
        "Your response must start with 'Alden:' on one line, followed by your actual response on the next line."
    )

    if next_student_name == "Human Student":
        answer_prompt += (
        "\n\nAt the end of your response, ask the user directly by addressing them as 'you'. "
        "For example, phrase it like this: 'What do you think about this, do you agree?'."
        )
    else:
        answer_prompt += (
        "\n\nAt the end of your response, call on the following student by name for their input: "
        f"{next_student_name}."
        )


    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state_model.student_3_answers = response.content
    _save_state(state_model)
    env.add_reply(response.content)
    
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),
        "last_agent": "student_agent_3",
    }

def get_next_student_name(state: AgentState, current_agent_name: str) -> str:
    """Returns the display name of the next student agent after the current one, if any."""
    selected_students = state.get("selected_students", [])

    if current_agent_name in selected_students:
        current_index = selected_students.index(current_agent_name)
        if current_index + 1 < len(selected_students):
            next_agent_id = selected_students[current_index + 1]  # Get the next agent ID
            return student_names[next_agent_id]  # Return the display name of the next student
    return None  # No next student

def get_next_student(state: AgentState):
    """Determines the next student or routes to the teacher for summarization after the last student."""
    selected_students = state.get("selected_students", [])
    last_agent = state["last_agent"]
    
    if last_agent in selected_students:
        current_index = selected_students.index(last_agent)
        if current_index + 1 < len(selected_students):
            return selected_students[current_index + 1]  # Next student
        else:
            return "teacher_agent"  # After last student, return to teacher
    return "teacher_agent"  # Default to teacher if no students left

# Build the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("teacher_agent", teacher_agent_node)
workflow.add_node("student_agent_1", student_agent_1)
workflow.add_node("student_agent_2", student_agent_2)
workflow.add_node("student_agent_3", student_agent_3)
workflow.add_node("user_node", user_node)  # Add the user node
workflow.set_entry_point("teacher_agent")

# Add conditional edges for teacher agent
workflow.add_conditional_edges("teacher_agent", lambda state: state["next"], {
    "student_agent_1": "student_agent_1",
    "student_agent_2": "student_agent_2",
    "student_agent_3": "student_agent_3",
    "user_node": "user_node",  # Route to user node if selected
    "END": END  # End the conversation after summarizing
})

# Add conditional edges for student agents
workflow.add_conditional_edges("student_agent_1", get_next_student, {
    "student_agent_2": "student_agent_2",
    "student_agent_3": "student_agent_3",
    "user_node": "user_node",  # Route to user node if selected
    "teacher_agent": "teacher_agent"  # Route back to teacher for summary
})

workflow.add_conditional_edges("student_agent_2", get_next_student, {
    "student_agent_1": "student_agent_1",
    "student_agent_3": "student_agent_3",
    "user_node": "user_node",  # Route to user node if selected
    "teacher_agent": "teacher_agent"  # Route back to teacher for summary
})

workflow.add_conditional_edges("student_agent_3", get_next_student, {
    "student_agent_1": "student_agent_1",
    "student_agent_2": "student_agent_2",
    "user_node": "user_node",  # Route to user node if selected
    "teacher_agent": "teacher_agent"  # Route back to teacher for summary
})

# Add conditional edge for user node
workflow.add_conditional_edges("user_node", lambda state: state["next"], {
    "END": END  # Always end after user node
})

# Compile the Graph
graph = workflow.compile()
user_input = env.list_messages()[-1]["content"]
print(f"User's latest input: {user_input}")
# Check for exit condition
if user_input.lower() in ["exit", "quit"]:
    print("Exiting...")
else:
    # Prepare input message for LangGraph
    messages = [HumanMessage(content=user_input)]

    # Invoke workflow with user input
    output = graph.invoke({"messages": messages})

    # Print output messages
    for message in output["messages"]:
        message.pretty_print()

    env.request_user_input()