import json
import requests
import random
from typing import Dict, TypedDict, Annotated
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, MessagesState, END, add_messages
from langchain_openai import ChatOpenAI
from nearai.agents.environment import Environment
import re
api_key = env.env_vars.get("OPEN_API_KEY", "default_value")
print(api_key)  # This will print your updated API key
llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    last_agent: str
    questions: str
    selected_students: list[str]
    student_1_answer: str  
    student_2_answer: str  
    student_3_answer: str  

def supervisor(state: AgentState):
    """Routes requests based on user intent and determines which student agents will answer."""
    print("Current State:", state)
    user_input = state["messages"][-1].content
    last_agent = state.get("last_agent", None)

    prompt = (
        f"User Request: '{user_input}'\n\n"
        "Determine if the user input is related to:\n"
        "- 'question_generation' (e.g., 'generate a question about...')\n"
        "- 'answer_question' (when the user is responding to a question that was asked)\n"
        "- 'chat' (general conversation)\n\n"
        "If the input is related to 'question_generation', respond with 'teacher'.\n"
        "If the input is the user providing an answer to a question, respond with 'student'.\n"
        "If the input is general conversation, respond with 'chat'."
    )
    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
    print("Decision:", decision)

    if decision == "teacher":
        return {"messages": state["messages"], "next": "teacher_agent", "last_agent": "teacher_agent"}

    elif decision == "student":
        # Randomly select how many student agents will answer (1, 2, or all 3)
        possible_students = ["student_agent_1", "student_agent_2", "student_agent_3"]
        selected_students = random.sample(possible_students, random.randint(1, 3))

        return {
            "messages": state["messages"],
            "next": selected_students[0], 
            "last_agent": "supervisor",
            "selected_students": selected_students
        }
    
    else:
        return {"messages": state["messages"], "next": "chat_agent", "last_agent": "chat_agent"}

def teacher_agent_node(state: AgentState):
    """Handles question generation tasks."""
    user_input = state["messages"][-1].content

    question_generation_prompt = (
        f"User Request: '{user_input}'\n\n"
        "You are a teacher generating questions. "
        "Generate clear and relevant questions based on the subject and quantity requested. "
        "Do not decide which student should answer; all students will receive the same question.\n\n"
        "Each response must start with:\n\n"
        "- 'Teacher's Question:' if only **one** question is generated.\n"
        "- 'Teacher's Questions:' if **multiple** questions are generated."
    )
    response = llm.invoke([SystemMessage(content=question_generation_prompt)])
    generated_questions = response.content  
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": "student_agent_1",
        "last_agent": "teacher_agent",
        "questions": generated_questions 
    }

def student_agent_1(state: AgentState):
    """Student Agent 1: Provides a basic response to a question."""
    print("Current State:", state)
    question = state.get("questions", "").strip()
    
    if not question:
        return {
            "messages": state["messages"] + [SystemMessage(content="No question available.")],
            "next": "END"
        }

    answer_prompt = (
        f"Question: '{question}'.\n\n"
        "You are Student 1, an AI assistant known for being precise and direct. "
        "Your responses should be short, clear, and strictly factual. Do not provide extra context "
        "or explanations unless specifically asked.\n\n"
        "If you receive one question, start with 'Student 1's Answer:'.\n"
        "If you receive multiple questions, start with 'Student 1's Answers:' and list them concisely.\n\n"
        "When answering multiple questions, apply the label only to the header, not to each individual answer."
    )
    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state["student_1_answer"] = response.content
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),  # Calls the function to get the next student
        "last_agent": "student_agent_1",
        "student_1_answer": state["student_1_answer"],
    }

def student_agent_2(state: AgentState):
    """Student Agent 2: Provides a more detailed response."""
    question = state["questions"]
    
    answer_prompt = (
        f"Question: '{question}'.\n\n"
        "You are Student 2, an AI assistant who explains answers briefly but with enough context "
        "to ensure understanding. While keeping responses concise, provide a short explanation or reasoning "
        "when necessary to enhance clarity.\n\n"
        "If you receive one question, start with 'Student 2's Answer:'.\n"
        "If you receive multiple questions, start with 'Student 2's Answers:' and list them with brief explanations.\n\n"
        "When answering multiple questions, apply the label only to the header, not to each individual answer."
    )
    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state["student_2_answer"] = response.content
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),  # Calls the function to get the next student
        "last_agent": "student_agent_2",
        "student_2_answer": state["student_2_answer"],
    }


def student_agent_3(state: AgentState):
    """Student Agent 3: Provides an alternative perspective or example."""
    question = state["questions"]
    
    answer_prompt = (
        f"Question: '{question}'.\n\n"
        "You are Student 3, an AI assistant that answers questions in a more conversational and engaging way. "
        "While still being informative, your responses should feel natural and friendly, sometimes including examples "
        "or analogies when helpful.\n\n"
        "If you receive one question, start with 'Student 3's Answer:'.\n"
        "If you receive multiple questions, start with 'Student 3's Answers:' and list them in an engaging manner.\n\n"
        "When answering multiple questions, apply the label only to the header, not to each individual answer."
    )
    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state["student_3_answer"] = response.content
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),  # Calls the function to get the next student
        "last_agent": "student_agent_3",
        "student_3_answer": state["student_3_answer"],
    }


def chat_agent(state: AgentState):
    """Handles general conversations."""
    response = llm.invoke(state["messages"]) 
    env.add_reply(response.content)
    return {"messages": state["messages"] + [response], "next": "END"}

def leaderboard_agent(state: AgentState):
    """Evaluates the answers of the selected student agents and the latest human message, then provides feedback in table format."""
    print("Current State in Leaderboard:", state)

    teacher_question = state.get("questions", "").strip()  # This is a multi-line string
    selected_students = state.get("selected_students", [])

    if not teacher_question or not selected_students:
        return {
            "messages": state["messages"] + [SystemMessage(content="No question or selected students available for evaluation.")],
            "next": "END",
            "last_agent": "leaderboard_agent"
        }

    # 🔹 Count the total number of questions using regex
    total_questions = len(re.findall(r"^\d+\.", teacher_question, re.MULTILINE))
    if total_questions == 0:
        total_questions = 1  # If no numbered questions are found, assume there's at least 1 question

    scores = {student: {"total_questions": total_questions, "score": 0} for student in selected_students}
    scores["Human"] = {"total_questions": total_questions, "score": 0}

    # Generate the correct answer using LLM
    correct_answer_prompt = (
        f"Question: '{teacher_question}'\n\n"
        "Provide the correct answer to this question. Be concise and accurate."
    )
    correct_answer = llm.invoke([SystemMessage(content=correct_answer_prompt)]).content.strip()

    # Retrieve and evaluate the latest human message
    latest_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    if latest_message:
        latest_answer = latest_message.content.strip()
        evaluation_prompt = (
            f"Question: '{teacher_question}'\n\n"
            f"Correct Answer: '{correct_answer}'\n\n"
            f"Latest Answer: '{latest_answer}'\n\n"
            "Evaluate whether the latest answer is correct. Respond with 'Correct' or 'Incorrect'."
        )
        latest_evaluation = llm.invoke([SystemMessage(content=evaluation_prompt)]).content.strip()
        if latest_evaluation == "Correct":
            scores["Human"]["score"] += 1

    # Evaluate student answers
    for student in selected_students:
        student_answer_key = f"student_{student.split('_')[-1]}_answer"
        student_answer = state.get(student_answer_key, "").strip()

        if not student_answer:
            continue

        evaluation_prompt = (
            f"Question: '{teacher_question}'\n\n"
            f"Correct Answer: '{correct_answer}'\n\n"
            f"Student Answer: '{student_answer}'\n\n"
            "Evaluate whether the student's answer is correct. Respond with 'Correct' or 'Incorrect'."
        )
        evaluation = llm.invoke([SystemMessage(content=evaluation_prompt)]).content.strip()
        if evaluation == "Correct":
            scores[student]["score"] += 1

    # Construct the leaderboard table
    leaderboard_table = "Leaderboard Evaluation:\n\n"
    leaderboard_table += "| Name            | Total Questions | Current Score |\n"
    leaderboard_table += "|----------------|----------------|---------------|\n"
    env.add_reply(leaderboard_table)
    for name, data in scores.items():
        leaderboard_table += f"| {name.ljust(15)} | {str(data['total_questions']).ljust(14)} | {str(data['score']).ljust(13)} |\n"

    return {
        "messages": state["messages"] + [SystemMessage(content=leaderboard_table)],
        "next": "END",
        "last_agent": "leaderboard_agent"
    }
# Build the LangGraph
workflow = StateGraph(AgentState)

# Register Nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("teacher_agent", teacher_agent_node)
workflow.add_node("student_agent_1", student_agent_1)
workflow.add_node("student_agent_2", student_agent_2)
workflow.add_node("student_agent_3", student_agent_3)
workflow.add_node("chat_agent", chat_agent)

# Define Routing Logic
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next"], {
    "teacher_agent": "teacher_agent",
    "student_agent_1": "student_agent_1",
    "student_agent_2": "student_agent_2",
    "student_agent_3": "student_agent_3",
    "chat_agent": "chat_agent",
    "END": END
})


workflow.add_edge("teacher_agent", END)

workflow.add_node("leaderboard_agent", leaderboard_agent)

def get_next_student(state: AgentState):
    """Returns the next student or routes to the leaderboard if it's the last student."""
    selected_students = state.get("selected_students", [])
    
    if state["last_agent"] in selected_students:
        current_index = selected_students.index(state["last_agent"])
        if current_index + 1 < len(selected_students):
            return selected_students[current_index + 1]  # Next student
        else:
            return "leaderboard_agent"  # Last student goes to leaderboard
    
    return "leaderboard_agent"  # Default to leaderboard if no students left

workflow.add_conditional_edges("student_agent_1", get_next_student, {
    "student_agent_2": "student_agent_2",
    "student_agent_3": "student_agent_3",
    "leaderboard_agent": "leaderboard_agent"
})

workflow.add_conditional_edges("student_agent_2", get_next_student, {
    "student_agent_1": "student_agent_1",
    "student_agent_3": "student_agent_3",
    "leaderboard_agent": "leaderboard_agent"
})

workflow.add_conditional_edges("student_agent_3", get_next_student, {
    "student_agent_1": "student_agent_1",
    "student_agent_2": "student_agent_2",
    "leaderboard_agent": "leaderboard_agent"
})


# Leaderboard agent ends the workflow
workflow.add_edge("leaderboard_agent", END)

workflow.add_edge("chat_agent", END)
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
    output = graph.invoke({"messages": messages}, config)

    # Print output messages
    for message in output["messages"]:
        message.pretty_print()

    env.request_user_input()