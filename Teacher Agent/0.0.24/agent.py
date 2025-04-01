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
    questions: str = ""
    human_score: int = 0
    student_1_total_score: int = 0  
    student_2_total_score: int = 0  
    student_3_total_score: int = 0  

def supervisor(state: AgentState):

    print("Current State from Supervisor:", state)
    user_input = state["messages"][-1].content
    last_agent = state.get("last_agent", None)

    prompt = (
   f"User Request: '{user_input}'\n\n"
    "Determine if the user input is related to:\n"
    "- 'question_generation' (e.g., 'generate a question about...')\n"
    "- 'answer_question' (when the user is responding to a question that was asked, even if they don't number their answers)\n"
    "- 'chat' (general conversation)\n"
    "- 'student_performance' (if the user is asking about student scores, comparisons, or performance insights)\n"
    "- 'unsure_answer' (if the user indicates they don’t know the answer, such as 'I don’t know' or 'Could you help?')\n"
    "- 'learn_topic' (if the user asks to learn or explore a specific subject, topic, or concept, e.g., 'Teach me about gravity' or 'I want to learn algebra')\n\n"

    "**Routing Logic:**\n"
    
    "- If the input is related to 'question_generation', 'student_performance', or 'learn_topic', respond with 'teacher'.\n"
    "- If the input is the user providing an answer (even without numbering), respond with 'student'.\n"
    "- If the input is general conversation, respond with 'chat'.\n"
    "- If the input suggests the user doesn’t know the answer (e.g., 'I don’t know', 'I’m not sure', 'Can you help?'), respond with 'student'.\n\n"

    "**Student Performance Queries (Route to 'teacher' agent):**\n"
    "- Asking about a specific student’s score (e.g., 'What is Student 1’s score?')\n"
    "- Comparing student scores (e.g., 'Who is performing better, Student 1 or Student 2?')\n"
    "- Finding the highest-scoring student (e.g., 'Who has the highest score?')\n"
    "- Requesting an overall student performance summary (e.g., 'How are all students performing?')\n"
    "- Seeking advice on improving scores (e.g., 'How can Student 2 improve their performance?')\n"

    "**Learning Requests (Route to 'teacher' agent):**\n"
    "- If the user asks to learn or explore a subject/topic (e.g.,'I want to understand quantum physics'),\n"
    "  classify it as 'learn_topic' and route it to the teacher agent."

    "**Uncertain Answer Responses (Route to 'student' agent):**\n"
    "- If the user expresses uncertainty (e.g., 'I don’t know the answer', 'Can you help me?', 'I’m not sure'),\n"
    "  redirect them to the 'student' agent for guidance, hints, or follow-up questions instead of giving direct answers."

    "**Answer Detection (Route to 'student' agent):**\n"
    "- If the user provides an answer, even without numbering (e.g., 'The capital of France is Paris.'), it is still classified as 'answer_question' and routed to the student agent."
    "- If the user uses numbering (e.g., '1. The capital of France is Paris.'), it is also classified as 'answer_question' and routed correctly."
    )

    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
    print("Decision:", decision)

    if decision == "teacher":
        return {"messages": state["messages"], "next": "teacher_agent", "last_agent": "teacher_agent"}

    elif decision == "student":
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
    try:
        # Read the state file safely
        saved_state = env.read_file(STATE_FILE)
        state_model = State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        state_model = State()

    user_input = state["messages"][-1].content
    print(state["messages"])
    print(env.list_messages())
    question_generation_prompt = SystemMessage(
    f"User Request: '{user_input}'\n\n"
    "You are a teacher agent in a virtual classroom. Your sole responsibility is to guide users through interactive learning by "
    "generating questions. When a user selects a subject and specifies the number of questions, you generate and present that exact number of "
    "questions.\n\n"

    "**Handling User Input:**\n"
    "- If the user specifies both a subject and a number of questions, generate exactly that many questions.\n"
    "- If the user only specifies a subject without mentioning the number of questions, respond with:\n"
    "  - 'How many questions would you like on [selected subject]?' and wait for the user's response before generating questions.\n"
    "- If the user asks for an explanation instead of questions, respond with:\n"
    "Each response must start with:\n"

    "- 'Teacher's Question:' if only **one** question is generated.\n"
    "- 'Teacher's Questions:' if **multiple** questions are generated.\n\n"

    "Each question must start with a number followed by a period (e.g., '1. ', '2. ', '3. ') and must not include 'Question 1:' or any other prefixes like 'Puzzle 1:'. "
    "Ensure that all questions are **unique** and do not repeat.\n\n"

    "**Important Restriction:** If the user asks for an explanation of a topic, you must **not** provide an answer. Instead, respond with: "
    "'I am a teacher agent. My purpose is to generate questions for learning, not to provide explanations or answers. Please attempt the questions to deepen your understanding.' "
    "You must strictly follow this rule in all cases.\n\n"
    
    "After presenting the questions, encourage the user to think and provide their answers. You can phrase it like this:\n"
    "'Take a moment to think and share your answers. If you're unsure, feel free to ask! I'll call any of your classmates Student 1, Student 2, or Student 3 to answer.'\n\n"

    "Additionally, you are responsible for tracking and remembering student performance throughout the learning session. Each student's total current "
    "score is recorded and updated as they answer questions. You can respond to queries about a student's score at any time.\n\n"

    "**Current Scores:**\n"
    f"- Student 1: {state_model.student_1_total_score}\n"
    f"- Student 2: {state_model.student_2_total_score}\n"
    f"- Student 3: {state_model.student_3_total_score}\n"
    f"- Your Score: {state_model.human_score}\n\n"

    "**🏆 Leaderboard:**\n"
    "| Student Name | Total Score |\n"
    "|--------------|------------|\n"
    f"| Student 1    | {state_model.student_1_total_score} |\n"
    f"| Student 2    | {state_model.student_2_total_score} |\n"
    f"| Student 3    | {state_model.student_3_total_score} |\n"
    f"| You          | {state_model.human_score} |\n\n"

    "- Rank students based on total scores, including the user’s score.\n"
    "- The highest score gets Rank 1.\n"
    "- If students have the same score, they should have the same rank.\n\n"

    "**If a user asks about the top-performing student, respond with:**\n"
    "- 'The best-performing student is [Student Name] with a total score of [Score]. Keep it up!'\n"
    "- If multiple students have the same highest score, say: 'Both [Student 1] and [Student 2] are leading with [Score]. Keep competing!'\n\n"

    "**If a user asks about overall student performance, how a student can improve, or advice on competing better, respond with:**\n"
    "- Provide a summary of the current student rankings using a leaderboard format.\n"
    "- If asked about improvement, suggest strategies such as revisiting weaker topics, practicing more questions, and maintaining consistency.\n"
    "- If asked about competing better, encourage students to review past mistakes, participate actively, and aim for accuracy and speed in answering questions.\n"
    "- Keep responses concise, focusing on motivation and actionable advice without unnecessary explanations.\n\n"

    "**User Score Response:**\n"
    f"- If the user asks about their score: 'Your current total score is {state_model.human_score}. Keep going!'\n"
    "- If the user asks how they compare to others:\n"
    "  - Compare their score with student scores and provide feedback.\n"
    "  - Example: 'You are ahead of Student X but behind Student Y. Keep pushing!'\n\n"

    "**If a user asks for an update on student scores or overall performance, provide the response in a table format:**\n\n"

    "**🏆 Leaderboard**\n"
    "| Student Name | Total Score | Rank |\n"
    "|--------------|------------|------|\n"
    f"| Student 1    | {state_model.student_1_total_score} | X |\n"
    f"| Student 2    | {state_model.student_2_total_score} | Y |\n"
    f"| Student 3    | {state_model.student_3_total_score} | Z |\n"
    f"| You          | {state_model.human_score} | W |\n\n"

    "- Rank students based on their total scores. The highest score gets Rank 1.\n"
    "- If students have the same score, they share the same rank.\n"
    "- Do not provide any unnecessary text—only return the formatted table.\n\n"

    "If a user asks about overall performance, respond with:\n"
    "- A summary of current rankings using the leaderboard.\n"
    "- Encouragement for students to improve their performance.\n"
    "- Advice on competing better, focusing on accuracy, speed, and consistency.\n\n"

    "Encourage students to improve their performance, engage actively, and compete in a friendly manner to enhance their learning experience."
    )


    # Extract content from env.list_messages()
    messages = [msg["content"] for msg in env.list_messages()]
    print("msg ---",messages)

    response = llm.invoke([question_generation_prompt] + messages)
    generated_questions = response.content  

    # Set the generated question in state_model
    state_model.questions = generated_questions  

    # Save the updated state back to the file
    env.write_file(STATE_FILE, state_model.model_dump_json())

    # Add reply to the environment
    env.add_reply(response.content)

    return {
        "messages": state["messages"] + [response],
        "next": "get_next_student(state)",
        "last_agent": "teacher_agent",
    }


def student_agent_1(state: AgentState):
    """Student Agent 1: Provides a basic response to a question."""
    print("Current State from Student 1--", state)  

    try:
        # Read the state file safely
        saved_state = env.read_file(STATE_FILE)
        state_model = State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        state_model = State()

    print("Current State from Student 1--", state_model)  

    question = state_model.questions.strip()
    
    if not question:
        no_question_msg = "No question available from Student--1"
        env.add_reply(no_question_msg)
        
        updated_state = {
            "messages": state["messages"] + [SystemMessage(content="No question available.")],
            "next": "END"
        }
        
        print("Updated State from Student 1 (No Question Case)--", updated_state)  
        return updated_state
    user_input = state["messages"][-1].content
    answer_prompt = (
        f"Question: '{question}'.\n\n"
        "You are Student 1, an AI assistant known for being precise and direct. "
        "Your responses should be short, clear, and strictly factual. Do not provide extra context "
        "or explanations unless specifically asked.\n\n"
        "If you receive one question, start with 'Student 1's Answer:'.\n"
        "If you receive multiple questions, start with 'Student 1's Answer:' and list them concisely.\n\n"
        "When answering multiple questions, apply the label only to the header, not to each individual answer."
    )

    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state_model.student_1_answers = response.content
    env.write_file(STATE_FILE, state_model.model_dump_json())
    env.add_reply(response.content)

    updated_state = {
        "messages": state["messages"] + [response],
        "next": get_next_student(state), 
        "last_agent": "student_agent_1",
    }
    
    print("Updated State from Student 1 (After Answering)--", updated_state)
    return updated_state

def student_agent_2(state: AgentState):
    """Student Agent 2: Provides a more detailed response."""
    try:
        # Read the state file safely
        saved_state = env.read_file(STATE_FILE)
        state_model = State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        state_model = State()

    print("Current State from Student 2--", state_model)  

    question = state_model.questions.strip()
    user_input = state["messages"][-1].content
    if not question:
        no_question_msg = "No question available from Student--2"
        env.add_reply(no_question_msg)
        return {
            "messages": state["messages"] + [SystemMessage(content="No question available.")],
            "next": "END"
        }
        
    answer_prompt = (
        f"Question: '{question}'.\n\n"
        "You are Student 2, an AI assistant who explains answers briefly but with enough context "
        "to ensure understanding. While keeping responses concise, provide a short explanation or reasoning "
        "when necessary to enhance clarity.\n\n"
        "If you receive one question, start with # 'Student 2's Answer:'.\n"
        "If you receive multiple questions, start with 'Student 2's Answer:' and list them with brief explanations.\n\n"
        "When answering multiple questions, apply the label only to the header, not to each individual answer."
    )

    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    state_model.student_2_answers = response.content
    env.write_file(STATE_FILE, state_model.model_dump_json())
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state),
        "last_agent": "student_agent_2",
    }


def student_agent_3(state: AgentState):
    """Student Agent 3: Provides an alternative perspective or example."""
    try:
        # Read the state file safely
        saved_state = env.read_file(STATE_FILE)
        state_model = State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        state_model = State()

    print("Current State from Student 3--", state_model)  

    question = state_model.questions.strip()
    user_input = state["messages"][-1].content
    if not question:
        no_question_msg = "No question available from Student--3"
        env.add_reply(no_question_msg)
        return {
            "messages": state["messages"] + [SystemMessage(content="No question available.")],
            "next": "END"
        }
        
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
    state_model.student_3_answers = response.content
    env.write_file(STATE_FILE, state_model.model_dump_json())
    print(f"Student 3's answer: {response.content}")
    env.add_reply(response.content)
    return {
        "messages": state["messages"] + [response],
        "next": get_next_student(state), 
        "last_agent": "student_agent_3",
    
    }


def chat_agent(state: AgentState):
    """Handles general conversations."""
    response = llm.invoke(state["messages"]) 
    env.add_reply(response.content)
    return {"messages": state["messages"] + [response], "next": "END"}


def leaderboard_agent(state: AgentState):
    """Evaluates the answers of the selected student agents and the latest human message, then provides feedback in table format."""
    print("Current State in Leaderboard:", state)

    try:
        # Read the state file safely
        saved_state = env.read_file(STATE_FILE)
        state_model = State.model_validate_json(saved_state) if saved_state else State()
    except Exception as e:
        print(f"Error loading state file: {e}")
        state_model = State()

    teacher_question = state_model.questions.strip()
    selected_students = state.get("selected_students", [])

    if not teacher_question or not selected_students:
        return {
            "messages": state["messages"] + [SystemMessage(content="No question or selected students available for evaluation.")],
            "next": "END",
            "last_agent": "leaderboard_agent"
        }

    # 🔹 Print all student answers from state
    print("All Student Answers from State File:")
    for attr_name in dir(state_model):
        if attr_name.startswith("student_") and attr_name.endswith("_answers"):
            print(f"{attr_name}: {getattr(state_model, attr_name)}")

    # 🔹 Count the total number of questions using regex
    total_questions = len(re.findall(r"^\d+\.", teacher_question, re.MULTILINE))
    if total_questions == 0:
        total_questions = 1  # If no numbered questions are found, assume there's at least 1 question

    # Retrieve the latest human message
    latest_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    latest_answer = latest_message.content.strip() if latest_message else ""

    # Prepare student answers dynamically
    student_answers = {}
    for student in selected_students:
        student_answer_key = f"student_{student.split('_')[-1]}_answers"
        student_answer = getattr(state_model, student_answer_key, "").strip()
        student_answers[student] = student_answer
        print(f"Student: {student}, Answer: {student_answer}")  # 🔹 Print selected student answers

    # 🔹 Print scores before update
    print("\n📌 Student Scores Before Update:")
    for student in selected_students:
        score_key = f"student_{student.split('_')[-1]}_total_score"
        print(f"{student}: {getattr(state_model, score_key, 0)}")

    # Generate the evaluation prompt
    student_answer_list = "\n".join([f"- {student}: {answer}" for student, answer in student_answers.items()])
    leaderboard_rows = "\n".join([f"| {student.ljust(11)} | {total_questions}         | Y          |" for student in selected_students])
    feedback_list = "\n".join([f"**{student}:**\n- [Dynamic feedback based on {student}'s performance]" for student in selected_students])

    eval_prompt = f"""
    You are an AI evaluator. Assess student and human responses strictly based on correctness and context.

    **Evaluation Criteria:**
    - Award 1 mark per correct answer.
    - Consider partial correctness only if it maintains the core meaning.
    - Answers must be directly relevant to the questions.

    **Questions:**
    {teacher_question}

    **Student Answers:**
    {student_answer_list}

    **Your Answer:**
    {latest_answer}

    **Tasks:**
    1. Evaluate the answers of the students and the human based on the questions and criteria.
    2. Assign a score out of {total_questions} for each student and the human.
    3. Generate a leaderboard table in the exact format provided below.
    4. Provide dynamic feedback for each student and the human based on their performance.

    **Leaderboard Table Format:**
    ## 🏆 **Leaderboard**

    | AgentName   | TotalQuestions | TotalScore |
    |-------------|----------------|------------|
    | You       | {total_questions}         | X          |
    {leaderboard_rows}

    **Dynamic Feedback Rules:**
    - If a participant scores 100%, provide positive reinforcement and encourage them to maintain their performance.
    - If a participant scores between 50% and 99%, highlight their strengths and suggest areas for improvement.
    - If a participant scores below 50%, provide constructive feedback and recommend revisiting the topic.

    **Provide the response in the following format:**

    ## 🏆 **Leaderboard**

    | AgentName   | TotalQuestions | TotalScore |
    |-------------|----------------|------------|
    | You       | {total_questions}         | X          |
    {leaderboard_rows}

    **Feedback:**

    **You:**
    - [Dynamic feedback based on the human's performance]

    {feedback_list}

    Keep practicing and focus on improving accuracy and detail in your answers.
    """

    # Invoke the LLM to evaluate and generate the leaderboard and feedback
    evaluation_response = llm.invoke([SystemMessage(content=eval_prompt)]).content.strip()
    human_pattern = r"\| You.*?\| (\d+) *\|"
    human_match = re.search(human_pattern, evaluation_response)
    if human_match:
        human_score = int(human_match.group(1))  # Extracted human score
        print(f"Extracted Human Score: {human_score}")
        state_model.human_score += human_score
    # 🔹 Extract scores using regex
    for student in selected_students:
        student_number = student.split('_')[-1]
        score_key = f"student_{student_number}_total_score"
        
        # Regex pattern to extract the student's score
        pattern = rf"\| {student}.*?\| (\d+) *\|"
        match = re.search(pattern, evaluation_response)
        
        if match:
            new_score = int(match.group(1))  # Extracted score
            setattr(state_model, score_key, getattr(state_model, score_key, 0) + new_score)  # Update state

    # 🔹 Print scores after update
    print("\n✅ Student Scores After Update:")
    for student in selected_students:
        score_key = f"student_{student.split('_')[-1]}_total_score"
        print(f"{student}: {getattr(state_model, score_key, 0)}")

    # Use env.add_reply to display the leaderboard table and feedback
    env.add_reply(evaluation_response)
    env.write_file(STATE_FILE, state_model.model_dump_json())
    return {
        "messages": state["messages"] + [SystemMessage(content=evaluation_response)],
        "next": "END",
        "last_agent": "leaderboard_agent"
    }

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
            return selected_students[current_index + 1]  
        else:
            return "leaderboard_agent"  
    
    return "END" 

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
    output = graph.invoke({"messages": messages})

    # Print output messages
    for message in output["messages"]:
        message.pretty_print()

    env.request_user_input()