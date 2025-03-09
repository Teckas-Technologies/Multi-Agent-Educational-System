import functools
import operator
from typing import Annotated, Sequence, TypedDict
from enum import Enum
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel  # ✅ UPDATED import
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import requests
from tabulate import tabulate  # Import for table formatting
import re
import json
from langgraph.checkpoint.memory import MemorySaver
from nearai.agents.environment import Environment
from nearai_langchain.orchestrator import NearAILangchainOrchestrator
STATE_FILE = "teacher_file.json"

# Step 1: Initiate LLM
api_key = env.env_vars.get("OPEN_API_KEY", "default_value")
print(api_key)  # This will print your updated API key
llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

class State(BaseModel):
    student_1_total_score: int = 0  
    student_2_total_score: int = 0  
    student_3_total_score: int = 0 

# Step 2: Define Student Agents using Prompts
def create_student_agent(llm: ChatOpenAI, system_prompt: str, name: str):
    system_prompt_template = PromptTemplate(
        template=system_prompt + """
        ONLY respond to the part of the query relevant to your purpose.
        IGNORE tasks you can't complete.
        Use the following context to answer your query if available: \n {agent_history} \n
        """,
        input_variables=["agent_history"],
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | llm

# Define prompts for each student
student1_prompt = """
You are Student 1, an AI assistant known for being precise and direct. Your responses should be short, clear, and strictly factual. Do not provide extra context or explanations unless specifically asked.

If you receive **one question**, start with **"🎓 Student 1's Answer:**".  
If you receive **multiple questions**, start with **"🎓 Student 1's Answers:**" and list them concisely.
When answering multiple questions, apply the 🎓 emoji only to the header, not to each individual answer.
"""

student2_prompt = """
You are Student 2, an AI assistant who explains answers briefly but with enough context to ensure understanding. While keeping responses concise, provide a short explanation or reasoning when necessary to enhance clarity.

If you receive **one question**, start with **"🎓 Student 2's Answer:**".  
If you receive **multiple questions**, start with **"🎓 Student 2's Answers:**" and list them with brief explanations.
When answering multiple questions, apply the 🎓 emoji only to the header, not to each individual answer.

"""

student3_prompt = """
You are Student 3, an AI assistant that answers questions in a more conversational and engaging way. While still being informative, your responses should feel natural and friendly, sometimes including examples or analogies when helpful.

If you receive **one question**, start with **"🎓 Student 3's Answer:"**.  
If you receive **multiple questions**, start with **"🎓 Student 3's Answers:"** and list them in an engaging manner.

When answering multiple questions, apply the 🎓 emoji only to the header, not to each individual answer.
"""



# Create student agents
student1_agent = create_student_agent(llm, student1_prompt, "Student1")
student2_agent = create_student_agent(llm, student2_prompt, "Student2")
student3_agent = create_student_agent(llm, student3_prompt, "Student3")


def teacher_node(state):
    # Load the current state from the file
    saved_state = env.read_file(STATE_FILE)
    state_model = State.model_validate_json(saved_state) if saved_state else State()

    # Construct the dynamic system prompt including current scores
    system_prompt = f"""
    You are a teacher generating questions.
Generate clear and relevant questions based on the subject and quantity requested.
Do not decide which student should answer; all students will receive the same question.

Each response must start with:
- **"🧑‍🏫 Teacher's Question:"** if only **one** question is generated.
- **"🧑‍🏫 Teacher's Questions:"** if **multiple** questions are generated.

**Important Restriction:**  
If the user asks for an explanation of a topic, you must **not** provide an answer. Instead, your response should be:  
'I am a teacher agent. My purpose is to generate questions for learning, not to provide explanations or answers. Please attempt the questions to deepen your understanding.'  
You must strictly follow this rule in all cases.

After presenting the questions, guide the user to choose the next chapter or topic to continue their learning journey.  
Always focus on generating questions in the specified format and facilitating topic progression.

---

**Student Performance Tracking:**  
Additionally, you are responsible for tracking and remembering student performance throughout the learning session. Each student's total current
score is recorded and updated as they answer questions. You can respond to queries about a student's score at any time.  
Example queries:
- "What is Student1's total score?"
- "How is Student2 performing overall?"
- "Show the leaderboard."

**Current Scores:**
- Student 1: {state_model.student_1_total_score}
- Student 2: {state_model.student_2_total_score}
- Student 3: {state_model.student_3_total_score}

---

**Leaderboard Format:**  
If a user asks for an update on student scores or overall performance, provide the response in a table format:

**🏆 Leaderboard**  
| Student Name | Total Score | Rank |
|--------------|------------|------|
| Student 1    | {state_model.student_1_total_score} | X |
| Student 2    | {state_model.student_2_total_score} | Y |
| Student 3    | {state_model.student_3_total_score} | Z |

- Rank students based on their total scores. The highest score gets Rank 1, the second highest Rank 2, and so on.
- If students have the same score, they should have the same rank.
- Do not provide any unnecessary text—only return the formatted table.

---

**If a user asks about the top-performing student, respond with:**  
- 'The best-performing student is [Student Name] with a total score of [Score]. Keep it up!'
- If multiple students have the same highest score, say: ' [Student 1] and [Student 2] [Student 3]are leading with [Score]. Keep competing!'

---

**If a user asks for a comparison between two students:**  
- Compare their scores directly.
- If one student has a higher score, say:  
  - "[Student X] is currently ahead with [Score] points, while [Student Y] has [Score] points. Keep pushing forward!"  
- If their scores are equal, say:  
  - "[Student X] and [Student Y] are tied with [Score] points. It’s a close competition!"
- Provide constructive feedback:  
  - If one student is significantly ahead, encourage the other student to practice more.  
  - If they are close in scores, motivate both students to compete actively.  

---
**If a user asks about overall student performance, improvement strategies, or how to compete better:**  
- Provide the current leaderboard.  
- If asked about improvement, suggest strategies like revisiting weaker topics, practicing more questions, and maintaining consistency.  
- If asked about competing better, encourage students to review past mistakes, participate actively, and aim for accuracy and speed.  
- Keep responses concise, focusing on motivation and actionable advice without unnecessary explanations.  

Encourage students to improve their performance, engage actively, and compete in a friendly manner to enhance their learning experience.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_history")
    ])

    # Teacher chain now only generates a question (the output will contain the question)
    teacher_chain = prompt | llm

    response = teacher_chain.invoke(state)  # Generate the question
    teacher_question = response.content  # Extract question content
    env.add_reply(response.content)
    # Check if the response is a casual message (not a question)
    if not teacher_question.startswith(("🧑‍🏫 Teacher's Question:", "🧑‍🏫 Teacher's Questions:")):
        # If it's a casual message, return it and mark it as such
        return {
            "messages": [AIMessage(content=teacher_question)],
            "agent_history": state["agent_history"],
            "is_casual": True  # Add a flag to indicate casual message
        }
    
    # If it's a question, proceed as normal
    return {
        "messages": [AIMessage(content=teacher_question)], 
        "agent_history": state["agent_history"],
        "question": teacher_question,  # Ensure it's explicitly returned
        "is_casual": False  # Add a flag to indicate it's not a casual message
    }


def crew_nodes(state, crew_member, name):
    question = state['messages'][-1].content  # Retrieve the last generated question

    input_data = {'messages': [AIMessage(content=question)], 'agent_history': state['agent_history']}
   
    result = crew_member.invoke(input_data)  # result is an AIMessage object

    # Print the student's response
    print(f"{name} Response: {result.content}")

    # Add the student's response to the environment using env.add_reply()
    env.add_reply(result.content)

    # Access the content attribute of the AIMessage object
    return {"agent_history": [AIMessage(content=result.content, name=name)]}


# Define the Graph State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  
    agent_history: Annotated[Sequence[BaseMessage], operator.add]  
    question: str 

def count_questions(question_text):
    """Counts the number of questions based on numbering (1., 2., 3., etc.)."""
    return len(re.findall(r"^\d+\.", question_text, re.MULTILINE))


def leaderboard_node(state): 
    responses = state["agent_history"]  # Collect student responses
    question = state.get("question", "")  # Retrieve the stored question
    total_questions = count_questions(question)  # Count the number of questions

    # Load previous state
    saved_state = env.read_file(STATE_FILE)
    state_model = State.model_validate_json(saved_state) if saved_state else State()

    print("Previous Scores:")
    print(f"Student 1: {state_model.student_1_total_score}")
    print(f"Student 2: {state_model.student_2_total_score}")
    print(f"Student 3: {state_model.student_3_total_score}")

    results = []
    feedback_messages = []

    for message in responses:
        student_name = message.name
        answer = message.content

        # Evaluate answer using LLM
        evaluation_prompt = f"""
        Question: {question}
        Student's Answer: {answer}
        Evaluate each answer separately, giving 1 mark per correct answer.
        Format response as: Correct Answers: X/{total_questions}.
        """
        evaluation_response = llm.invoke(evaluation_prompt)  # LLM evaluates the answer
        evaluation_text = evaluation_response.content.strip()

        # Extract correct answers count
        correct_match = re.search(r"Correct Answers:\s*(\d+)/\d+", evaluation_text)
        secured_marks = int(correct_match.group(1)) if correct_match else 0  
        print(f"{student_name} secured marks: {secured_marks}")
        
        # Map student name to state attributes
        if "Student1" in student_name:
            state_model.student_1_total_score += secured_marks
            print(f"Updated Student 1 Total Score: {state_model.student_1_total_score}")
        elif "Student2" in student_name:
            state_model.student_2_total_score += secured_marks
            print(f"Updated Student 2 Total Score: {state_model.student_2_total_score}")
        elif "Student3" in student_name:
            state_model.student_3_total_score += secured_marks
            print(f"Updated Student 3 Total Score: {state_model.student_3_total_score}")

        results.append([student_name, secured_marks])

        # Generate short motivational feedback
        feedback_prompt = f"""
        Student: {student_name}
        Score: {secured_marks}/{total_questions}
        Provide a **short** motivational feedback in 1 sentence.
        """
        feedback_response = llm.invoke(feedback_prompt)
        feedback_text = feedback_response.content.strip()

        feedback_messages.append(f"**{student_name}:** {feedback_text}")

    # Sort results by highest score
    results.sort(key=lambda x: x[1], reverse=True)

    # Log updated scores
    print("Updated Scores:")
    print(f"Student 1: {state_model.student_1_total_score}")
    print(f"Student 2: {state_model.student_2_total_score}")
    print(f"Student 3: {state_model.student_3_total_score}")

    # Save updated state
    env.write_file(STATE_FILE, state_model.model_dump_json())

    # Format leaderboard as a table string
    leaderboard_table = "| Agent Name      | Score (Current) |\n"
    leaderboard_table += "|-----------------|-----------------|\n"
    for student_name, score in results:
        leaderboard_table += f"| {student_name.ljust(15)} | {str(score).center(15)} |\n"

    # Format final response
    feedback_text = "\n".join(feedback_messages)
    final_response = f"🏆 **Leaderboard**\n\n{leaderboard_table}\n\n📢 **Feedback:**\n\n{feedback_text}"

    env.add_reply(final_response)
    return {"messages": [AIMessage(content=final_response)]}
workflow = StateGraph(AgentState)

# Define student nodes
student1_node = functools.partial(crew_nodes, crew_member=student1_agent, name="Student1")
student2_node = functools.partial(crew_nodes, crew_member=student2_agent, name="Student2")
student3_node = functools.partial(crew_nodes, crew_member=student3_agent, name="Student3")

workflow.add_node("Student1", student1_node)
workflow.add_node("Student2", student2_node)
workflow.add_node("Student3", student3_node)

# Teacher Agent responsible for generating a question
workflow.add_node("Teacher", teacher_node)
workflow.set_entry_point("Teacher")

# Leaderboard node
workflow.add_node("Leaderboard", leaderboard_node)

# Conditional edge: If the teacher's response is casual, end the workflow
def route_based_on_casual(state):
    if state.get("is_casual", False):
        return END  # End the workflow if it's a casual message
    else:
        return "Student1"  # Proceed to student nodes if it's a question

workflow.add_conditional_edges(
    "Teacher",
    route_based_on_casual,
    {
        END: END,  # End the workflow if casual
        "Student1": "Student1",  # Proceed to student nodes if it's a question
    }
)

# Make the teacher call all three students simultaneously
workflow.add_edge("Student1", "Student2")
workflow.add_edge("Student2", "Student3")
workflow.add_edge("Student3", "Leaderboard")

# End the workflow after Leaderboard
workflow.add_edge("Leaderboard", END)
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
# Get user input dynamically
# Get user input dynamically
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