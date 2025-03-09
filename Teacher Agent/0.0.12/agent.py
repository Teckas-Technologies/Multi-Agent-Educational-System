import functools
import os, getpass
import operator
from typing import Annotated, Sequence, TypedDict
from enum import Enum
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel  # :white_check_mark: UPDATED import
from langgraph.graph import StateGraph, END, MessagesState
from langchain_openai import ChatOpenAI
import requests
from tabulate import tabulate  # Import for table formatting
import re
import json

os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", None)
llm = ChatOpenAI(model="gpt-4o")


def teacher_node(state: MessagesState):
    """Teacher node responsible for generating questions."""
    print("Current MessagesState:", state)

    # Define the system message for the teacher
    sys_msg = SystemMessage(
        content="""
        You are a teacher generating questions.
        Generate clear and relevant questions based on the subject and quantity requested.
        Do not decide which student should answer; all students will receive the same question.

        Each response must start with:
        - 'Teacher's Question:' if only **one** question is generated.
        - 'Teacher's Questions:' if **multiple** questions are generated.

        For Student Performance Tracking:
        - Each student's total current score is recorded and updated as they answer questions.
        - You can respond to queries about a student's score or overall performance at any time.
        - Example queries:
            - "What is Student1's total score?"
            - "How is Student2 performing overall?"
            - "Show the leaderboard."
        - Always provide accurate and up-to-date information about student performance.
        """
    )

    # Invoke the LLM with the system message and the current messages state
    response = llm.invoke([sys_msg] + state["messages"])
    teacher_question = response.content  # Extract the content of the response
    env.add_message("agent", teacher_question)
    print("Assistant Response:", teacher_question)

    # Check if the response is a casual message (not a question)
    if not teacher_question.startswith(("Teacher's Question:", "Teacher's Questions:")):
        # If it's a casual message, return it and mark it as such
        return {
            "messages": [AIMessage(content=teacher_question)],
           
            "is_casual": True  # Add a flag to indicate casual message
        }
    
    # If it's a question, proceed as normal
    return {
        "messages": [AIMessage(content=teacher_question)], 
        
        "is_casual": False  # Add a flag to indicate it's not a casual message
    }

def student1_node(state: MessagesState):
    # Log the current message state for debugging
    print("Current MessagesState:", state)

    # Define system message instructions for Student 1
    sys_msg = SystemMessage(
        content="""
        You are Student 1 agent, a knowledgeable and precise AI assistant. 
        Your role is only to provide direct, clear, and accurate answers to each question you receive. 
        Keep responses concise and to the point, avoiding unnecessary details or explanations. 
        Focus on delivering factual and well-structured answers in a straightforward manner.

        If you receive **one question**, start your response with 'Student 1 Answer: '.  
        If you receive **multiple questions**, start your response with 'Student 1 Answers: ' 
        and list your answers accordingly.
        """
    )

    # Invoke LLM with the system message and the current messages state
    response = llm.invoke([sys_msg] + state["messages"])
    env.add_message("agent", response.content)
    # Log the full assistant response
    print("Assistant Response", response)

    # Return the response wrapped in a dictionary
    return {"messages": [response]}

def student2_node(state: MessagesState):
    # Log the current message state for debugging
    print("Current MessagesState:", state)

    # Define system message instructions for Student 1
    sys_msg = SystemMessage(
        content="""
        You are Student 2 agent, a knowledgeable and precise AI assistant. 
        Your role is only to provide direct, clear, and accurate answers to each question you receive. 
        Keep responses concise and to the point, avoiding unnecessary details or explanations. 
        Focus on delivering factual and well-structured answers in a straightforward manner.

        If you receive **one question**, start your response with 'Student 2 Answer: '.  
        If you receive **multiple questions**, start your response with 'Student 2 Answers: ' 
        and list your answers accordingly.
        """
    )

    # Invoke LLM with the system message and the current messages state
    response = llm.invoke([sys_msg] + state["messages"])

    # Log the full assistant response
    print("Assistant Response ", response)
    env.add_message("agent", response.content)
    # Return the response wrapped in a dictionary
    return {"messages": [response]}

def student3_node(state: MessagesState):
    # Log the current message state for debugging
    print("Current MessagesState:", state)

    # Define system message instructions for Student 1
    sys_msg = SystemMessage(
        content="""
        You are Student 3 agent, a knowledgeable and precise AI assistant. 
        Your role is only to provide direct, clear, and accurate answers to each question you receive. 
        Keep responses concise and to the point, avoiding unnecessary details or explanations. 
        Focus on delivering factual and well-structured answers in a straightforward manner.

        If you receive **one question**, start your response with 'Student 3 Answer: '.  
        If you receive **multiple questions**, start your response with 'Student 3 Answers: ' 
        and list your answers accordingly.
        """
    )

    # Invoke LLM with the system message and the current messages state
    response = llm.invoke([sys_msg] + state["messages"])
    env.add_message("agent", response.content)
    # Log the full assistant response
    print("Assistant Response", response)

    # Return the response wrapped in a dictionary
    return {"messages": [response]}


def leaderboard_node(state: MessagesState):
    """Leaderboard node to evaluate student responses and calculate scores."""
    print("Current MessagesState:", state)

    sys_msg = SystemMessage(
        content="""
        You are the leaderboard evaluator. Your task is to analyze student responses 
        and calculate their scores based on the number of correct answers.

        Scoring System:
        - Each question carries 1 mark.
        - Count the number of correct answers for each student.
        - Rank students based on their total score.

        Output Format:
        - Rank students from highest to lowest score.
        - Display each student's name, response summary, total correct answers, and total score.
        
        Example Output:

        Leaderboard:
        1. Student_A - Correct Answers: 3, Total Score: 3
        2. Student_B - Correct Answers: 2, Total Score: 2
        3. Student_C - Correct Answers: 1, Total Score: 1

        Now, evaluate the student responses and generate the leaderboard.
        """
    )

    response = llm.invoke([sys_msg] + state["messages"])
    env.add_message("agent", response.content)
    print("Leaderboard Evaluation:", response)
    return {"messages": [response]}

# Define the Graph State
workflow = StateGraph(MessagesState)

# Define teacher node
workflow.add_node("Teacher", teacher_node)
workflow.set_entry_point("Teacher")

# Define student nodes
workflow.add_node("Student1", student1_node)
workflow.add_node("Student2", student2_node)
workflow.add_node("Student3", student3_node)

# Define leaderboard node
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

# Set up workflow edges
workflow.add_edge("Student1", "Student2")
workflow.add_edge("Student2", "Student3")
workflow.add_edge("Student3", "Leaderboard")
workflow.add_edge("Leaderboard", END)

# Compile workflow graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Request user input
user_input = env.list_messages()[-1]["content"]

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
        print(message.content)

    env.request_user_input()