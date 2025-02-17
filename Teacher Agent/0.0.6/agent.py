from nearai.agents.environment import Environment
from pydantic import BaseModel
import time
import re
STATE_FILE = "teacher_file.json"

class State(BaseModel):
    questions: str = ""
    child_thread_id: str = ""
    student_1_answers: str = ""
    student_2_answers: str = ""
    student_1_total_score: int = 0  # Cumulative Score for Student 1
    student_2_total_score: int = 0  # Cumulative Score for Student 2


def initialize_state(env: Environment) -> State:
    """Always initialize a new state with fresh values."""
    print("Initializing new state (ignoring saved state)...")
    return State()  # Always return a fresh state instance


def generate_questions(state: State,env: Environment) -> str:
    """Generate questions using the teacher agent with detailed logging."""
    
    print(" Starting generate_questions function...")

    # Retrieve conversation history in descending order (latest first)
    message_history = env.list_messages(order="desc",limit=100000000000000)
    print(f"Retrieved Message History: {message_history}")

    # Extract the first message's content
    if message_history:
        first_message_content = message_history[0].get("content", "")
        print(f" First Message Content: {first_message_content}")
    else:
        first_message_content = "No prior user messages found."
        print(f" No message history found.")

    # Define the prompt for the teacher agent
    prompt = {
       "role": "teacher",
    "content": f"You are a teacher agent in a virtual classroom. Your sole responsibility is to guide users through interactive learning by "
               f"generating questions. When a user selects a subject and specifies the number of questions, you generate and present that exact number of "
               f"questions. Each response must start with:\n\n"
               
               f"- 'Teacher's Question:' if only **one** question is generated.\n"
               f"- 'Teacher's Questions:' if **multiple** questions are generated.\n\n"

               f"Each question must start with a number followed by a period (e.g., '1. ', '2. ', '3. ') and must not include 'Question 1:' or any other prefixes like 'Puzzle 1:'. "
               f"Ensure that all questions are **unique** and do not repeat. You are an AI teacher, so you must generate **different** questions every time.\n\n"

               f"**Important Restriction:** If the user asks for an explanation of a topic, you must **not** provide an answer. Instead, your response should be: "
               f"'I am a teacher agent. My purpose is to generate questions for learning, not to provide explanations or answers. Please attempt the questions to deepen your understanding.' "
               f"You must strictly follow this rule in all cases.\n\n"

               f"After presenting the questions, guide the user to choose the next chapter or topic to continue their learning journey. Always focus on generating questions "
               f"in the specified format and facilitating topic progression.\n\n"

               f"Additionally, you are responsible for tracking and remembering student performance throughout the learning session. Each student's total current "
               f"score is recorded and updated as they answer questions. You can respond to queries about a student's score at any time.\n\n"

               f"**Student Performance Tracking:**\n"
               f"- Student 1 Score: {state.student_1_total_score}\n"
               f"- Student 2 Score: {state.student_2_total_score}\n"
               f"(Additional students can be tracked as needed.)\n\n"

               f"**If a user asks: 'Give me an update on all students' scores or overall performance review ', provide the response in a table format:**\n\n"
               
               f"| Student Name | Total Score | Rank |\n"
               f"|--------------|------------|------|\n"
               f"| Student 1    | {state.student_1_total_score} | X |\n"
               f"| Student 2    | {state.student_2_total_score} | Y |\n"

               f"- Rank students based on their total scores. The highest score gets Rank 1, the second highest Rank 2, and so on.\n"
               f"- If students have the same score, they should have the same rank.\n"
               f"- Do not provide any unnecessary text—only return the formatted table.\n\n"

               f"**If a user asks about the top-performing student, respond with:**\n"
               f"- 'The best-performing student is [Student Name] with a total score of [Score]. Keep it up!'\n"
               f"- If multiple students have the same highest score, say: 'Both [Student 1] and [Student 2] are leading with [Score]. Keep competing!'\n\n"

               f"Encourage students to improve their performance, engage actively, and compete in a friendly manner to enhance their learning experience."


    }


    print(f" Created Prompt for Teacher Agent: {prompt}")

    # Convert first message content into a message dictionary
    first_message_dict = {"role": "user", "content": first_message_content}

    # Generate questions using the teacher agent
    result = env.completion([prompt, first_message_dict])
    print(f"Generated Questions by Teacher Agent:\n{result}")

    # Store the generated questions in the environment
    env.add_message("agent", result)

    
    return result

def call_student_agents(state: State, env: Environment, questions: str) -> None:
    """Call multiple student agents with the generated questions and store a single thread ID in state."""
    print("Sending questions to multiple Student Agents...")

    student_agents = [
        ("luna_grilka.near", "student-agent-1", "0.0.2"),
        ("luna_grilka.near", "student-agent-2", "0.0.1"),  # Replace with actual second agent
    ]

    student_thread_id = ""  # Initialize as an empty string

    for i, (namespace, agent_name, version) in enumerate(student_agents, start=1):
        print(f"Calling {agent_name} ({namespace}, version {version})...")

        # Modify the query to specify the student number
        student_query = f"Answer the following questions: {questions}"

        # Run agent and get thread ID (assumed to be the same for both)
        student_thread_id = env.run_agent(
            namespace,
            agent_name,
            version,
            query=student_query,
            fork_thread=False
        )

        print(f"{agent_name} Thread ID: {student_thread_id}")

    # Store only one thread ID in state
    state.child_thread_id = student_thread_id
    print(f"Stored Child Thread ID in State: {state.child_thread_id}")





def extract_student_answers(state: State, messages) -> tuple:
    """Extract the most recent student answers from messages and store them in the state."""
    
    student_1_answers, student_2_answers = "", ""
    found_student_1, found_student_2 = False, False  # Track whether answers were found

    print("Extracting Student Answers from Messages...")

    for msg in messages:  # Iterate in reverse to get the latest response first
        if msg["role"] == "assistant":
            content = msg["content"]

            # Check if the message contains Student 1's answer and hasn't been set yet
            if not found_student_1 and ("Student 1 Answer" in content or "Student 1 Answers" in content):
                student_1_answers = content.split("Student 1 Answer")[-1].strip()
                state.student_1_answers = student_1_answers  # Store in state
                found_student_1 = True  # Mark as found
                print(f"Stored Student 1 Answer in State ----- {state.student_1_answers}")

            # Check if the message contains Student 2's answer and hasn't been set yet
            if not found_student_2 and ("Student 2 Answer" in content or "Student 2 Answers" in content):
                student_2_answers = content.split("Student 2 Answer")[-1].strip()
                state.student_2_answers = student_2_answers  # Store in state
                found_student_2 = True  # Mark as found
                print(f"Stored Student 2 Answer in State ---- {state.student_2_answers}")

            # Stop searching once both answers are found
            if found_student_1 and found_student_2:
                break

    # If no structured answers were found, print a warning
    if not found_student_1 and not found_student_2:
        print("\n No structured student answers found.\n")

    return student_1_answers, student_2_answers


def evaluate_answers(env: Environment, state: State) -> str:
    print("Evaluating answers...")

    if not state.student_1_answers.strip() and not state.student_2_answers.strip():
        print("No answers provided.")
        return "No answers provided."

    total_questions = state.questions.strip().count("\n") + 1

    eval_prompt = f"""
    Evaluate the following student answers. Assign 1 mark per correct answer.
    Provide the response strictly in **table format** as shown below.

    Questions:
    {state.questions}

    Student 1 Answers:
    {state.student_1_answers}

    Student 2 Answers:
    {state.student_2_answers}

    **Format response strictly as:**
    
    | AgentName   | TotalQuestions | TotalScore | Performance Rank |
    |------------|---------------|------------|-----------------|
    | Student 1  | {total_questions}         | X          | Rank 1/2/3 |
    | Student 2  | {total_questions}         | Y          | Rank 1/2/3 |

    Do not add any additional text. Only return the table.
    """

    evaluation_result = env.completion([{"role": "system", "content": eval_prompt}])

    env.add_message("agent", evaluation_result)

    match_1 = re.search(r"Student 1\s+\|\s+\d+\s+\|\s+(\d+)\s+\|", evaluation_result)
    match_2 = re.search(r"Student 2\s+\|\s+\d+\s+\|\s+(\d+)\s+\|", evaluation_result)

    student_1_score = int(match_1.group(1)) if match_1 else 0
    student_2_score = int(match_2.group(1)) if match_2 else 0

    # Update cumulative scores
    state.student_1_total_score += student_1_score
    state.student_2_total_score += student_2_score

    print(f"Updated Student 1 Total Score: {state.student_1_total_score}")
    print(f"Updated Student 2 Total Score: {state.student_2_total_score}")

    return evaluation_result

def task(env: Environment):
    print("Starting task execution...")

    try:
        saved_state = env.read_file(STATE_FILE)
        state = State.model_validate_json(saved_state) if saved_state else initialize_state(env)
    except Exception as e:
        print(f"Error loading saved state: {e}")
        state = initialize_state(env)

    print(f"Loaded State: {state}")

    generated_questions = generate_questions(state, env)
    if not generated_questions.strip():
        print("No questions were generated. Stopping execution.")
        return

    lines = generated_questions.split("\n")

    # Check if the first line starts with "Teacher's Questions:"
    if not lines[0].strip().startswith(("Teacher's Questions:", "Teacher's Question:")):
        print("Missing 'Teacher's Questions:' header. Stopping execution.")
        return

    # Extract numbered questions (ignores non-numbered lines)
    numbered_questions = [
        line.strip()
        for line in lines[1:]  # Skip the "Teacher's Questions:" header
        if re.match(r"^\d+\.", line.strip())  # Matches any number followed by a dot
    ]

    # Validate extracted questions
    if not numbered_questions:
        print("No valid numbered questions found. Stopping execution.")
        return

    # Store validated questions
    state.questions = "\n".join(numbered_questions)

    call_student_agents(state, env, state.questions)

    print("Waiting 5 seconds for student responses...")
    time.sleep(10)

    list_msg = env.list_messages(order="desc", limit=100000000000000)
    print(f"Messages after: {list_msg}")
    extract_student_answers(state, list_msg)

    print("\n----- Evaluating Answers -----\n")
    evaluate_answers(env, state)

    env.write_file(STATE_FILE, state.model_dump_json())


task(env)