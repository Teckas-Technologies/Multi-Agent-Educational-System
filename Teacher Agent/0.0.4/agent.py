from nearai.agents.environment import Environment
from pydantic import BaseModel

STATE_FILE = "a_file.json"

class State(BaseModel):
    questions: str = ""  # Store generated questions
    child_thread_id: str = ""  # Track student agent thread ID
    student_answers: str = ""  # Store student answers

agents_mapping = {
    "student_response": "luna_grilka.near/student-agent/0.0.5",
}

def initialize_state(env: Environment) -> State:
    """Initialize state from the file or create a new one."""
    print("Initializing state...")
    state_file = env.read_file(STATE_FILE)
    if not state_file:
        print("No existing state file found. Initializing new state.")
        return State()
    print(f"Loaded state from file: {state_file}")
    return State.model_validate_json(state_file)

def generate_questions(env: Environment) -> str:
    """Generate questions using the teacher agent."""
    print("Requesting question generation from Teacher Agent...")
    prompt = {
        "role": "system",
        "content": "You are a teacher agent in a virtual classroom. Your sole responsibility is to guide users through interactive learning by \
generating questions. When a user selects a subject and specifies the number of questions, you generate and present that exact number of \
questions. You encourage the user to reflect on the answers and gain knowledge independently. You must not provide explanations, answers, \
or additional context for the questions, even if requested. After presenting the questions, guide the user to choose the next chapter or topic \
to continue their learning journey. Always focus on generating questions and facilitating topic progression."
    }
    
    result = env.completion([prompt] + env.list_messages())
    env.add_message("agent", result)
    
    print(f"Generated Questions by Teacher:\n{result}")
    return result

def extract_questions(messages) -> str:
    """Extract structured questions from messages."""
    for msg in reversed(messages):
        if any(line.strip().startswith(f"{i}.") for i in range(1, 11) for line in msg['content'].split('\n')):
            print(f"Extracted Questions from Messages:\n{msg['content']}")
            return msg["content"]  # Return entire message containing questions
    return ""

def call_student_agent(env: Environment, questions: str) -> str:
    """Call the student agent with the generated questions."""
    print("Sending questions to Student Agent...")
    student_thread = env.run_agent(
         "luna_grilka.near",  # Replace with actual namespace
         "student-agent", 
            "0.0.3", 
        query=questions,
        fork_thread=False
    )
    print(f"Student Agent Thread ID: {student_thread}")
    child_messages = env.list_messages(student_thread)
    print(f"Messages in student agent thread >>>>>>>>>>>>>>>>>>> {child_messages}")

    return student_thread

def extract_student_answers(messages) -> str:
    """Extract student answers from messages."""
    for msg in reversed(messages):
        if msg["role"] == "system":
            print(f"Extracted Student Answer:\n{msg['content']}")
            return msg["content"]  # Return the first found student answer
    return ""

def task(env: Environment):
    print("Starting task execution...")
    state = initialize_state(env)

    print("Generating questions...")
    generated_questions = generate_questions(env)

    # Extract and store questions
    extracted_questions = extract_questions(env.list_messages())
    state.questions = extracted_questions

    # Call student agent
    if extracted_questions:
        print("Passing questions to student agent...")
        state.child_thread_id = call_student_agent(env, extracted_questions)

    # Wait for student response (optional: add a delay if needed)
    print("Waiting for student response...")

    # Extract and store student answers
    student_answers = extract_student_answers(env.list_messages())
    state.student_answers = student_answers

    print(f"Final Stored State:\nQuestions:\n{state.questions}\nStudent Answers:\n{state.student_answers}")

    # Save updated state
    env.write_file(STATE_FILE, state.model_dump_json())
    print("Task execution completed.")

task(env)