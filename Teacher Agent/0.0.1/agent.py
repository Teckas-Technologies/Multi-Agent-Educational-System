from nearai.agents.environment import Environment
from pydantic import BaseModel

STATE_FILE = "teacher_file.json"

REMEMBER_PROMPT = """
You are an expert at deciding if parts of a message should be remembered in the user's memory.

**Instructions:**

- If parts of the message are relevant, respond with **ONLY** the parts that should be remembered.
- If none of the message is relevant, respond with an empty string.
- **Do not** include any additional text, explanations, or greetings.
"""

def to_remember(message: str, env: Environment):
    """Determines if parts of the message should be remembered."""
    res = env.completion(
        [
            {"role": "system", "content": REMEMBER_PROMPT},
            {"role": "user", "content": message},
        ]
    ).strip()
    if res == "" or res == '""' or res == "[]":
        return None
    return res


class State(BaseModel):
    user_query: str
    useful_memories: str
    child_thread_id: str
    first_run_for_query: bool
    ready_for_completions: bool


agents_mapping = {
    "question_generate": {"keywords": ["generate", "questions"], "agent": "luna_grilka.near/teacher-agent/0.0.5"},
    "student_response": {"keywords": ["answer", "response"], "agent": "luna_grilka.near/student-agent/0.0.3"},
}

def initialize_state(env: Environment) -> State:
    state_file = env.read_file(STATE_FILE)
    if not state_file:
        return State(
            user_query="",
            useful_memories="",
            child_thread_id="",
            first_run_for_query=True,
            ready_for_completions=True,
        )
    return State.model_validate_json(state_file)


def process_first_run(state: State, env: Environment):
    user_query = env.list_messages()[-1]["content"]
    state.user_query = user_query

    # If there's something to remember, store it in memory
    remember = to_remember(user_query, env)
    if remember and remember != "":
        env.add_user_memory(remember)
        env.add_system_log(f"Memory updated: {remember}")

    # state.useful_memories = env.query_user_memory(user_query)

    # Check for relevant keywords and call corresponding agents
    for key, info in agents_mapping.items():
        if any(keyword in user_query for keyword in info["keywords"]):
            agent_path = info["agent"]
            namespace, agent_name, version = agent_path.split("/")
            child_thread = env.run_agent(
                namespace, agent_name, version, fork_thread=False
            )
            state.child_thread_id = child_thread
            state.ready_for_completions = False
            break

    state.first_run_for_query = False


def process_completions(state: State, env: Environment):
    previous_messages = env.list_messages()[:-1]

    # Process response from the child thread (student agent)
    child_messages = env.list_messages(state.child_thread_id)

    completion = env.completion(
        messages=[
            {
                "role": "system",
                "content": "You are the ultimate assistant, and you are tasked to always follow what user wants.",
            },
            {
                "role": "user",
                "content": str(previous_messages),
            },
            {
                "role": "system",
                "content": f"Here are relevant user memories for his request:\n{state.useful_memories}",
            },
            {
                "role": "system",
                "content": f"Response from an agent that was required to satisfy users query:\n{str(child_messages)}",
            },
            {"role": "user", "content": state.user_query},
        ]
    )
    env.add_reply(completion)

    state.child_thread_id = ""
    state.first_run_for_query = True
    state.ready_for_completions = True


def task(env: Environment):
    state = initialize_state(env)

    if state.child_thread_id:
        state.ready_for_completions = True

    if state.first_run_for_query:
        process_first_run(state, env)

    if state.ready_for_completions:
        process_completions(state, env)

    # Writing the updated state back to the file
    env.write_file(STATE_FILE, state.model_dump_json())


# Now execute the task with user interaction logic
task(env)