from nearai.agents.environment import Environment
from pydantic import BaseModel

STATE_FILE = "assistant_state.json"

REMEMBER_PROMPT = """
You are an expert at deciding if parts of a message should be remembered in the user's memory.

**Instructions:**

- If parts of the message are relevant, respond with **ONLY** the parts that should be remembered.
- If none of the message is relevant, respond with an empty string.
- **Do not** include any additional text, explanations, or greetings.
"""

def to_remember(message: str, env: Environment):
    res = env.completion([
        {"role": "system", "content": REMEMBER_PROMPT},
        {"role": "user", "content": message},
    ]).strip()
    
    if res in ["", "\"\"", "[]"]:
        return None
    return res

class State(BaseModel):
    user_query: str
    useful_memories: str
    child_thread_id: str
    first_run_for_query: bool
    ready_for_completions: bool

agents_mapping = {
    "teacher-agent": {"keywords": ["generate", "questions"], "agent": "luna_grilka.near/teacher-agent/0.0.5"},
    "student-agent": {"keywords": ["what", "explain", "how", "clarify", "describe","why","example", "answer", "solve","?"], "agent": "luna_grilka.near/student-agent/0.0.3"},
}

def initialize_state(env: Environment) -> State:
    return State(
        user_query="",
        useful_memories="",
        child_thread_id="",
        first_run_for_query=True,
        ready_for_completions=True,
    )

def process_first_run(state: State, env: Environment):
    user_query = env.list_messages()[-1]["content"]
    state.user_query = user_query

    remember = to_remember(user_query, env)
    if remember:
        env.add_user_memory(remember)
        env.add_system_log(f"Memory updated: {remember}")
    
    state.useful_memories = env.query_user_memory(user_query)

    for key, info in agents_mapping.items():
        if any(keyword in user_query for keyword in info["keywords"]):
            namespace, agent_name, version = info["agent"].split("/")
            state.child_thread_id = env.run_agent(namespace, agent_name, version, fork_thread=False)
            state.ready_for_completions = False
            break

    state.first_run_for_query = False

def task(env: Environment):
    state = initialize_state(env)
    process_first_run(state, env)
    env.write_file(STATE_FILE, state.model_dump_json())

task(env)