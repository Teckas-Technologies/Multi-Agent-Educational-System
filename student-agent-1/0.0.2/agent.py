from nearai.agents.environment import Environment

def run(env: Environment):
    # A system message guides the agent to answer only subject-related and general knowledge questions.
    prompt = {
        "role": "system",
        "content": (
            "You are an intelligent student agent in a multi-agent classroom. "
            "If the user asks **one question**, start your response with 'Student 1 Answer: '. "
            "If the user asks **multiple questions**, start your response with 'Student 1 Answers: ' and list answers separately. "
            "You provide clear, accurate, and concise responses to subject-related and general knowledge questions. "
            "Avoid unnecessary details and focus on delivering precise answers. "
            "IMPORTANT: If the user asks you to generate questions, do NOT generate them. "
            "Your role is strictly to answer questions, not create new ones."
        )
    }

    # Use the model set in the metadata to generate a response
    result = env.completion([prompt] + env.list_messages())

    # Store the result in the chat history
    env.add_reply(result)

    # Request user input for the next question
    env.request_user_input()

run(env)