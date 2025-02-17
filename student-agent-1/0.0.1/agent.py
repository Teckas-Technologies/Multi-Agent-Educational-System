from nearai.agents.environment import Environment

def run(env: Environment):
    # A system message guides the agent to answer only subject-related and general knowledge questions.
    prompt = {
        "role": "system",
        "content": (
            "You are an intelligent student agent capable of only answering subject-related and general knowledge questions. "
            "You provide clear, accurate, and concise responses to user questions. Your answers should be fact-based and easy to understand. "
            "Avoid unnecessary details and focus on delivering a precise answer. "
            "IMPORTANT: If the user asks you to generate questions from any subject or specific topics, do NOT generate them. "
            "Your role is strictly to answer your questions, not create new ones."
        )
    }

    # Use the model set in the metadata to generate a response
    result = env.completion([prompt] + env.list_messages())

    # Store the result in the chat history
    env.add_reply(result)

    # Request user input for the next question
    env.request_user_input()

run(env)