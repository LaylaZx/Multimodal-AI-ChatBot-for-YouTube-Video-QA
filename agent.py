
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Define the QA and summarization tools
def run_TranscriptQA(question: str,chain) -> str:
    output = chain({'query': question})
    answer = output.get('result', '')
    sources = "\n\n".join(
        f"{doc.metadata.get('source', 'unknown')}: {doc.page_content}"
        for doc in output.get('source_documents', [])
    )
    if not answer.strip():
        return "Sorry, this information is not in the video transcript."
    return f"Answer:\n{answer}\n\nSources:\n{sources}"


def summarize_transcript(text: str,chain) -> str:
    output = chain({'query': text})
    answer = output.get('result', '')
    sources = "\n\n".join(doc.page_content for doc in output.get('source_documents', []))
    return f"Answer:\n{answer}\n\nSources:\n{sources}"


def generate_tools(chain):
    tools = [
        Tool(
            name="TranscriptQA",
            func=lambda question: run_TranscriptQA(question, chain),
            description="Answer questions based on the provided video transcript.",
            return_direct=True
        ),
        Tool(
            name="TranscriptSummarizer",
            func=lambda text: summarize_transcript(text, chain),
            description="Generate a concise summary of a given transcript text.",
            return_direct=True
        ),
    ]

    return tools

def build_agent(chat_model, chain):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Build the conversational agent
    # I initialize the agent with my tools, the LLM, and the memory buffer
    agent = initialize_agent(
        tools=generate_tools(chain),
        llm=chat_model,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION , #CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        memory=memory,
        handle_parsing_errors=True,
        verbose=True,
        early_stopping_method="generate",
        max_iterations=4,
        return_intermediate_steps=True,
        agent_kwargs={
            "prefix": """
    You are a ann English /IELTS expert. Use this transcript to answer questions, never answer based on your own knowledge.:

    {transcript_chunks}

    Always respond with 'Final Answer: <your answer>'.
    Human: {input}
    """}
    )
    
    return agent 


# if __name__ == '__main__':
#     print("Agent is ready! Type 'exit' to quit.")
#     while True:
#         query = input("You: ")
#         if query.lower() in ('exit', 'quit'):
#             break
#         agent = build_agent()
#         result = agent.invoke(query)
#         print(result)
