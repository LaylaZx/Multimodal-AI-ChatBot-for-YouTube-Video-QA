import os
import uuid
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Import RAG pipeline utilities
from rag_pipeline import load_environment, load_and_split_documents, create_vectorstore, build_qa_chain


# Load environment and initialize LLM
api_key, chat_model = load_environment()

# Prepare the transcript chunks and vector store
TRANSCRIPT_FILE = 'transcript.txt'  
docs = load_and_split_documents(TRANSCRIPT_FILE)
vectorstore = create_vectorstore(docs)
chain = build_qa_chain(vectorstore, chat_model, False)

# Define the QA and summarization tools

def run_TranscriptQA(question: str) -> str:
    output = chain({'query': question})
    answer = output.get('result', '')
    sources = "\n\n".join(
        f"{doc.metadata.get('source', 'unknown')}: {doc.page_content}"
        for doc in output.get('source_documents', [])
    )
    if not answer.strip():
        return "Sorry, this information is not in the video transcript."
    return f"Answer:\n{answer}\n\nSources:\n{sources}"


def summarize_transcript(text: str) -> str:
    output = chain({'query': text})
    answer = output.get('result', '')
    sources = "\n\n".join(doc.page_content for doc in output.get('source_documents', []))
    return f"Answer:\n{answer}\n\nSources:\n{sources}"

tools = [
    Tool(
        name="TranscriptQA",
        func=run_TranscriptQA,
        description="Answer questions based solely on the provided video transcript.",
        return_direct=True
    ),
    Tool(
        name="TranscriptSummarizer",
        func=summarize_transcript,
        description="Generate a concise summary of a given transcript text.",
        return_direct=True
    ),
]

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Build the conversational agent
# I initialize the agent with my tools, the LLM, and the memory buffer
agent = initialize_agent(
    tools=tools,
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
    


if __name__ == '__main__':
    print("Agent is ready! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ('exit', 'quit'):
            break
        result = agent.invoke(query)
        print(result)
