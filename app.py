import os
import glob
import yt_dlp
from pydub import AudioSegment
import streamlit as st
from dataclasses import dataclass
from typing import Literal
import openai

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# --- Setup OpenAI key ---
openai.api_key = st.secrets["openai_api_key"]

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def load_css():
    if os.path.exists("static/styles.css"):
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

def download_audio(youtube_url, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        mp3_file = os.path.splitext(filename)[0] + '.mp3'
    return mp3_file

def chunk_audio(mp3_file, chunk_length_ms=20*60*1000):
    audio = AudioSegment.from_file(mp3_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_filename = f"{mp3_file}_chunk_{i//chunk_length_ms}.mp3"
        chunk.export(chunk_filename, format="mp3")
        chunks.append(chunk_filename)
    return chunks

def transcribe_chunk_with_openai(chunk_file):
    with open(chunk_file, "rb") as audio_file:
        transcript_response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
        return transcript_response["text"]

def transcribe_chunks(chunk_files):
    transcript = ""
    for i, chunk_file in enumerate(chunk_files):
        st.info(f"Transcribing chunk {i+1}/{len(chunk_files)}...")
        text = transcribe_chunk_with_openai(chunk_file)
        transcript += text + "\n"
    return transcript

def save_transcript(text, path="files/transcripts/transcript.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def initialize_agent_with_transcript(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
    vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=st.secrets["openai_api_key"], temperature=0),
        retriever=retriever,
        return_source_documents=False,
    )
    tools = [
        Tool(
            name="IELTS Video QA",
            func=qa_chain.run,
            description="Answers questions about the IELTS/English video transcript."
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools,
        ChatOpenAI(openai_api_key=st.secrets["openai_api_key"]),
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=False,
    )
    return agent

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "transcript_ready" not in st.session_state:
        st.session_state.transcript_ready = False

def on_submit_chat():
    user_msg = st.session_state.human_prompt
    st.session_state.history.append(Message("human", user_msg))
    if st.session_state.agent:
        response = st.session_state.agent.run(user_msg)
        st.session_state.history.append(Message("ai", response))
    else:
        st.session_state.history.append(Message("ai", "Transcript not loaded yet. Please process a YouTube URL first."))

# --- Streamlit UI ---
load_css()
initialize_session_state()

st.title("IELTS/English YouTube Video Chatbot with OpenAI Whisper API")

with st.form("youtube_form"):
    youtube_url = st.text_input("Enter YouTube URL (IELTS/English topic)", "")
    process_button = st.form_submit_button("Download & Transcribe")

    if process_button and youtube_url.strip():
        with st.spinner("Downloading audio..."):
            audio_file = download_audio(youtube_url)
        with st.spinner("Chunking audio..."):
            chunks = chunk_audio(audio_file)
        with st.spinner("Transcribing audio chunks with OpenAI Whisper API..."):
            transcript_text = transcribe_chunks(chunks)
        save_transcript(transcript_text)
        st.success("Transcription complete!")
        st.session_state.agent = initialize_agent_with_transcript("files/transcripts/transcript.txt")
        st.session_state.transcript_ready = True

chat_container = st.container()
with chat_container:
    for chat in st.session_state.history:
        align_class = "" if chat.origin == "ai" else "row-reverse"
        icon = "ai_icon.png" if chat.origin == "ai" else "user_icon.png"
        bubble_class = "ai-bubble" if chat.origin == "ai" else "human-bubble"
        div = f"""
        <div class="chat-row {align_class}">
            <img class="chat-icon" src="app/static/{icon}" width=32 height=32>
            <div class="chat-bubble {bubble_class}">&#8203;{chat.message}</div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

with st.form("chat_form"):
    st.markdown("**Chat with the video transcript:**")
    cols = st.columns([6, 1])
    cols[0].text_input(
        "Your question",
        value="What tips does the video give for IELTS listening?",
        key="human_prompt",
        label_visibility="collapsed",
    )
    cols[1].form_submit_button(
        "Send",
        on_click=on_submit_chat,
        disabled=not st.session_state.transcript_ready,
        type="primary",
    )
