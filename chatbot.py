import os
import streamlit as st
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import Literal
from pydub import AudioSegment
from langchain.schema import AIMessage  # Added import for AIMessage
from langchain_core.tracers.context import tracing_v2_enabled

# Backend imports from the original app_test.py
from rag_pipeline import (
    load_environment,
    download_audio,
    configure_ffmpeg,
    prepare_directories,
    list_audio_files,
    sanitize_basename,
    process_audio_file,
    write_transcript_header,
    split_into_chunks,
    export_chunk,
    transcribe_chunk,
    load_and_split_documents,
    create_vectorstore,
    build_qa_chain,
    append_transcript
)
from agent import build_agent

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# Load custom CSS for styling (bubbles, icons, etc.)
def load_css():
    if os.path.exists("static/styles.css"):
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

# Initialize session state for history, agent, and transcript readiness
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "transcript_ready" not in st.session_state:
        st.session_state.transcript_ready = False

# Process YouTube URL: download, transcribe, build vectorstore and agent
def process_youtube(youtube_url: str):
    api_key, chat_model, client = load_environment()
    ffmpeg_path: str = "ffmpeg"
    chunk_length_ms: int = 10 * 60 * 1000
    video_path = download_audio(youtube_url)
    configure_ffmpeg(ffmpeg_path)

    base_name = sanitize_basename(video_path)
    chunk_folder = os.path.join("chunks", base_name)
    transcripts_dir = os.path.join("transcripts")
    prepare_directories(chunk_folder, transcripts_dir)

    audio_files = list_audio_files("output_audio")
    for audio_path in audio_files:
        name = sanitize_basename(audio_path)
        process_audio_file(audio_path, chunk_folder, transcripts_dir, chunk_length_ms, client)
        transcript_path = os.path.join(transcripts_dir, f"{name}.txt")
        write_transcript_header(transcript_path, name)

        audio = AudioSegment.from_file(audio_path)
        chunks = split_into_chunks(audio, 10 * 60 * 1000)
        for idx, chunk in enumerate(chunks):
            chunk_path = export_chunk(chunk, chunk_folder, name, idx)
            text = transcribe_chunk(chunk_path, client)
            append_transcript(transcript_path, idx, text)

    docs = load_and_split_documents(transcripts_dir)
    vectorstore = create_vectorstore(docs, api_key=api_key)
    chain = build_qa_chain(vectorstore, chat_model, return_source_documents=False)
    st.session_state.agent = build_agent(chat_model=chat_model, chain=chain)
    st.session_state.transcript_ready = True
    st.success("Transcription complete and agent initialized!")

# Handler for the Download button
def on_download_click():
    url = st.session_state.youtube_url
    if url.strip():
        process_youtube(url)

# Handler for chat submission
def on_submit_chat():
    with tracing_v2_enabled():
        user_msg = st.session_state.human_prompt
        st.session_state.history.append(Message("human", user_msg))
        text = ""
        if st.session_state.agent:
            resp = st.session_state.agent.invoke(user_msg)
            if isinstance(resp, dict):
                # Extract final text only
                if "output" in resp:
                    text = resp["output"]
                elif "text" in resp:
                    text = resp["text"]
                elif "chat_history" in resp:
                    # Ensure AIMessage is imported for this
                    for msg in reversed(resp["chat_history"]):
                        if isinstance(msg, AIMessage):
                            text = msg.content
                            break
                else:
                    text = str(resp)
            else:
                text = str(resp)
        else:
            text = "Transcript not loaded yet..."
        st.session_state.history.append(Message("ai", text))

# --- UI Layout ---
load_css()
initialize_session_state()

st.title("Hello Custom CSS YouTube QA Chatbot 🤖")

# YouTube URL form
with st.form("youtube-form"):
    st.markdown("**YouTube Transcription**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Enter YouTube URL", key="youtube_url", label_visibility="collapsed"
    )
    cols[1].form_submit_button(
        "Download & Transcribe", on_click=on_download_click
    )

# Chat container
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
status_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row {'row-reverse' if chat.origin=='human' else ''}">
    <img class="chat-icon" src="static/{'ai_icon.png' if chat.origin=='ai' else 'user_icon.png'}" width=32 height=32>
    <div class="chat-bubble {'ai-bubble' if chat.origin=='ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    for _ in range(3):
        st.markdown("")

# Chat input form
with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Your question", key="human_prompt", label_visibility="collapsed"
    )
    cols[1].form_submit_button(
        "Submit", type="primary", on_click=on_submit_chat,
        disabled=not st.session_state.transcript_ready
    )

# Status indicator
status_placeholder.caption(f"Transcript ready: {st.session_state.transcript_ready}")

# Enable Enter key for submission
components.html(
    """
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') submitButton.click();
});
</script>
    """, height=0, width=0
)
