import os
import streamlit as st
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import Literal
from pydub import AudioSegment
from langchain.schema import AIMessage  # Added import for AIMessage

# Backend imports
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

# Load custom CSS
def load_css():
    if os.path.exists("./static/styles.css"):
        with open("./static/styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "transcript_ready" not in st.session_state:
        st.session_state.transcript_ready = False

# Process YouTube URL pipeline
def process_youtube(youtube_url: str):
    api_key, chat_model, client = load_environment()
    ffmpeg_path: str = "ffmpeg"
    video_path = download_audio(youtube_url)
    chunk_length_ms: int = 10 * 60 * 1000
    configure_ffmpeg(ffmpeg_path)

    base_name = sanitize_basename(video_path)
    chunk_folder = os.path.join("chunks", base_name)
    transcripts_dir = os.path.join("transcripts")
    prepare_directories(chunk_folder, transcripts_dir)

    for audio_path in list_audio_files("output_audio"):
        name = sanitize_basename(audio_path)
        process_audio_file(audio_path,chunk_folder,transcripts_dir,chunk_length_ms,client)
        transcript_path = os.path.join(transcripts_dir, f"{name}.txt")
        write_transcript_header(transcript_path, name)

        audio = AudioSegment.from_file(audio_path)
        for idx, chunk in enumerate(split_into_chunks(audio, 10 * 60 * 1000)):
            chunk_path = export_chunk(chunk, chunk_folder, name, idx)
            text = transcribe_chunk(chunk_path, client)
            append_transcript(transcript_path, idx, text)

    docs = load_and_split_documents(transcripts_dir)
    vectorstore = create_vectorstore(docs, api_key=api_key)
    chain = build_qa_chain(vectorstore, chat_model, return_source_documents=False)
    st.session_state.agent = build_agent(chat_model=chat_model, chain=chain)
    st.session_state.transcript_ready = True
    st.success("🎉 Video processed! BandUp is live, ask me anything about this video and get answers instantly!")


# Chat submission handler
def on_submit_chat():
    user_msg = st.session_state.human_prompt
    st.session_state.history.append(Message("human", user_msg))
    text = ""

    if st.session_state.agent:
        resp = st.session_state.agent.invoke(user_msg)
        
        # Extract raw text from response
        if isinstance(resp, dict):
            if "output" in resp:
                raw_text = resp["output"]
            elif "text" in resp:
                raw_text = resp["text"]
            elif "chat_history" in resp:
                raw_text = ""
                for msg in reversed(resp["chat_history"]):
                    if isinstance(msg, AIMessage):
                        raw_text = msg.content
                        break
            else:
                raw_text = str(resp)
        else:
            raw_text = str(resp)
        
        # Clean the text: remove "Answer:" and anything after "Sources:"
        if raw_text.startswith("Answer:"):
            raw_text = raw_text[len("Answer:"):].strip()
        if "Sources:" in raw_text:
            raw_text = raw_text.split("Sources:")[0].strip()

        text = raw_text
    else:
        text = "Transcript not loaded yet..."

    st.session_state.history.append(Message("ai", text))


# --- UI ---
load_css()
initialize_session_state()

st.title("🚀 Let's BandUP !")
st.markdown("✨BandUP! Your IELTS Assistant! 🤩" )
st.markdown("Don't watch the whole thing!  just drop your IELTS video and ask me 😉")
# YouTube URL form with spinner below input
with st.form("youtube-form"):
    st.markdown("**YouTube Transcription**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Enter YouTube URL", key="youtube_url", label_visibility="collapsed"
    )
    upload_clicked = cols[1].form_submit_button("Upload")

# Spinner shown under the form upon upload click
if upload_clicked and st.session_state.youtube_url.strip():
    with st.spinner("Video is being processed..."):
        process_youtube(st.session_state.youtube_url)

# Chat container
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
status_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row {'row-reverse' if chat.origin=='human' else ''}">
    <img class="chat-icon" src="./static/{'ai_icon.png' if chat.origin=='ai' else 'user_icon.png'}" width=32 height=32>
    <div class="chat-bubble {'ai-bubble' if chat.origin=='ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Chat input form
with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input("Your question", key="human_prompt", label_visibility="collapsed")
    cols[1].form_submit_button(
        "Submit", type="primary", on_click=on_submit_chat,
        disabled=not st.session_state.transcript_ready
    )

# Status indicator
status_placeholder.caption(f"Transcript ready: {st.session_state.transcript_ready}")

# Enable Enter key
components.html(
    """
<script>
const doc = window.parent.document;
const btns = Array.from(doc.querySelectorAll('.stButton > button'));
const submitBtn = btns.find(el => el.innerText === 'Submit');
doc.addEventListener('keydown', e => e.key === 'Enter' && submitBtn.click());
</script>
    """, height=0, width=0
)
