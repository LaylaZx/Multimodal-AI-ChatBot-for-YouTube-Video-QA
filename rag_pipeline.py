# rag_pipeline.py

import os
import re
import glob
import math
from dotenv import load_dotenv
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
from pydub import AudioSegment
from transformers import pipeline
from langchain.document_loaders import TextLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer


def load_environment():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment.')
    tracer = LangChainTracer()
    callback_manager = CallbackManager([tracer])

    chat_model = ChatOpenAI(
        openai_api_key=api_key, 
        temperature=0, 
        callback_manager=callback_manager,
        verbose=True)
    client = OpenAI()
    return api_key, chat_model, client


import os
import yt_dlp as youtube_dl
from yt_dlp.utils import DownloadError

def download_audio(url: str, output_dir: str = "output_audio") -> str:
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # use the video title as filename, but restrict to safe chars
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "nocheckcertificate": True,
        "quiet": True,
        "restrictfilenames": True
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # ydl.prepare_filename(info) gives the path *before* postprocessing,
            # so it will be something like ".../<safe_title>.webm" (or whatever format).
            downloaded_path = ydl.prepare_filename(info)
    except DownloadError as e:
        print("DownloadError:", e)
        raise

    # swap out whatever original extension for .mp3
    base, _ = os.path.splitext(downloaded_path)
    final_path = base + ".mp3"

    return final_path


# Add the ffmpeg/bin directory to the system PATH
ffmpeg_bin = r"C:\Users\007T\OneDrive\Desktop\ironhack2025\ai-bootcamp-final-project\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin

# Explicitly tell pydub where the ffmpeg executable is
AudioSegment.converter = os.path.join(ffmpeg_bin, "ffmpeg.exe")

# === 1. FFmpeg Configuration ===
def configure_ffmpeg(ffmpeg_path: str):
    """
    Configure pydub to use the specified FFmpeg executable.
    """
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.converter = ffmpeg_path

# === 2. Directory Preparation ===
def prepare_directories(*dirs):
    """
    Ensure that each directory in `dirs` exists.
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

# === 3. File Listing ===
def list_audio_files(input_folder: str) -> list[str]:
    """
    Return a list of all .mp3 file paths under `input_folder`.
    Raises ValueError if none found.
    """
    files = glob.glob(os.path.join(input_folder, "*.mp3"))
    if not files:
        raise ValueError(f"❌ No MP3 files found in {input_folder}")
    return files

# === 4. Filename Sanitization ===
def sanitize_basename(path: str) -> str:
    """
    Given a file path, return a sanitized base name (letters, digits, underscores).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return re.sub(r"_+", "_", safe).strip("_")

# === 5. Chunk Splitting ===
def split_into_chunks(audio: AudioSegment, chunk_length_ms: int) -> list[AudioSegment]:
    """
    Split `audio` into chunks of `chunk_length_ms` milliseconds.
    """
    total = len(audio)
    count = math.ceil(total / chunk_length_ms)
    return [audio[i * chunk_length_ms : min((i + 1) * chunk_length_ms, total)]
            for i in range(count)]

# === 6. Chunk Exporting ===
def export_chunk(chunk: AudioSegment, output_dir: str, base_name: str, index: int) -> str:
    """
    Export a single `chunk` to WAV under `output_dir` named "{base_name}_chunk_{index}.mp3".
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_name}_chunk_{index}.mp3"
    path = os.path.join(output_dir, filename)
    chunk.export(path, format="mp3")
    return path

# === 7. Transcription ===
def transcribe_chunk(chunk_path: str, client) -> str:
    with open(chunk_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        )
    return response.text

# === 8. Transcript File Handling ===
def write_transcript_header(transcript_file: str, base_name: str):
    """
    Write the initial header for a transcript file.
    """
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(f"# Transcript for {base_name}\n\n")

def append_transcript(transcript_file: str, chunk_filename: str, text: str):
    """
    Append a chunk's transcription to the transcript file.
    """
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"## {chunk_filename}\n{text}\n\n")

# === 9. Processing Pipeline for One Audio File ===
def process_audio_file(
    audio_path: str,
    chunk_folder: str,
    transcripts_dir: str,
    chunk_length_ms: int,
    client
):
    """
    Process one audio file: sanitize name, split into chunks, transcribe, and save transcript.
    """
    base_name = sanitize_basename(audio_path)
    print(f"\n🎧 Processing: {base_name}")

    # Prepare the transcript file
    transcript_path = os.path.join(transcripts_dir, f"{base_name}.txt")
    write_transcript_header(transcript_path, base_name)

    # Load audio and split into chunks
    audio = AudioSegment.from_file(audio_path)
    chunks = split_into_chunks(audio, chunk_length_ms)

    # Export and transcribe each chunk
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = export_chunk(chunk, chunk_folder, base_name, idx)
        print(f"🔹 Transcribing chunk {idx}/{len(chunks)}...")
        text = transcribe_chunk(chunk_path, client)
        append_transcript(transcript_path, os.path.basename(chunk_path), text)

    print(f"✅ Done. Transcript saved to: {transcript_path}")

# === 10. Main Orchestration ===
def process_audio(
    input_folder: str = "output_audio",
    chunk_folder: str = "chunks",
    transcripts_dir: str = "transcripts",
    chunk_length_ms: int = 10 * 60 * 1000,
    ffmpeg_path: str = r"C:\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe",
    client=None
):
    """
    Configure environment, prepare directories, and process all audio files in input_folder.
    """
    configure_ffmpeg(ffmpeg_path)
    prepare_directories(chunk_folder, transcripts_dir)
    audio_files = list_audio_files(input_folder)
    for audio_path in audio_files:
        process_audio_file(audio_path, chunk_folder, transcripts_dir, chunk_length_ms, client)


def load_and_split_documents(dirc_path: str = "./transcripts",
                             chunk_size: int = 600,
                             chunk_overlap: int = 100) -> list:
    
    loader = DirectoryLoader(
            dirc_path,
            glob="*.txt",
            loader_cls=TextLoader
        )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '']
    )
    return splitter.split_documents(docs)


def create_vectorstore(docs: list, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002") 
    return FAISS.from_documents(docs, embeddings)


 def build_qa_chain(vector_store, chat_model, return_source_documents) -> RetrievalQA:
# # Custom Prompt Template for Strict QA
#     STRICT_QA_PROMPT = PromptTemplate(
#     template="""You are a ielts assistant. Using the provided video transcript data and tools, answer the question below.
# If you cannot use the data or you did not find anything related in that data that might help you to answer, say "This information is not mentioned in this part of the transcript."

# After your answer, rate how relevant your answer is to the question on a scale from 1 (not relevant) to 5 (very relevant).

# Transcript chunk:
# {context}

# Question:
# {question}

# Answer:

# Relevance score (1-5):""",
#     input_variables=["context", "question"]
# )

    return RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=return_source_documents,
            #chain_type_kwargs={"prompt": STRICT_QA_PROMPT},
            verbose=True)


def answer_query(qa_chain: RetrievalQA, query: str) -> dict:
    return qa_chain({'query': query})


# def main(video_url: str):
#     api_key, chat_model = load_environment()
#     print('▪ Downloading video…')
#     video_path = download_youtube_video(video_url)
#     print('▪ Extracting audio…')
#     audio_path = extract_audio(video_path)
#     print('▪ Splitting audio…')
#     chunks = split_audio(audio_path)
#     print('▪ Transcribing…')
#     texts = transcribe_chunks(chunks)
#     transcript_file = save_transcripts(texts)
#     print('▪ Splitting documents…')
#     docs_chunks = load_and_split_documents(transcript_file)
#     print('▪ Creating vectorstore…')
#     vs = create_vectorstore(docs_chunks, api_key)
#     print('▪ Building QA chain…')
#     qa = build_qa_chain(vs, chat_model)

#     while True:
#         q = input('\\nYour question (type \"exit\" to quit): ')
#         if q.lower().startswith('exit'):
#             break
#         res = answer_query(qa, q)
#         print(f"\\nAnswer:\\n{res['result']}\"")
#         print('\\nSources:')
#         for doc in res.get('source_documents', []):
#             print('- ', doc.metadata.get('source', 'unknown'))


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='RAG Pipeline CLI')
#     parser.add_argument('url', help='YouTube video URL')
#     args = parser.parse_args()
#     main(args.url)
