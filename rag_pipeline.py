# rag_pipeline.py

import os
import glob
from dotenv import load_dotenv
import yt_dlp as youtube_dl
from pydub import AudioSegment
from transformers import pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


def load_environment():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment.')
    chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0)
    return api_key, chat_model


def download_youtube_video(url: str, output_dir: str = 'videos') -> str:
    os.makedirs(output_dir, exist_ok=True)
    opts = {'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s')}
    with youtube_dl.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def extract_audio(video_path: str, audio_path: str = None) -> str:
    if audio_path is None:
        base, _ = os.path.splitext(video_path)
        audio_path = f'{base}.wav'
    AudioSegment.from_file(video_path).export(audio_path, format='mp3')
    return audio_path


def split_audio(audio_path: str,
                chunk_length_ms: int = 60_000,
                output_dir: str = 'audio_chunks') -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(audio_path)
    paths = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        p = os.path.join(output_dir, f'chunk_{i}.wav')
        chunk.export(p, format='wav')
        paths.append(p)
    return paths


def transcribe_chunks(chunk_paths: list[str],
                      model_name: str = 'openai/whisper-small') -> list[str]:
    
    recognizer = pipeline('automatic-speech-recognition', model=model_name)
    return [recognizer(p)['text'] for p in chunk_paths]


def save_transcripts(transcripts: list[str],
                     output_path: str = 'transcript.txt') -> str:
   
    with open(output_path, 'w', encoding='utf-8') as f:
        for t in transcripts:
            f.write(t + '\n')
    return output_path


def load_and_split_documents(file_path: str,
                             chunk_size: int = 1000,
                             chunk_overlap: int = 100) -> list:
    
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '']
    )
    return splitter.split_documents(docs)


def create_vectorstore(docs: list):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 
    return FAISS.from_documents(docs, embeddings)


def build_qa_chain(vector_store, chat_model, STRICT_QA_PROMPT, return_source_documents) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": STRICT_QA_PROMPT},
            verbose=True)


def answer_query(qa_chain: RetrievalQA, query: str) -> dict:
    return qa_chain({'query': query})


def main(video_url: str):
    api_key, chat_model = load_environment()
    print('▪ Downloading video…')
    video_path = download_youtube_video(video_url)
    print('▪ Extracting audio…')
    audio_path = extract_audio(video_path)
    print('▪ Splitting audio…')
    chunks = split_audio(audio_path)
    print('▪ Transcribing…')
    texts = transcribe_chunks(chunks)
    transcript_file = save_transcripts(texts)
    print('▪ Splitting documents…')
    docs_chunks = load_and_split_documents(transcript_file)
    print('▪ Creating vectorstore…')
    vs = create_vectorstore(docs_chunks, api_key, store_type='faiss', persist_dir='faiss_index')
    print('▪ Building QA chain…')
    qa = build_qa_chain(vs, chat_model)

    while True:
        q = input('\\nYour question (type \"exit\" to quit): ')
        if q.lower().startswith('exit'):
            break
        res = answer_query(qa, q)
        print(f"\\nAnswer:\\n{res['result']}\"")
        print('\\nSources:')
        for doc in res.get('source_documents', []):
            print('- ', doc.metadata.get('source', 'unknown'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RAG Pipeline CLI')
    parser.add_argument('url', help='YouTube video URL')
    args = parser.parse_args()
    main(args.url)
