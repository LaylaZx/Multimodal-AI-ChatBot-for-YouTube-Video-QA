# BandUp: YouTube QA Bot for IELTS Videos
## Project Goal
This project enables real-time question answering for IELTS preparation videos through an AI chatbot that exclusively uses video transcript content. Users provide a YouTube link, and the system:

 - Automatically extracts/subtitles

 - Builds a searchable knowledge base

 - Provides precise answers via a conversational interface

Designed for language learners and educators, it ensures 100% transcript-based responses to maintain factual accuracy for exam preparation.

## Architecture Overview
![RAG Architecture](https://miro.medium.com/v2/resize:fit:1400/1*9EBJ4y4avBQ6wq6Wn8X3Xg.pngAG workflow diagram*

## Core Components
1. Video Processing Pipeline

YouTube video downloader (yt-dlp)

Audio extraction and chunking (pydub)

Whisper-based transcription (transformers)

2. Text Processing

Transcript chunking (RecursiveCharacterTextSplitter)

OpenAI embeddings for vectorization

3. Retrieval-Augmented Generation

FAISS vector store for efficient similarity search

Customized RetrievalQA chain with strict source enforcement

4. Conversational Interface

LangChain conversational agent with memory

Dual tools: QA and summarization

5. Evaluation System

Automated QA pair generation

LLM-based accuracy scoring

Faithfulness verification pipeline
