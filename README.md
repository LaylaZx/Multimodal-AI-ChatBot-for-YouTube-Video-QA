# 🚀 BandUp: YouTube QA Bot for IELTS Videos
## Project Goal
This project enables real-time question answering for IELTS preparation videos through an AI chatbot that exclusively uses video transcript content. Users provide a YouTube link, and the system:

 - Automatically extracts/subtitles
 - Builds a searchable knowledge base
 - Provides precise answers via a conversational interface

Designed for language learners and educators, it ensures 100% transcript-based responses to maintain factual accuracy for exam preparation.
> **Note:** This project is intended as a shortcut to extract key info from IELTS videos, not a full educational tool.


## Description
BandUp is an AI-powered chatbot designed to help users interactively answer questions about IELTS preparation videos on YouTube. Users simply provide a YouTube video link, and the system automatically extracts the video’s English subtitles or transcribes the audio, processes the content, and builds a searchable knowledge base. The chatbot then allows users to ask natural language questions and receive answers strictly based on the video’s content, ensuring accuracy and relevance for IELTS exam preparation.

This tool is especially valuable for language learners, teachers, and students who want to quickly review or clarify topics covered in IELTS videos without watching the entire content. BandUp leverages cutting-edge AI techniques to deliver a seamless, source-aware Q&A experience, and can be easily adapted to other educational domains with subtitle-enabled videos.

## Key Features
- Strict Source Adherence: Zero external knowledge contamination
- IELTS-Optimized: Specialized for language test preparation content
- Self-Healing Pipeline: Automatic retries for failed transcript chunks
- Portable Design: Works with any YouTube video containing English subtitles


## Architecture Overview
![photo_2025-05-06_10-09-40](https://github.com/user-attachments/assets/c59e278d-500e-417b-b595-be5c97e8cd6f)


## Core Components
### 1. 🎥 Video Processing Pipeline

 - YouTube video downloader (yt-dlp)
 - Audio extraction and chunking (pydub)
 - Whisper-based transcription (transformers)

### 2. 📝 Text Processing

 - Transcript chunking (RecursiveCharacterTextSplitter)
 - OpenAI embeddings for vectorization

### 3. 🔍 Retrieval-Augmented Generation

 - FAISS vector store for efficient similarity search
 - Customized RetrievalQA chain with strict source enforcement

### 4. 💬 Conversational Interface

 - LangChain conversational agent with memory
 - Dual tools: QA and summarization

### 5. 📊 Evaluation System

 - Automated QA pair generation
 - LLM-based accuracy scoring 
 - Faithfulness verification pipeline

## 💡 Why This Architecture?
### Every component was chosen for:

- **Reliability:** Each step (downloading, transcription, chunking, embedding, retrieval) is robust and modular.

- **Transparency:** Strict adherence to source content ensures factuality and trust.

- **Scalability:** FAISS and chunked processing enable handling large video libraries.


## 🧠 Design Rationale & Architecture Logic
### 1. Transcript-Only QA (Strict Source Adherence)
- **Why:** 
Research shows that LLMs can hallucinate or introduce external knowledge. By restricting answers strictly to the video transcript, we guarantee factuality and relevance-crucial for IELTS preparation. Additionally, the goal of the project is to help users extract key information from IELTS videos without needing to watch the entire content, effectively serving as a shortcut tool rather than a full educational bot.

- **How:** 
All QA chains and agent prompts are explicitly instructed to rely only on the transcript. If the answer is not present in the transcript, the system provides a fallback response instead of generating assumptions-ensuring transparency and trustworthiness.

### 2. Chunked Processing & Retrieval-Augmented Generation
- **Why:** 
Long transcripts exceed LLM context windows and reduce retrieval precision. Chunking (using RecursiveCharacterTextSplitter) ensures each vector represents a coherent, retrievable segment.

- **How:** 
Transcripts are split into overlapping chunks, embedded, and indexed with FAISS for fast, accurate retrieval.

### 3. OpenAI Whisper for Transcription
- **Why:** 
Whisper is state-of-the-art for robust,speech-to-text, crucial for diverse YouTube content.

- **How:** 
Audio is automatically split and transcribed, with error handling for failed chunks (self-healing pipeline).

### 4. FAISS Vector Store
- **Why:** 
FAISS offers scalable, efficient similarity search, essential for real-time QA over large transcript corpora.

### 5. LangChain Conversational Agent
- **Why:** 
LangChain’s agent and memory modules allow for multi-turn, context-aware conversations, enabling follow-ups and summaries.

### 6. Automated Evaluation
- **Why:** 
To ensure the system’s answers are faithful and accurate, we use LLM-based QA generation and scoring (see rag_evaluation.py).



## 🛠️ Methodology
### **Building the System**
- **Data Ingestion:** Download YouTube audio using yt-dlp, extract and chunk audio with pydub.

- **Transcription:** Transcribe audio chunks using OpenAI Whisper, handling errors and retries automatically.

- **Text Processing:** Split transcripts into overlapping chunks for embedding.

- **Embedding & Indexing:** Generate OpenAI embeddings for each chunk, store in FAISS for efficient retrieval.

- **Conversational QA:** Use LangChain’s RetrievalQA and agent modules to answer questions and summarize, strictly from the transcript.

- **Evaluation:** Automatically generate QA pairs from the transcript, run predictions, and score accuracy using an LLM.

### Testing the System
> **Note:**
> While formal unit tests were not implemented, the system underwent comprehensive functional testing within the Jupyter notebook. This approach ensured that each pipeline stage-downloading, transcription, chunking, embedding, retrieval, and agent-based QA-was validated in an integrated, end-to-end manner using real IELTS video data.

#### Functional Testing Approach
- **Stepwise Execution:** Each function and pipeline component was executed in Jupyter notebook cells, allowing for immediate inspection of outputs and intermediate results.

- **Manual Verification:** Outputs (such as transcripts, chunked texts, embeddings, and QA responses) were manually reviewed for correctness and quality.

- **Error Handling Validation:** The notebook workflow was used to simulate and handle edge cases, such as failed downloads or transcription errors, confirming the pipeline’s robustness.

- **Conversational QA Validation:** The agent was interactively tested with a variety of questions to ensure it adhered strictly to transcript content and handled out-of-scope queries gracefully.

#### End-to-End Testing: 
- Full pipeline runs on sample IELTS videos with manual and automated evaluation.

#### Faithfulness Checks: 
- LLM-based evaluation ensures answers are always grounded in the transcript.

> **Scientific Rationale**
> - **Why Functional Testing?**  Functional testing in the notebook was chosen to focus on end-to-end system validation, reflecting how real users interact with the pipeline. This approach is particularly effective for complex, multi-stage AI workflows where integration between components is critical.


### Transparency and Reproducibility:
- By documenting all steps and results in the notebook, the testing process is transparent and reproducible for collaborators and future users.

### Future Work
- **Unit Testing:**
For production deployment, adding automated unit tests for individual functions (e.g., chunking, transcription, retrieval) is recommended to catch regressions and ensure maintainability.

- **Multi-Language Support:** Swap Whisper models or add translation for non-English content.
- **???**

## ⚙️ Setup Instructions
1. Clone the Repository

2. Install Dependencies
    - Ensure you have Python 3.10+ installed.
    - Install dependencies using:
       - 'pip install -r requirements.txt'
3. Install FFmpeg
    - BandUp uses pydub for audio processing, which requires FFmpeg.
    - Download and install FFmpeg for your OS from ffmpeg.org.
    - If you are on Windows, you can place the FFmpeg binary in a directory and update the path in the code or environment.
4. Configure FFmpeg Path
    - If FFmpeg is not in your system PATH, update the path in the code (e.g., in rag_pipeline.py):
      - AudioSegment.converter = "path/to/ffm
5. Set Up Environment Variables
    - Create a .env file in the root directory and add your OpenAI API key:
      - OPENAI_API_KEY=your_openai_key_here

## 📬 Questions or Contributions?
Feel free to contact us!
This project is designed for the open-source community and welcomes improvements and adaptations.


