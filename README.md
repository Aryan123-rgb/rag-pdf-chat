 # AI PDF Chat 📚💬

A Streamlit-based application that allows users to chat with their PDF documents using Google's Gemini AI. The app uses RAG (Retrieval Augmented Generation) to provide accurate, context-aware responses to questions about PDF content.

## Features

- 📄 PDF text extraction and processing
- 🔍 Advanced text chunking with overlap for better context retention
- 💾 Local vector store persistence using Chroma DB
- 🤖 Powered by Google's Gemini 1.5 Pro AI model
- 📱 User-friendly Streamlit interface

## Prerequisites

- Python 3.8+
- Google AI API key

## Installation

1. Clone the repository: 
```bash
git clone <repository-url>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google AI API key:
```
API_KEY=your_api_key_here
```

## Required Dependencies

- streamlit
- PyPDF2
- langchain
- langchain-google-genai
- google-generativeai
- python-dotenv
- chromadb

## Usage

1. Run the application:
```bash
streamlit run pdf-chat.py
```

2. Upload a PDF file using the sidebar
3. Click 'Submit' to process the PDF
4. Ask questions about the PDF content in the chat interface

## How It Works

1. **PDF Processing**: The app extracts text from uploaded PDFs and splits it into manageable chunks
2. **Vector Storage**: Text chunks are converted to embeddings and stored in a local Chroma database
3. **RAG Implementation**: Uses Retrieval Augmented Generation to:
   - Convert user questions into vector embeddings
   - Retrieve relevant context from the stored vectors
   - Generate accurate answers using Gemini AI

## Features in Detail

- **Persistent Storage**: Vector embeddings are stored locally for faster subsequent queries
- **Chunk Optimization**: Uses RecursiveCharacterTextSplitter with overlap for better context
- **Error Handling**: Robust error handling for PDF processing and text extraction
- **User-Friendly Interface**: Clean Streamlit UI with clear feedback messages

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your chosen license here]