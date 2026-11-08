# 🤖 Dialogix – HR Policy Chatbot using RAG

An AI-powered chatbot that helps employees find answers to HR-related questions from policy documents without manually searching through lengthy PDFs.

I built Dialogix to explore how **Retrieval-Augmented Generation (RAG)** can make internal company documents easier to access. Instead of asking an LLM to answer from its general knowledge, the chatbot first searches the available HR documents and uses the retrieved content to generate an answer.

## ✨ What I Built

- 📄 PDF processing to extract text from HR policy documents.
- 🔍 Text chunking and embedding generation for document indexing.
- 🧠 Semantic search using FAISS / Chroma.
- 📚 Retrieval of relevant policy sections based on user questions.
- 🎯 Reranking to improve the relevance of retrieved results.
- 💬 LLM integration to generate answers using retrieved context.
- 📊 Google Sheets integration to log chatbot responses.

## ⚙️ How It Works

1. 📄 HR policy PDFs are loaded and their text is extracted using PyMuPDF.
2. ✂️ The extracted text is split into smaller chunks.
3. 🧠 Each chunk is converted into embeddings using Hugging Face models.
4. 💾 Embeddings are stored in a local vector database.
5. 🔍 When a user asks a question, the system searches for relevant chunks.
6. 🤖 The retrieved content is passed to the LLM to generate an answer.
7. 📊 The response can be logged in Google Sheets.

## 🛠️ Tech Stack

- **Language:** Python
- **LLM Framework:** LangChain
- **LLM:** OpenAI-compatible API endpoints
- **Embeddings:** Hugging Face, Sentence Transformers
- **Vector Search:** FAISS, Chroma
- **PDF Processing:** PyMuPDF
- **Logging:** Google Sheets, gspread

## 📁 Project Structure

```text
Dialogix/
│
├── main.py
├── retriver.py
├── retriver2.py
├── prompt.py
├── embbeder.py
├── sheets_logger.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## 🚀 Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Dialogix.git
cd Dialogix
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
LLM_URL1=your_llm_endpoint
LLM_MODEL1=your_model_name
LLM_API_KEY1=your_api_key
```

### 5. Add HR policy PDFs

Place your PDF documents inside the `Policy_Documents_iAgami` folder.

### 6. Run the application

```bash
python main.py
```

## 💡 Example Questions

- 📅 What is the annual leave policy?
- 🏥 How many sick leaves are allowed?
- 📈 What is the employee appraisal process?
- 👥 What is the recruitment process?

## 🧩 Things I Learned

Building this project helped me understand that getting good answers from an LLM is not just about choosing a model. Document chunking, embedding quality, retrieval accuracy, and prompt design all make a difference.

One challenge I explored was improving retrieval when the chatbot selected irrelevant document chunks. I experimented with different retrieval approaches and reranking to improve the results.



<p align="center">
  💻 Built with Python, LangChain & RAG
  <br/>
  🚀 Exploring practical applications of Generative AI
</p>
