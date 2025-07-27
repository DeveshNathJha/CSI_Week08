# CSI_Week08

# Loan Chatbot using RAG (Retrieval-Augmented Generation)

This project is an intelligent **Q&A chatbot** for loan-related queries using **Retrieval-Augmented Generation (RAG)**. It retrieves relevant context from the training dataset and generates accurate answers using a lightweight Hugging Face model (`google/flan-t5-base`).

---

## Features

- Contextual answers from real loan data
- Lightweight Hugging Face model for efficient inference
- Built using:
  - `pandas`, `scikit-learn` for preprocessing and similarity matching
  - `transformers`, `torch` for language modeling
  - `Streamlit` for interactive chatbot interface

---

## Project Structure
<pre lang="markdown">
'''

loan_chatbot_rag/
├── app/
│ └── app.py # Streamlit app
├── data/
│ └── Training Dataset Cleaned.csv # Preprocessed dataset
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Data cleaning
│ └── 02_rag_pipeline_huggingface.ipynb # RAG pipeline
├── requirements.txt
├── README.md

'''
</pre>
---

##  Setup Instructions

### 1. Clone the Repository
<pre>

git clone https://github.com/kumarHrishav/loan_chatbot_rag.git
cd loan_chatbot_rag

Install Dependencies
Use pip:
pip install -r requirements.txt

Or using Anaconda (optional):

conda create -n rag_chatbot python=3.10
conda activate rag_chatbot
pip install -r requirements.txt

Run Locally

streamlit run app/app.py
</pre>
