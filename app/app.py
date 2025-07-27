import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Loan Chatbot", layout="centered")

# Load and preprocess the data
@st.cache_resource
def load_data():
    df = pd.read_csv(r"C:\Users\kumar\OneDrive\Desktop\Celebal Technology\loan_chatbot_rag\data\cleaned_loan_dataset.csv")
    df['combined_text'] = df.fillna('').apply(lambda row: ' '.join(str(x) for x in row), axis=1)
    return df

df = load_data()

# TF-IDF Vectorizer
@st.cache_resource
def get_vectorizer_and_matrix():
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_vectorizer_and_matrix()

# Context Retriever
def retrieve_context(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-top_k:][::-1]
    return "\n".join(df['combined_text'].iloc[top_indices])

# Load Hugging Face model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Generate Answer
def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
# st.set_page_config(page_title="Loan Chatbot", layout="centered")
st.title(" Loan Approval Q&A Chatbot")
st.write("Ask a question based on the loan dataset:")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
        st.success("✅ Answer:")
        st.write(answer)
