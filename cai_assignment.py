import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import streamlit as st
from nltk.tokenize import word_tokenize
import re

# ==========================
# STEP 1: DATA COLLECTION & PREPROCESSING
# ==========================

def load_and_preprocess_data(csv_file):
    """
    Load financial statements CSV file, clean and structure data.
    """
    df = pd.read_csv(csv_file)
    df = df.astype(str)  # Convert all columns to string type
    df.fillna("N/A", inplace=True)
    return df

# Load financial data & pre-process
csv_file = "Financial_Statements.csv"  # Change this path if needed
data = load_and_preprocess_data(csv_file)

# Convert each row into a textual representation
text_chunks = data.apply(lambda row: " | ".join(row.values), axis=1).tolist()

# ==========================
# STEP 2: BASIC RAG IMPLEMENTATION
# ==========================

# Load a pre-trained sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Open-source, lightweight

# Convert text chunks into embeddings
embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)

# Store embeddings in FAISS vector database to index the data
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)  # L2 similarity search
index.add(embeddings)  # Add vectors to FAISS index

# ==========================
# STEP 3: ADVANCED RAG IMPLEMENTATION
# ==========================

# Tokenize text chunks for BM25 keyword-based search
tokenized_corpus = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Load a cross-encoder for re-ranking retrieved documents
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_documents(query, top_k=5):
    """
    Retrieve relevant documents using FAISS (vector search) and BM25 (keyword search).
    """
    # Compute embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Retrieve top-k results from FAISS (vector search)
    _, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [text_chunks[i] for i in faiss_indices[0]]
    
    # Retrieve top-k results from BM25 (keyword search)
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [text_chunks[i] for i in bm25_indices]
    
    # Combine results and remove duplicates
    combined_results = list(set(faiss_results + bm25_results))
    return combined_results

def rerank_documents(query, documents):
    """
    Re-rank retrieved documents using a cross-encoder for improved relevance.
    """
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    
    # Normalize scores to a 0–100 range
    min_score, max_score = min(scores), max(scores)
    st.write(f"\n**min_score :** {min_score}  \n**max_score :** {max_score} \n**scores: **{scores}% scores")

    # confidence_scores = [(s - min_score) / (max_score - min_score) * 100 if max_score > min_score else 50 for s in scores]
    if abs(max_score - min_score) < 2:
        confidence_scores = [50] * len(scores)
    else:
        confidence_scores = [(s - min_score) / (max_score - min_score) * 100 for s in scores]
    
    ranked_results = sorted(zip(documents, confidence_scores), key=lambda x: x[1], reverse=True)
    return ranked_results


def extract_revenue_from_text(text, year):
    """
    Extract revenue value from structured financial statement text.
    """
    columns = text.split('|')
    headers = data.columns.tolist()  # Extract column headers

    if str(year) in columns:
        year_index = columns.index(str(year))
        try:
            revenue_index = headers.index("Revenue")  # Find the "Revenue" column
            revenue = columns[revenue_index].strip()
            if revenue.replace('.', '', 1).isdigit():
                return f"Revenue in {year}: **${revenue}**"
        except ValueError:
            return "Revenue data not found."
    
    return "Revenue data not found."


# ==========================
# STEP 4: UI DEVELOPMENT (STREAMLIT)
# ==========================

def main():
    st.title("Financial RAG Chatbot")
    st.write("Ask financial questions based on company financial data.")

    query = st.text_input("Enter your financial question:")
    if query:
        documents = retrieve_documents(query)
        ranked_docs = rerank_documents(query, documents)

        if ranked_docs:
            # Check for revenue extraction
            if "revenue" in query.lower():
                year_match = re.search(r"\d{4}", query)
                if year_match:
                    extracted_revenue = extract_revenue_from_text(ranked_docs[0][0], year_match.group(0))
                    confidence = ranked_docs[0][1]
                    st.subheader("Top Answer:")
                    st.write(f"{extracted_revenue}  \n**Confidence Score:** {confidence:.2f}%")
                else:
                    st.subheader("Top Answer:")
                    st.write("Revenue data not found.")
            else:
                top_doc, top_confidence = ranked_docs[0]
                st.subheader("Top Answer:")
                st.write(f"{top_doc}  \n**Confidence Score:** {top_confidence:.2f}%")

            st.subheader("Other Relevant Results:")
            for doc, conf in ranked_docs[1:]:
                st.write(f"- {doc}  \n**Confidence Score:** {conf:.2f}%")
        else:
            st.write("No relevant results found.")

# ==========================
# STEP 5: GUARDRAIL IMPLEMENTATION
# ==========================

def validate_query(query):
    """
    Validate financial relevance of a query.
    """
    financial_terms = ["revenue", "profit", "earnings", "market cap", "share price"]
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = [np.dot(query_embedding, embedding_model.encode([term], convert_to_numpy=True).T) for term in financial_terms]
    return any(score > 0.5 for score in similarities)

def filter_response(response):
    """
    Prevent misleading responses.
    """
    if "N/A" in response or len(response) < 10:
        return "No reliable answer found. Please refine your question."
    return response

if __name__ == "__main__":
    st.title("Financial Chatbot")
    user_query = st.text_input("Enter a financial question:")
    
    if user_query:
        if validate_query(user_query):
            retrieved_docs = retrieve_documents(user_query)
            ranked_docs = rerank_documents(user_query, retrieved_docs)
            if ranked_docs:
                response, confidence = ranked_docs[0]
                response = filter_response(response)
                st.write(f"{response}  \n**Confidence Score:** {confidence:.2f}%")
            else:
                st.write("No relevant results found.")
        else:
            st.write("Invalid query. Please ask a financial-related question. \n**Confidence Score:** 0%")
