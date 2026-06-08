import os
import numpy as np
import json
import streamlit as st
import google.generativeai as genai
import chromadb
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langgraph_agent.document_loader import load_document
from dotenv import load_dotenv
load_dotenv()


# Function to embed documents
def embed_docs(text):

    # Embedding Model
    """
    Embeds a list of documents into a ChromaDB vector store.

    Args:
        text (list[str]): A list of documents to embed.

    Returns:
        chromadb.Collection: A ChromaDB collection with embedded documents.
    """
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Creating vector store embeddings
    vector_store = list(
        embedding_model.embed(text) )

    # ChromaDB Client
    client = chromadb.EphemeralClient()

    # Get or create a collection
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Upload data to ChromaDB
    collection.add(
        ids=[str(idx) for idx in range(len(text))],
        documents=text,
        embeddings=vector_store,
        metadatas=[{"description": doc} for doc in text]
    )

    st.caption("🔢 | Embedding done!")
    st.caption(f'➡️ | There are {collection.count()} documents in the collection')

    return collection


## Function to get documents from ChromaDB
def get_doc_answer(docs, query: str, k: int = 2) -> list[str]:
    """
    Retrieves k documents from ChromaDB based on query.

    Args:
    docs (chromadb.Collection): ChromaDB collection to search in.
    query (str): The query to search for.
    k (int): The number of documents to retrieve. Defaults to 2.

    Returns:
    list[str]: A list of k document descriptions.
    """
    st.caption(":black_circle: | Retrieving nodes")

    # Embedding Model
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Embed Query
    query_embedded = list(embedding_model.query_embed(query))[0]

    collection = docs
    # Retrieve data from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedded],
        n_results=k
    )

    description_hits = []
    if results and results['documents']:
        for doc_list in results['documents']:
            description_hits.extend(doc_list)

    return description_hits


def rerank(query, retrieved_docs):

    # Create Reranker
    reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')
    
    # Return scores between query and each document
    new_scores = list(
    reranker.rerank(query, retrieved_docs)
    )  
    
    # Sort them in order of relevance defined by reranker
    ranking = [ (i, score) for i, score in enumerate(new_scores) ]
    ranking.sort(
        key=lambda x: x[1], reverse=True
    )  

    # Print reranked results
    description_hits = []
    for i, rank in enumerate(ranking):
        # print(f'''Reranked result number {i+1} is \"{retrieved_docs[rank[0]]}\"''')
        description_hits.append(retrieved_docs[rank[0]])

    return description_hits


# LLM-as-a-Judge Prompt
llm_judge_prompt = """
You are an expert evaluator of Retrieval-Augmented Generation systems.

User question:
{query}

Retrieved documents:
{retrieved_docs}

Generated answer:
{answer}

Evaluate the answer using the retrieved documents.

Answer the following in JSON:
{{
  "relevant_docs": true | false,
  "sufficient_context": true | false,
  "score": number between 0 and 1
}}

Guidelines:
- relevant_docs = false if documents do not address the user question
- sufficient_context = false if documents are related but incomplete
- score should reflect overall answer quality and faithfulness
"""

# Function LLM-as-a-Judge
def llm_judge(query, retrieved_docs, answer):
    """
    Evaluate the answer using the retrieved documents.

    Args:
        query (str): The user query.
        retrieved_docs (list[str]): The retrieved documents.
        answer (str): The generated answer.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    prompt = llm_judge_prompt.format(query=query, retrieved_docs=retrieved_docs, answer=answer)
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return json.loads(response.text)


if __name__ == "__main__":
    
    query= "What is A/B testing?"

    embedded_docs = embed_docs()

    retrieved_docs = get_doc_answer(docs=embedded_docs, query=query, k=5)
    
    print('\n ---')
    print('Reranking results...\n')

    final_docs = rerank(query=query, retrieved_docs=retrieved_docs)
    print('\n Final Docs---')
    print(final_docs)

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt_text = "Use the following documents to answer the user question: " + str(final_docs) + "If the answer cannot be found in the documents, respond with 'I didn't find any relevant documents.' User question: " + query
    ai_answer = model.generate_content(prompt_text)

    print("LLM Answer:")
    print(ai_answer.text, "\n")

    print("LLM Judge:")
    print(llm_judge(query=query, 
              retrieved_docs=final_docs, 
              answer=ai_answer.text))