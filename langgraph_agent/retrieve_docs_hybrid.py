import numpy as np
import chromadb
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding 
from fastembed.rerank.cross_encoder import TextCrossEncoder


def initialize_models():
    "Initialize the three embedding models for Hybrid search"
    print("Initializing models...")
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print("Dense embedding model loaded (all-MiniLM-L6-v2)")
    bm25_model = SparseTextEmbedding("Qdrant/bm25")
    print("BM25 embedding model loaded (Qdrant/bm25)")
    late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    print("Late interaction embedding model loaded (colbert-ir/colbertv2.0)")

    return dense_model, bm25_model, late_interaction_model


# Function to embed documents
def embed_docs(documents):

    # Embedding Model
    """
    Embeds a list of documents into a ChromaDB vector store using dense embeddings.
    Note: ChromaDB uses dense embeddings natively. For hybrid search with BM25 and sparse vectors,
    consider preprocessing the documents with BM25 ranking before ChromaDB retrieval.

    Args:
        documents (list[str]): A list of documents to embed.

    Returns:
        chromadb.Collection: A ChromaDB collection with embedded documents.
    """

    # Initialize embedding model
    print("Initializing models...")
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print("Dense embedding model loaded (all-MiniLM-L6-v2)")
    
    # Embedding Documents with dense model
    dense_embeddings = list(dense_model.embed(doc for doc in documents))
    
    # ChromaDB Client
    client = chromadb.EphemeralClient()

    # Get or create a collection
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Upload data to ChromaDB
    collection.add(
        ids=[str(idx) for idx in range(len(documents))],
        documents=documents,
        embeddings=dense_embeddings,
        metadatas=[{"document": doc} for doc in documents]
    )

    print("Embedding done!")
    print(f'There are {collection.count()} documents in the collection\n')

    return collection


## Function to get documents from ChromaDB
def get_doc_answer(docs, query: str, k: int = 3) -> list[str]:
    """
    Retrieves k documents from ChromaDB based on query using dense embeddings.

    Args:
        docs (chromadb.Collection): The ChromaDB collection.
        query (str): The query to search for.
        k (int): The number of documents to retrieve. Defaults to 3.

    Returns:
        list[str]: A list of k document descriptions.
    """
    print("Retrieving nodes")

    # Embedding Model for query
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # Embed Query
    query_embedded = list(dense_model.query_embed(query))[0]

    # Retrieve data from ChromaDB
    collection = docs
    results = collection.query(
        query_embeddings=[query_embedded],
        n_results=k
    )

    return results



if __name__ == "__main__":
    
    # Sample Text
    text = [
        "In deep learning, the transformer is an artificial neural network architecture based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table.", 
        "At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished.",
        "Transformers have the advantage of having no recurrent units, therefore requiring less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM)."
        "Later variations have been widely adopted for training large language models (LLMs) on large (language) datasets.",
        "The modern version of the transformer was proposed in the 2017 paper 'Attention Is All You Need' by researchers at Google.",
        "The predecessors of transformers were developed as an improvement over previous architectures for machine translation, but have found many applications since.",
        "They are used in large-scale natural language processing, computer vision (vision transformers), reinforcement learning,[6][7] audio,[8] multimodal learning, robotics,[9] and even playing chess.[10] It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs)[11] and BERT[12] (bidirectional encoder representations from transformers).",
        "Transformers are the foundational neural network architecture enabling modern Large Language Models (LLMs) like ChatGPT, allowing them to process sequences of text efficiently using a self-attention mechanism to weigh word importance, leading to deep contextual understanding for tasks like text generation, translation, and summarization, essentially powering AI's ability to understand and create human-like language.",
        "LLMs use stacked transformer blocks (encoders/decoders) to predict the next word by understanding context from vast datasets, making them powerful tools for complex NLP applications.",
        "The transformer model is a type of neural network architecture that excels at processing sequential data, most prominently associated with large language models (LLMs).",
        "Transformer models have also achieved elite performance in other fields of artificial intelligence (AI), such as computer vision, speech recognition and time series forecasting."
    ]

    query= "nothing to see here"

    embedded_docs = embed_docs(documents=text)

    retrieved_docs = get_doc_answer(docs=embedded_docs, query=query, k=3)
    
    print('\n ---')
    print('Results:\n')
    print(retrieved_docs)