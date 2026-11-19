# ==========================================================================================================
# Imports
# ==========================================================================================================

import json
import os
import shutil

import numpy as np


from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer



# ==========================================================================================================
# Server declaration
# ==========================================================================================================

server = FastMCP(name="Multimodal-Retrieval_Server", instructions="Server for multimodal retrieval of restaurant and recipe information.")


# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def _unwrap(res: dict):  
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")


def embed_texts(texts, batch_size=64):
    # ---- Text embedding model (384-d) ----
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)


# ==========================================================================================================
# Tools
# ==========================================================================================================

@server.tool()
def retrieve_articles(chromadb_dir: str, query: str, k: int = 1, where: dict | None = None):
    """
    Function to retrieve similar articles from the article database based on a text query,
    with optional metadata filtering.
    Args:
        chromadb_dir: str
            The directory where the Chroma database is persisted.
        query: str
            The text query to search for.
        k: int
            The number of top results to return.
        where: dict | None
            Optional metadata filtering conditions. For example:
                where = {"cuisine": "Italian", "location": "New York"}
            If None, no filtering is applied.
    Returns:
        Four lists: ids, docs, metas, dists
            - ids: List of document IDs of the top-k results.
            - docs: List of document contents of the top-k results.
            - metas: List of metadata dictionaries of the top-k results.
            - dists: List of distances (similarity scores) of the top-k results.
    """

    chromadb = Chroma(
        collection_name="restaurant_articles",
        persist_directory=chromadb_dir,
    )

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _unwrap(res)


@server.tool()
def initialize_persistent_restaurant_database():

    # --------------------------------------- Loader ---------------------------------------

    with open("../Data/structured_restaurant_data.json", "r") as f:
        restaurants = json.load(f)

    print(f"Loaded restaurants: {len(restaurants)}")


    # --------------------------------------- Docs ---------------------------------------

    article_docs = []

    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue

        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style','')}\n"
            f"Location: {r.get('location','')}"
        )

        # GUARANTEED UNIQUE
        doc_id = f"rest_{i}"

        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "source": "restaurant",
                },
            )
        )

    print("article docs:", len(article_docs))

    
    # --------------------------------------- Embeddings ---------------------------------------

    embeddings = embed_texts([d.page_content for d in article_docs]).tolist()

    DB_DIR = str("chroma_multimodal")

    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)  # Reset vector DB (important for reruns)


    # --------------------------------------- Vector Store ---------------------------------------

    article_db = Chroma(
        collection_name="restaurant_articles",
        persist_directory=DB_DIR,
    )

    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=embeddings,
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )

    print("Article DB ready")



# ==============================================
# ENTRY POINT
# ==============================================

# Run the Server in STDIO mode
if __name__ == "__main__":
    server.run()