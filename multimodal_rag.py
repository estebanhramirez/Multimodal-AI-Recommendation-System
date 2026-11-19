# ==========================================================================================================
# Imports
# ==========================================================================================================

# Standard library
import glob
import json
import os
import shutil
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

# Chroma returns lists-of-lists; unwrap the first query.
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



# ==========================================================================================================
# Embedding Models
# ==========================================================================================================

def embed_texts(texts, batch_size=64):
    # ---- Text embedding model (384-d) ----
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)


@torch.no_grad()
def embed_images(paths, batch_size=16):
    # ---- CLIP embedding model (512-d) for image + query text ----
    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()

    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))

    return (np.vstack(vecs))



# ==========================================================================================================
# Retrieval functions
# ==========================================================================================================

def retrieve_articles(chromadb, query: str, k: int = 5, where: dict | None = None):
    """
    Function to retrieve similar articles from the article database based on a text query,
    with optional metadata filtering.
    Args:
        query: str
            The text query to search for.
        k: int
            The number of top results to return.
        where: dict | None
            Optional metadata filtering conditions. For example:
                where = {"cuisine": "Italian", "location": "New York"}
            If None, no filtering is applied.
    Returns:

    """

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _unwrap(res)


def retrieve_images_by_image(chromadb, query_image_path: str, k: int = 5, where: dict | None = None):
    """
    Function to retrieve similar images from the image database based on a query image,
    with optional metadata filtering.

    Args:
        query_image_path: str
            The file path to the query image.
        k: int
            The number of top results to return.
        where: dict | None
            Optional metadata filtering conditions. For example:
                where = {"cuisine": "Italian"}
            If None, no filtering is applied.
    Returns:
    """

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _unwrap(res)



# ==========================================================================================================
# Beginning of executions
# ==========================================================================================================

def main():

    with open("../Data/structured_restaurant_data.json", "r") as f:
        restaurants = json.load(f)

    with open("../Data/augmented_food_recipe.json", "r") as f:
        recipes = json.load(f)

    print(f"✅ Loaded restaurants: {len(restaurants)}")
    print(f"✅ Loaded recipes:     {len(recipes)}")


    IMG_DIR  = "../Data/recipe_images"
    image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
    print(f"✅ Images found: {len(image_paths)}")

    # -------- articles --------
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

    print("✅ article docs:", len(article_docs))


    # -------- images --------
    image_docs = []

    for i, (p, rec) in enumerate(zip(image_paths, recipes)):
        doc_id = f"img_{i}"

        image_docs.append(
            Document(
                # keeps retrieval results readable
                page_content=rec.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": doc_id,
                    "image_path": p,
                    "source": "recipe_image",
                    "recipe_id": rec.get("id"),
                    "cuisine": rec.get("cuisine"),
                },
            )
        )

    print("✅ image docs:", len(image_docs))



    DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)  # Reset vector DB (important for reruns)

    # ----- article DB -----

    article_db = Chroma(
        collection_name="restaurant_articles",
        persist_directory=DB_DIR,
    )

    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=embed_texts([d.page_content for d in article_docs]).tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )

    print("✅ Article DB ready")


    # ----- image DB -----

    image_db = Chroma(
        collection_name="food_images",
        persist_directory=DB_DIR,
    )

    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=embed_images([d.metadata["image_path"] for d in image_docs]).tolist(),
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )

    print("✅ Image DB ready")
    print("🎉 Multimodal Vector Index Construction COMPLETE")


    q = "cozy restaurant with noodles and warm atmosphere"

    ids, docs, metas, dists = retrieve_articles(article_db, q, k=5, where=None)
    print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")

    print("✅ Demo 1 complete")


    ids, docs, metas, dists = retrieve_images_by_image(image_db, query_image_path="query_image.jpeg", k=5, where=None)

    if len(ids) == 0:
        print("⚠️ No results found with current filter.")
    else:
        print_hits(ids, docs, metas, dists, title="Demo 3 — Image similarity search (image→image)")

    print("✅ Demo 3 complete")



# ==========================================================================================================
# Entry point
# ==========================================================================================================

if __name__ == "__main__":
    main()