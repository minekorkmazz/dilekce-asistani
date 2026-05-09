"""
Chroma http server'a indexleme yapar.
Sadece Docker entrypoint tarafından çağrılır.
"""
import os, json, chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "/app/chunks.json")
EMBED_MODEL = "trmteb/turkish-embedding-model"
COLLECTION_NAME = "dilekce_turkembed"
BATCH_SIZE = 50


def main():
    # Chunks yükle
    data = json.loads(Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    chunks = data["chunks"]
    texts = [c["text"] for c in chunks]
    print(f"📦 {len(chunks)} chunk yüklenecek")

    # Embedding
    print(f"⬇️  Model yükleniyor: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("✍️  Embedding üretiliyor...")
    all_vectors = []
    for i in tqdm(range(0, len(texts), 32), desc="Embedding"):
        batch = texts[i:i + 32]
        vecs = model.encode(batch, normalize_embeddings=True)
        all_vectors.extend([[float(x) for x in v] for v in vecs])

    # Chroma'ya yaz
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("💾 Chroma'ya yazılıyor...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Chroma"):
        bc = chunks[i:i + BATCH_SIZE]
        bv = all_vectors[i:i + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c in bc],
            embeddings=bv,
            documents=[c["text"] for c in bc],
            metadatas=[{
                "dosya_adi": c["metadata"]["dosya_adi"],
                "hukuk_dallari": c["metadata"]["hukuk_dallari"],
                "belge_turu": c["metadata"]["belge_turu"],
                "konu": c["metadata"].get("konu", ""),
                "doc_id": c["metadata"]["doc_id"],
            } for c in bc],
        )

    print(f"✅ Indexleme tamamlandı: {collection.count()} chunk")


if __name__ == "__main__":
    main()
