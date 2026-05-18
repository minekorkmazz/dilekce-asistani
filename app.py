"""
Dilekçe RAG — Streamlit Arayüzü
=================================
Kurulum:
    pip install streamlit groq chromadb sentence-transformers python-dotenv

Çalıştır:
    streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# RUSTFS (S3) STORAGE
# ─────────────────────────────────────────

import boto3
from botocore.client import Config
from datetime import datetime

def get_rustfs_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("RUSTFS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("RUSTFS_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

def ensure_bucket(client, bucket: str):
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)

def save_to_rustfs(dilekce: str, user_problem: str) -> str:
    """Dilekçeyi rustfs'e kaydeder ve indirme url'i döner."""
    try:
        client = get_rustfs_client()
        bucket = os.getenv("RUSTFS_BUCKET", "dilekce-ciktilari")
        ensure_bucket(client, bucket)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"dilekce_{timestamp}.txt"

        content_with_meta = "Kullanıcı Sorusu: " + user_problem + "\n" + "="*60 + "\n" + dilekce

        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content_with_meta.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        # Presigned URL (1 saat geçerli)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception as e:
        return f"HATA: {e}"



# ─────────────────────────────────────────
# MEVZUAT MCP
# ─────────────────────────────────────────

import re
import asyncio

MEVZUAT_MCP_URL = "https://mevzuat.surucu.dev/mcp"

def kanun_no_cikar(metin: str) -> list:
    """Dilekçe metninden kanun numaralarını çıkar."""
    return list(set(re.findall(r'\b(\d{4})\s*sayılı', metin, re.IGNORECASE)))

async def _kanun_sorgula_async(kanun_nolari: list) -> list:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    sonuclar = []
    async with streamablehttp_client(MEVZUAT_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for no in kanun_nolari:
                try:
                    result = await session.call_tool(
                        "search_mevzuat",
                        {"mevzuat_no": no, "mevzuat_tur": "KANUN", "page_size": 5}
                    )
                    text = result.content[0].text
                    match = re.search(rf"\[{no}\]\s+(.+?)\s+\(", text)
                    ad = match.group(1) if match else "—"
                    bulundu = no in text and "Error" not in text
                    sonuclar.append({"no": no, "ad": ad, "guncel": bulundu})
                except Exception as e:
                    sonuclar.append({"no": no, "ad": "Sorgu hatası", "guncel": False})
    return sonuclar


YARGI_MCP_URL = "https://yargimcp.surucu.dev/mcp"

# Hukuk dalı → Yargıtay dairesi eşleştirmesi
HUKUK_DALI_DAIRE = {
    "is_hukuku":      "H9",
    "borclar_hukuku": "H11",
    "ticaret_hukuku": "H11",
    "icra_hukuku":    "H12",
    "aile_hukuku":    "H2",
    "medeni_hukuk":   "H1",
    "trafik_hukuku":  "H17",
    "tuketici_hukuku":"H3",
    "ceza_hukuku":    "ALL",
    "idare_hukuku":   "ALL",
}

async def _yargi_ara_async(sorgu: str, hukuk_dali: str) -> list:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession
    import json

    daire = HUKUK_DALI_DAIRE.get(hukuk_dali, "ALL")
    court_types = ["DANISTAYKARAR"] if hukuk_dali == "idare_hukuku" else ["YARGITAYKARARI"]

    async with streamablehttp_client(YARGI_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for attempt in range(3):
                result = await session.call_tool(
                    "search_bedesten_unified",
                    {
                        "phrase": sorgu,
                        "court_types": court_types,
                        "birimAdi": daire,
                    }
                )
                text = result.content[0].text
                data = json.loads(text)

                if data.get("error") == "rate_limit_exceeded":
                    retry_after = float(data.get("retry_after", 4))
                    await asyncio.sleep(retry_after + 1)
                    continue

                kararlar = []
                for d in data.get("decisions", [])[:3]:
                    kararlar.append({
                        "mahkeme":   d.get("birimAdi", ""),
                        "tarih":     d.get("kararTarihiStr", d.get("kararTarihi", "")[:10]),
                        "karar_no":  d.get("kararNo", ""),
                        "esas_no":   d.get("esasNo", ""),
                        "doc_id":    d.get("documentId", ""),
                    })
                return kararlar

    return []

def yargi_ara(sorgu: str, hukuk_dali: str) -> list:
    """Emsal karar ara."""
    try:
        return asyncio.run(_yargi_ara_async(sorgu, hukuk_dali))
    except Exception:
        return []

def yargi_sorgu_ozetle(dilekce: str, api_key: str) -> str:
    """Dilekçeden Yargıtay araması için kısa anahtar kelime özeti üret."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Verilen dilekçeden Yargıtay kararı aramak için en uygun 3-5 kelimelik Türkçe arama sorgusu üret. Sadece anahtar kelimeleri yaz, başka hiçbir şey yazma."},
            {"role": "user",   "content": dilekce[:1000]},
        ],
        temperature=0,
        max_tokens=30,
    )
    return response.choices[0].message.content.strip()

def mevzuat_kontrol(dilekce: str) -> list:
    """Dilekçedeki kanunları Mevzuat MCP ile kontrol et."""
    kanun_nolari = kanun_no_cikar(dilekce)
    if not kanun_nolari:
        return []
    return asyncio.run(_kanun_sorgula_async(kanun_nolari))

# ─────────────────────────────────────────
# SAYFA
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Dilekçe Asistanı",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Dilekçe Asistanı")
st.caption("Probleminizi yazın, size uygun dilekçeyi hazırlayalım.")

# ─────────────────────────────────────────
# sidebar
# ─────────────────────────────────────────

groq_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ Ayarlar")

    top_k = st.slider("Kaç şablon kullanılsın?", 1, 5, 3)

    st.divider()
    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("**Vektör DB:** Chroma HTTP")
    st.markdown("**Storage:** RustFS")
    st.markdown("**Embedding:** trmteb/turkish-embedding-model")
    st.divider()
    if groq_key:
        st.success("API bağlantısı aktif")
    else:
        st.error("GROQ_API_KEY eksik")


# ─────────────────────────────────────────
# MODEL YÜKLEMESİ (cache)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("trmteb/turkish-embedding-model")

@st.cache_resource(show_spinner="Veritabanı bağlanıyor...")
def load_chroma():
    import chromadb

    host = os.getenv("CHROMA_HOST")
    port = int(os.getenv("CHROMA_PORT", 8000))

    if host:
        client = chromadb.HttpClient(host=host, port=port)
    else:
        client = chromadb.PersistentClient(path="./chroma_db")

    return client.get_collection("dilekce_turkembed")

@st.cache_resource(show_spinner="BM25 index yükleniyor...")
def load_bm25():
    """chunks.json'dan BM25 index oluştur — bellekte tutar, Chroma'ya dokunmaz."""
    from rank_bm25 import BM25Okapi
    import json
    from pathlib import Path

    data   = json.loads(Path("/app/chunks.json").read_text(encoding="utf-8"))
    chunks = data["chunks"]

    # Türkçe tokenize: küçük harf + split
    corpus = [c["text"].lower().split() for c in chunks]
    bm25   = BM25Okapi(corpus)
    return bm25, chunks

def rrf(semantic_hits: list, bm25_hits: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion — iki listeyi birleştir."""
    scores = {}
    for rank, hit in enumerate(semantic_hits):
        key = hit["dosya_adi"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    for rank, hit in enumerate(bm25_hits):
        key = hit["dosya_adi"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Sırala ve semantic hit bilgilerini koru
    hit_map = {h["dosya_adi"]: h for h in semantic_hits + bm25_hits}
    sorted_keys = sorted(scores, key=lambda x: -scores[x])
    return [hit_map[k] for k in sorted_keys if k in hit_map]


# ─────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────

def retrieve(query: str, top_k: int) -> list[dict]:
    model      = load_embed_model()
    collection = load_chroma()
    bm25, chunks = load_bm25()

    # 1. Semantic search (Chroma)
    q_vec   = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k * 3,   # RRF için daha fazla aday al
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = []
    for i in range(len(results["ids"][0])):
        semantic_hits.append({
            "score":      round(1 - results["distances"][0][i], 4),
            "dosya_adi":  results["metadatas"][0][i]["dosya_adi"],
            "hukuk_dali": results["metadatas"][0][i]["hukuk_dallari"],
            "belge_turu": results["metadatas"][0][i]["belge_turu"],
            "konu":       results["metadatas"][0][i].get("konu", ""),
            "text":       results["documents"][0][i],
        })

    # 2. BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    top_bm25_idx    = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k * 3]
    bm25_hits = []
    for idx in top_bm25_idx:
        c = chunks[idx]
        bm25_hits.append({
            "score":      round(float(bm25_scores[idx]), 4),
            "dosya_adi":  c["metadata"]["dosya_adi"],
            "hukuk_dali": c["metadata"]["hukuk_dallari"],
            "belge_turu": c["metadata"]["belge_turu"],
            "konu":       c["metadata"].get("konu", ""),
            "text":       c["text"],
        })

    # 3. RRF ile birleştir
    merged = rrf(semantic_hits, bm25_hits)
    return merged[:top_k]


# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────

SYSTEM_PROMPT = """Sen deneyimli bir Türk hukuk asistanısın.
Kullanıcının anlattığı problemi dinleyip, verilen dilekçe şablonunu
baz alarak doldurulmuş, hazır bir dilekçe yazıyorsun.

Kurallar:
- Dilekçeyi Türk hukuku standartlarında, resmi dilde yaz
- Şablondaki yapıyı koru: DAVACI, KONU, AÇIKLAMALAR, HUKUKİ SEBEPLER, SONUÇ VE İSTEM
- Kullanıcının vermediği bilgileri [AD SOYAD], [TARİH] gibi köşeli parantez ile bırak
- Şablonda geçen kanun maddelerini ve hukuki dayanakları koru
- Uydurma bilgi ekleme
"""

def generate(user_problem: str, hits: list[dict], api_key: str) -> str:
    from groq import Groq

    best   = hits[0]
    prompt = f"""Kullanıcının problemi:
{user_problem}

---
Aşağıdaki dilekçe şablonunu baz alarak doldurulmuş bir dilekçe yaz:

ŞABLON ({best['dosya_adi']} | {best['hukuk_dali']}):
{best['text']}
"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# ANA ARAYÜZ
# ─────────────────────────────────────────

user_problem = st.text_area(
    "Probleminizi anlatın",
    placeholder=(
        "Örnek: 5 yıl önce aldığım ceza infaz edildi, "
        "yasal süresi de doldu ama adli sicil kaydımda hâlâ görünüyor. "
        "İş başvurularımda sorun çıkıyor, silinmesini istiyorum."
    ),
    height=140,
)

col1, col2 = st.columns([1, 1])
with col1:
    gonder = st.button("📝 Dilekçe Oluştur", type="primary", use_container_width=True)
with col2:
    temizle = st.button("🗑️ Temizle", use_container_width=True)

if temizle:
    st.rerun()

MIN_CHARS = 20
MAX_CHARS = 2000

if gonder:
    problem = user_problem.strip()

    if not problem:
        st.warning("Lütfen probleminizi yazın.")
        st.stop()

    if len(problem) < MIN_CHARS:
        st.warning(f"Probleminizi daha ayrıntılı anlatın. (En az {MIN_CHARS} karakter)")
        st.stop()

    if len(problem) > MAX_CHARS:
        st.error(f"Metin çok uzun. Lütfen {MAX_CHARS} karakteri aşmayacak şekilde özetleyin. (Şu an: {len(problem)} karakter)")
        st.stop()

    if not groq_key:
        st.error("GROQ_API_KEY tanımlı değil. .env dosyasını kontrol edin.")
        st.stop()

    user_problem = problem

    # Retrieval
    with st.spinner("🔍 Şablonlar aranıyor..."):
        hits = retrieve(user_problem, top_k)

    # Bulunan şablonlar
    with st.expander(f"📂 {len(hits)} şablon bulundu", expanded=False):
        for h in hits:
            st.markdown(
                f"**{h['dosya_adi']}**  \n"
                f"Skor: `{h['score']}`  |  Dal: `{h['hukuk_dali']}`  |  "
                f"Tür: `{h['belge_turu']}`"
            )
            if h["konu"]:
                st.caption(f"Konu: {h['konu']}")
            st.divider()

    # LLM üretim
    with st.spinner("✍️ Dilekçe hazırlanıyor..."):
        try:
            dilekce = generate(user_problem, hits, groq_key)
        except Exception as e:
            st.error(f"Hata: {e}")
            st.stop()

    # Sonuç
    st.success("Dilekçeniz hazır!")
    st.text_area("📄 Üretilen Dilekçe", value=dilekce, height=500)

    # İndir
    st.download_button(
        label="⬇️ TXT olarak indir",
        data=dilekce.encode("utf-8"),
        file_name="dilekce.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Rustfs'e kaydet
    with st.spinner("☁️  Dilekçe kaydediliyor..."):
        rustfs_url = save_to_rustfs(dilekce, user_problem)

    if rustfs_url.startswith("HATA"):
        st.warning(f"Bulut kaydı başarısız: {rustfs_url}")
    else:
        st.success("☁️  Dilekçe buluta kaydedildi!")
        st.markdown(f"[📥 Buluttan indir (1 saat geçerli)]({rustfs_url})")

    # Mevzuat kontrolü
    with st.spinner("Kanunlar doğrulanıyor..."):
        mevzuat_sonuclari = mevzuat_kontrol(dilekce)

    with st.expander(" Mevzuat Doğrulama", expanded=True):
        if mevzuat_sonuclari:
            for m in mevzuat_sonuclari:
                if m["guncel"]:
                    st.success(f"{m['no']} sayılı {m['ad']} — Güncel")
                else:
                    st.warning(f" {m['no']} sayılı kanun doğrulanamadı")
        else:
            st.info("Bu dilekçede kanun referansı tespit edilmedi.")

    # Yargı emsal karar
    hukuk_dali = hits[0]["hukuk_dali"].split(", ")[0] if hits else "diger"
    with st.spinner(" Emsal kararlar aranıyor..."):
        yargi_sorgu = yargi_sorgu_ozetle(dilekce, groq_key)
        emsal_kararlar = yargi_ara(yargi_sorgu, hukuk_dali)

    with st.expander("⚖️ Emsal Kararlar", expanded=True):
        if emsal_kararlar:
            for k in emsal_kararlar:
                st.markdown(f"**{k['mahkeme']}** — {k['tarih']}")
                st.markdown(f"Esas No: `{k['esas_no']}` | Karar No: `{k['karar_no']}`")
                st.divider()
        else:
            st.info("Emsal karar bulunamadı.")

    st.info(
        "💡 Köşeli parantez içindeki alanları `[AD SOYAD]`, `[TARİH]` "
        "kendi bilgilerinizle doldurun. Bu dilekçe bilgi amaçlıdır, "
        "avukatınıza danışmanız önerilir.",
        icon="⚠️",
    )