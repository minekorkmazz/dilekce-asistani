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


# ─────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────

def retrieve(query: str, top_k: int) -> list[dict]:
    model = load_embed_model()
    collection = load_chroma()

    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "score":      round(1 - results["distances"][0][i], 4),
            "dosya_adi":  results["metadatas"][0][i]["dosya_adi"],
            "hukuk_dali": results["metadatas"][0][i]["hukuk_dallari"],
            "belge_turu": results["metadatas"][0][i]["belge_turu"],
            "konu":       results["metadatas"][0][i].get("konu", ""),
            "text":       results["documents"][0][i],
        })
    return hits


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

if gonder:
    if not user_problem.strip():
        st.warning("Lütfen probleminizi yazın.")
        st.stop()

    if not groq_key:
        st.error("GROQ_API_KEY tanımlı değil. .env dosyasını kontrol edin.")
        st.stop()

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

    st.info(
        "💡 Köşeli parantez içindeki alanları `[AD SOYAD]`, `[TARİH]` "
        "kendi bilgilerinizle doldurun. Bu dilekçe bilgi amaçlıdır, "
        "avukatınıza danışmanız önerilir.",
        icon="⚠️",
    )