<div align="center">

```
⚖️ DİLEKÇE ASİSTANI
```

**Yapay zeka destekli Türkçe dilekçe üretim sistemi**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-orange?style=flat-square)](https://trychroma.com)
[![RustFS](https://img.shields.io/badge/RustFS-S3%20Storage-red?style=flat-square)](https://rustfs.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🎯 Ne Yapar?

Kullanıcı yaşadığı hukuki sorunu sade bir dille yazar. Sistem:

1. **300 dilekçe şablonu** arasından en alakalı 3'ünü bulur
2. **LLaMA 3.3 70B** ile şablonu doldurulmuş, hazır dilekçeye dönüştürür
3. Üretilen dilekçeyi **RustFS**'e kaydeder, indirilebilir link üretir

```
"Adli sicil kaydımın silinmesini istiyorum"
        ↓
⚖️ Hazır, imzaya hazır dilekçe
```

---

## 🏗️ Mimari

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   ChromaDB       │     │    RustFS       │
│   :8501         │     │   :8000          │     │    :9000/9001   │
│                 │     │  300 vektör      │     │  Dilekçe arşivi │
└────────┬────────┘     └──────────────────┘     └────────▲────────┘
         │                                                  │
         │         ┌──────────────────┐                    │
         └────────▶│   Groq API       │────────────────────┘
                   │  LLaMA 3.3 70B   │
                   └──────────────────┘
```

### Servisler

| Servis | Teknoloji | Port | Görev |
|--------|-----------|------|-------|
| `streamlit` | Python + Streamlit | 8501 | Web arayüzü |
| `chroma` | ChromaDB | 8000 | Vektör veritabanı |
| `rustfs` | RustFS | 9000/9001 | Object storage |
| `indexer` | Python | — | İlk açılışta indexleme |

---

## ⚙️ RAG Pipeline

```
TXT Şablonları (300 adet)
        ↓
dilekce_chunker.py → chunks.json
        ↓
trmteb/turkish-embedding-model
        ↓
ChromaDB (cosine similarity)
        ↓
Top-3 şablon + kullanıcı sorunu
        ↓
LLaMA 3.3 70B (Groq API)
        ↓
Doldurulmuş dilekçe → RustFS
```

### Embedding Modeli

`trmteb/turkish-embedding-model` — TR-MTEB benchmark'ının resmi Türkçe embedding modeli. Turkish-colBERT'e kıyasla retrieval metriklerinde %19-26 üstünlük sağlıyor.

---

## 🚀 Kurulum

### Gereksinimler
- Docker Desktop
- Groq API key (ücretsiz → [console.groq.com](https://console.groq.com))

### 1. Repo'yu klonla

```bash
git clone https://github.com/kullanici/dilekce-asistani.git
cd dilekce-asistani
```

### 2. `.env` dosyası oluştur

```env
GROQ_API_KEY=gsk_...
RUSTFS_ROOT_USER=admin
RUSTFS_ROOT_PASSWORD=guclu_sifre_yaz
```

### 3. Başlat

```bash
docker compose up --build
```

İlk açılışta `indexer` servisi 300 şablonu otomatik vektörleştirir (~3-4 dk).  
Sonraki açılışlarda bu adım atlanır, direkt başlar.

### 4. Aç

| Servis | URL |
|--------|-----|
| Dilekçe Asistanı | http://localhost:8501 |
| RustFS Console | http://localhost:9001 |

---

## 📁 Proje Yapısı

```
dilekce-asistani/
├── app.py                  # Streamlit arayüzü + RustFS entegrasyonu
├── indexer.py              # Chroma indexleme (Docker'da otomatik çalışır)
├── dilekce_chunker.py      # TXT → chunks.json pipeline
├── chunks.json             # 300 dilekçe şablonu (metadata ile)
├── Dockerfile              # Streamlit + indexer image
├── docker-compose.yml      # 4 servis orkestrasyonu
├── requirements.txt
└── .env                    # API key'ler 
```

---

## 🤖 Kullanılan Teknolojiler

| Katman | Teknoloji | Neden? |
|--------|-----------|--------|
| Embedding | `trmteb/turkish-embedding-model` | Türkçe'ye özel, ücretsiz, lokal |
| Vektör DB | ChromaDB | Sıfır config, cosine similarity |
| LLM | LLaMA 3.3 70B (Groq) | Açık kaynak, veri gizliliği |
| Storage | RustFS | MinIO fork, Rust ile yazılmış, S3-uyumlu |
| UI | Streamlit | Hızlı prototip, Python-native |
| Orkestrasyon | Docker Compose | 4 servis, tek komut |

---

## 📊 Desteklenen Hukuk Dalları

`idare_hukuku` · `borclar_hukuku` · `ceza_hukuku` · `is_hukuku` · `icra_hukuku` · `imar_hukuku` · `tuketici_hukuku` · `medeni_hukuk` · `vergi_hukuku` · `emeklilik` · `fikri_mulkiyet` · `trafik_hukuku` · `aile_hukuku` · `saglik_hukuku` · `sosyal_hizmetler`

---

## ⚠️ Yasal Uyarı

Bu sistem bilgi amaçlıdır. Üretilen dilekçeler hukuki tavsiye niteliği taşımaz. Önemli hukuki işlemler için bir avukattan destek alınız.

---

## 🤝 Katkı

PR'lar açık. Özellikle:
- Daha iyi Türkçe embedding modeli denemeleri
- UI iyileştirmeleri
- Pipeline optimizasyonları

---

<div align="center">

**Türk hukuku için, Türkçe, açık kaynak**

</div>