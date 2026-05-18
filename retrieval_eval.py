"""
Retrieval Değerlendirme — LLM-as-Judge
========================================
RAG sisteminin retrieval kalitesini ölçer.

Adımlar:
1. chunks.json'dan rastgele 50 şablon seç
2. Her şablon için LLM'e "bu şablona uygun kullanıcı sorgusu üret" dedirt
3. O sorguyu RAG'a ver, top-3 şablon getir
4. LLM'e "Bu şablon bu sorguya uygun mu? 1-5 puan ver" dedirt
5. Sonuçları raporla

Kullanım:
    pip install groq chromadb sentence-transformers tqdm
    set GROQ_API_KEY=gsk_...
    python retrieval_eval.py
"""

import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────

CHUNKS_FILE     = "chunks.json"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "dilekce_turkembed"
EMBED_MODEL     = "trmteb/turkish-embedding-model"
GROQ_MODEL      = "llama-3.3-70b-versatile"
N_SAMPLES       = 50   # kaç şablon test edilsin
TOP_K           = 3    # kaç şablon getirilsin
RESULTS_FILE    = "retrieval_eval_results.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────
# 1. SORGU ÜRETME
# ─────────────────────────────────────────

def sorgu_uret(sablon_metni: str, client: Groq) -> str:
    """Şablona bakarak gerçekçi bir kullanıcı sorgusu üret."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Verilen hukuki dilekçe şablonuna bakarak, "
                    "bu dilekçeyi yazmak isteyen bir vatandaşın "
                    "sade Türkçeyle yazacağı 1-2 cümlelik problemi yaz. "
                    "Sadece problemi yaz, başka hiçbir şey yazma."
                )
            },
            {
                "role": "user",
                "content": sablon_metni[:800]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# 2. RETRIEVAL
# ─────────────────────────────────────────

def retrieve(query: str, model, collection, top_k: int = TOP_K) -> list:
    q_vec   = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "score":     round(1 - results["distances"][0][i], 4),
            "dosya_adi": results["metadatas"][0][i]["dosya_adi"],
            "text":      results["documents"][0][i][:500],
        })
    return hits


# ─────────────────────────────────────────
# 3. LLM-AS-JUDGE
# ─────────────────────────────────────────

def judge(sorgu: str, sablon_dosya: str, getirilen_sablonlar: list, client: Groq) -> dict:
    """
    LLM'e sor: getirilen şablonlar bu sorgu için uygun mu?
    Her şablon için 1-5 puan ver.
    """
    sablonlar_str = "\n\n".join([
        f"ŞABLON {i+1} ({h['dosya_adi']}, skor: {h['score']}):\n{h['text']}"
        for i, h in enumerate(getirilen_sablonlar)
    ])

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir hukuk asistanısın. "
                    "Kullanıcının sorgusu ve getirilen dilekçe şablonlarını değerlendir. "
                    "Her şablon için 1-5 arası puan ver (5=çok uygun, 1=hiç uygun değil). "
                    "Sadece JSON formatında yanıt ver: "
                    "{\"puan_1\": X, \"puan_2\": Y, \"puan_3\": Z, \"en_iyi\": N, \"aciklama\": \"...\"}"
                )
            },
            {
                "role": "user",
                "content": f"KULLANICI SORGUSU: {sorgu}\n\nGETİRİLEN ŞABLONLAR:\n{sablonlar_str}"
            }
        ],
        temperature=0,
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()

    # JSON parse — birkaç farklı formatı dene
    def parse_json(t):
        import re
        # 1. Direkt parse
        try:
            return json.loads(t)
        except Exception:
            pass
        # 2. Code block içinden çıkar
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # 3. İlk { } bloğunu bul
        match = re.search(r"\{[^{}]+\}", t, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        # 4. Puanları regex ile çek
        p1 = re.search(r'puan_1["\':\s]+(\d)', t)
        p2 = re.search(r'puan_2["\':\s]+(\d)', t)
        p3 = re.search(r'puan_3["\':\s]+(\d)', t)
        if p1:
            return {
                "puan_1": int(p1.group(1)),
                "puan_2": int(p2.group(1)) if p2 else 0,
                "puan_3": int(p3.group(1)) if p3 else 0,
                "en_iyi": 1,
                "aciklama": "regex parse"
            }
        return None

    result = parse_json(text)
    if result:
        return result
    return {"puan_1": 0, "puan_2": 0, "puan_3": 0, "en_iyi": 0, "aciklama": "parse hatası"}


# ─────────────────────────────────────────
# 4. ANA DEĞERLENDİRME
# ─────────────────────────────────────────

def main():
    print("🔧 Sistem yükleniyor...\n")

    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = SentenceTransformer(EMBED_MODEL)

    # Chroma bağlantısı
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        chroma_client = chromadb.HttpClient(host=chroma_host, port=int(os.getenv("CHROMA_PORT", 8000)))
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Chunks yükle
    data   = json.loads(Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    chunks = data["chunks"]

    # Rastgele N_SAMPLES şablon seç
    random.seed(42)
    ornekler = random.sample(chunks, min(N_SAMPLES, len(chunks)))
    print(f"📋 {len(ornekler)} şablon seçildi\n")

    # Değerlendirme
    sonuclar = []
    toplam_puan_1 = 0  # ilk sonucun puanı
    hit_at_3      = 0  # top-3'te 4+ puanlı sonuç var mı

    for i, ornek in enumerate(tqdm(ornekler, desc="Değerlendirme")):
        try:
            # 1. Sorgu üret
            sorgu = sorgu_uret(ornek["text"], groq_client)
            time.sleep(0.5)  # rate limit

            # 2. Retrieval
            hits = retrieve(sorgu, embed_model, collection)

            # 3. Judge
            degerlendirme = judge(sorgu, ornek["metadata"]["dosya_adi"], hits, groq_client)
            time.sleep(0.5)

            # Metrikleri hesapla
            puan_1 = degerlendirme.get("puan_1", 0)
            puanlar = [
                degerlendirme.get("puan_1", 0),
                degerlendirme.get("puan_2", 0),
                degerlendirme.get("puan_3", 0),
            ]
            top3_hit = any(p >= 4 for p in puanlar)

            toplam_puan_1 += puan_1
            if top3_hit:
                hit_at_3 += 1

            sonuc = {
                "ornek_no":       i + 1,
                "kaynak_sablon":  ornek["metadata"]["dosya_adi"],
                "uretilen_sorgu": sorgu,
                "retrieval_hits": [h["dosya_adi"] for h in hits],
                "puanlar":        puanlar,
                "en_iyi":         degerlendirme.get("en_iyi", 0),
                "aciklama":       degerlendirme.get("aciklama", ""),
            }
            sonuclar.append(sonuc)

            # Her örnekten sonra ara kaydet
            n_tmp = len(sonuclar)
            output_tmp = {
                "ozet": {
                    "n_ornk":       n_tmp,
                    "ort_puan_1":   round(toplam_puan_1 / n_tmp, 2),
                    "hit_at_3_pct": round(hit_at_3 / n_tmp * 100, 1),
                },
                "sonuclar": sonuclar
            }
            Path(RESULTS_FILE).write_text(
                json.dumps(output_tmp, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        except Exception as e:
            print(f"\n  [!] {i+1}. örnek hatası: {e}")
            continue

    # Rapor
    n = len(sonuclar)
    ort_puan = toplam_puan_1 / n if n > 0 else 0
    hit_rate = hit_at_3 / n if n > 0 else 0

    print(f"\n{'='*50}")
    print(f"📊 DEĞERLENDİRME SONUÇLARI")
    print(f"{'='*50}")
    print(f"  Test edilen örnek   : {n}")
    print(f"  İlk sonuç ort. puan : {ort_puan:.2f} / 5")
    print(f"  Hit@3 (≥4 puan)     : %{hit_rate*100:.1f}")
    print(f"{'='*50}\n")

    # Puan dağılımı
    from collections import Counter
    dagilim = Counter()
    for s in sonuclar:
        dagilim[s["puanlar"][0]] += 1
    print("İlk sonuç puan dağılımı:")
    for puan in sorted(dagilim.keys(), reverse=True):
        bar = "█" * dagilim[puan]
        print(f"  {puan} puan: {bar} ({dagilim[puan]})")

    # Kaydet
    output = {
        "ozet": {
            "n_ornk":        n,
            "ort_puan_1":    round(ort_puan, 2),
            "hit_at_3_pct":  round(hit_rate * 100, 1),
        },
        "sonuclar": sonuclar
    }
    Path(RESULTS_FILE).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n💾 Detaylı sonuçlar: {RESULTS_FILE}")


if __name__ == "__main__":
    main()