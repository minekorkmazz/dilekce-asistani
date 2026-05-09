"""
Dilekçe RAG — Single Chunk Pipeline
=====================================
Her TXT dosyası = 1 chunk (tüm belge)

Kullanım:
    python dilekce_chunker.py --input ./dilekce_txt --output chunks.json
"""

import re
import json
import hashlib
import argparse
from pathlib import Path


# ─────────────────────────────────────────
# 1. METADATA ÇIKARIMI
# ─────────────────────────────────────────

KANUN_HUKUK_DALI = {
    "4857": "is_hukuku",
    "6098": "borclar_hukuku",
    "6102": "ticaret_hukuku",
    "5237": "ceza_hukuku",
    "5271": "ceza_hukuku",
    "2709": "anayasa_hukuku",
    "4721": "medeni_hukuk",
    "6100": "medeni_usul",
    "2577": "idare_hukuku",
    "6502": "tuketici_hukuku",
    "2004": "icra_hukuku",
    "5434": "emeklilik",
    "5352": "ceza_hukuku",
    "2828": "sosyal_hizmetler",
    "3194": "imar_hukuku",
    "5846": "fikri_mulkiyet",
}

DOSYA_ADI_KEYWORDS = {
    "trafik":   "trafik_hukuku",
    "aile":     "aile_hukuku",
    "is_":      "is_hukuku",
    "isci":     "is_hukuku",
    "icra":     "icra_hukuku",
    "nufus":    "nufus_hukuku",
    "ceza":     "ceza_hukuku",
    "tuketici": "tuketici_hukuku",
    "emekli":   "emeklilik",
    "adli":     "ceza_hukuku",
    "imar":     "imar_hukuku",
    "dis_":     "saglik_hukuku",
    "saglik":   "saglik_hukuku",
    "kira":     "borclar_hukuku",
    "miras":    "medeni_hukuk",
    "boga":     "aile_hukuku",
    "nafaka":   "aile_hukuku",
    "velayet":  "aile_hukuku",
    "sozlesme": "borclar_hukuku",
    "tasinmaz": "imar_hukuku",
    "itiraz":   "idare_hukuku",
    "idare":    "idare_hukuku",
    "vergi":    "vergi_hukuku",
    "iflas":    "icra_hukuku",
    "marka":    "fikri_mulkiyet",
    "patent":   "fikri_mulkiyet",
    "telif":    "fikri_mulkiyet",
}

def extract_metadata(text: str, filename: str) -> dict:
    fname = filename.lower()

    # Hukuk dalı — kanun numaralarından
    hukuk_dallari = set()
    for k in re.findall(r"\b(\d{4})\s*sayılı", text, re.IGNORECASE):
        if k in KANUN_HUKUK_DALI:
            hukuk_dallari.add(KANUN_HUKUK_DALI[k])

    # Hukuk dalı — dosya adı keywordlerinden
    for keyword, dal in DOSYA_ADI_KEYWORDS.items():
        if keyword in fname:
            hukuk_dallari.add(dal)

    if not hukuk_dallari:
        hukuk_dallari.add("diger")

    # Belge türü
    tl = text.lower()
    if "sözleşme" in tl or "sozlesme" in tl:
        belge_turu = "sozlesme"
    elif "itiraz" in fname:
        belge_turu = "itiraz_dilekcesi"
    elif "talep" in fname or "form" in fname:
        belge_turu = "talep_dilekcesi"
    elif "dava" in fname or "mahkemesi" in tl or "hakimliği" in tl:
        belge_turu = "dava_dilekcesi"
    else:
        belge_turu = "diger"

    # Konu satırı (varsa)
    konu_match = re.search(r"KONU\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    konu = konu_match.group(1).strip()[:200] if konu_match else ""

    return {
        "dosya_adi":     filename,
        "hukuk_dallari": ", ".join(sorted(hukuk_dallari)),
        "belge_turu":    belge_turu,
        "konu":          konu,
        "doc_id":        hashlib.md5(filename.encode()).hexdigest()[:8],
    }


# ─────────────────────────────────────────
# 2. ANA PIPELINE
# ─────────────────────────────────────────

def process_directory(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    chunks = []
    stats = {
        "toplam_dosya": 0,
        "hukuk_dali_dagilimi": {},
        "belge_turu_dagilimi": {},
    }

    txt_files = sorted(input_path.glob("**/*.txt"))
    print(f"📂 {len(txt_files)} TXT dosyası bulundu...\n")

    for fp in txt_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) < 30:
                print(f"  ⚠ {fp.name} — çok kısa, atlandı")
                continue

            metadata = extract_metadata(text, fp.name)

            chunks.append({
                "chunk_id": metadata["doc_id"],
                "text":     text,
                "metadata": metadata,
            })

            stats["toplam_dosya"] += 1
            for dal in metadata["hukuk_dallari"].split(", "):
                stats["hukuk_dali_dagilimi"][dal] = \
                    stats["hukuk_dali_dagilimi"].get(dal, 0) + 1
            bt = metadata["belge_turu"]
            stats["belge_turu_dagilimi"][bt] = \
                stats["belge_turu_dagilimi"].get(bt, 0) + 1

            print(f"  ✓ {fp.name}")
        except Exception as e:
            print(f"  ✗ {fp.name} — HATA: {e}")

    output = {"stats": stats, "chunks": chunks}
    Path(output_file).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n Tamamlandı!")
    print(f"   Dosya / Chunk : {stats['toplam_dosya']}")
    print(f"   Çıktı         : {output_file}")
    print(f"\n Hukuk Dalı Dağılımı:")
    for dal, n in sorted(stats["hukuk_dali_dagilimi"].items(), key=lambda x: -x[1]):
        print(f"   {dal:<25} {n}")
    print(f"\n Belge Türü Dağılımı:")
    for bt, n in sorted(stats["belge_turu_dagilimi"].items(), key=lambda x: -x[1]):
        print(f"   {bt:<25} {n}")


# ─────────────────────────────────────────
# 3. CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dilekçe Single-Chunk Pipeline")
    parser.add_argument("--input",  default="./dilekce_txt", help="TXT klasörü")
    parser.add_argument("--output", default="./chunks.json", help="Çıktı JSON")
    args = parser.parse_args()
    process_directory(args.input, args.output)