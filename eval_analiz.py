"""
Retrieval Eval Sonuç Analizi
"""
import json
from pathlib import Path

data = json.loads(Path("retrieval_eval_results.json").read_text(encoding="utf-8"))
sonuclar = data["sonuclar"]

print(f"Özet: {data['ozet']}\n")

# 0 puan alanlar
print("❌ 0 puan alanlar:")
for s in sonuclar:
    if s["puanlar"][0] == 0:
        print(f"  Kaynak: {s['kaynak_sablon']}")
        print(f"  Sorgu : {s['uretilen_sorgu']}")
        print(f"  Gelen : {s['retrieval_hits']}")
        print(f"  Açıkl : {s['aciklama']}")
        print()

# 1 puan alanlar
print("⚠️  1 puan alanlar:")
for s in sonuclar:
    if s["puanlar"][0] == 1:
        print(f"  Kaynak: {s['kaynak_sablon']}")
        print(f"  Sorgu : {s['uretilen_sorgu']}")
        print(f"  Gelen : {s['retrieval_hits']}")
        print()

# Hit@3 hesapla
hit3 = sum(1 for s in sonuclar if any(p >= 4 for p in s["puanlar"]))
print(f"Hit@3 (≥4 puan): {hit3}/{len(sonuclar)} = %{hit3/len(sonuclar)*100:.1f}")
EOF