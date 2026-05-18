import asyncio
import re
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

ORNEK_DILEKCE = """
ADALET BAKANLIĞI
ADLİ SİCİL İSTATİSTİK GENEL MÜDÜRLÜĞÜ'NE

5352 sayılı Adli Sicil Kanunu madde 12 uyarınca kaydımın silinmesini talep ediyorum.
4857 sayılı İş Kanunu kapsamında haklarım saklıdır.
2577 sayılı İdari Yargılama Usulü Kanunu hükümleri çerçevesinde...
"""

def kanun_no_cikar(metin: str) -> list:
    return list(set(re.findall(r'\b(\d{4})\s*sayılı', metin, re.IGNORECASE)))

async def kanun_sorgula(session, kanun_no: str) -> dict:
    result = await session.call_tool(
        "search_mevzuat",
        {
            "mevzuat_no": kanun_no,
            "mevzuat_tur": "KANUN",
            "page_size": 5
        }
    )
    text = result.content[0].text
    # Sonuçtan kanun adını çek
    match = re.search(rf'\[{kanun_no}\]\s+(.+?)\s+\(', text)
    ad = match.group(1) if match else "Bilinmiyor"
    bulundu = kanun_no in text
    return {"no": kanun_no, "ad": ad, "bulundu": bulundu, "ham": text[:200]}

async def main():
    url = "https://mevzuat.surucu.dev/mcp"

    kanun_nolari = kanun_no_cikar(ORNEK_DILEKCE)
    print(f"📋 Dilekçeden çıkarılan kanun numaraları: {kanun_nolari}\n")

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for no in kanun_nolari:
                sonuc = await kanun_sorgula(session, no)
                durum = "✅ Güncel" if sonuc["bulundu"] else "⚠️ Bulunamadı"
                print(f"  {durum} | {sonuc['no']} sayılı {sonuc['ad']}")

if __name__ == "__main__":
    asyncio.run(main())