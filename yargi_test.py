import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import json

async def main():
    url = "https://yargimcp.surucu.dev/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for attempt in range(3):
                result = await session.call_tool(
                    "search_bedesten_unified",
                    {
                        "phrase": "kıdem tazminatı haksız fesih",
                        "court_types": ["YARGITAYKARARI"],
                        "birimAdi": "H9",
                    }
                )
                data = json.loads(result.content[0].text)

                if data.get("error") == "rate_limit_exceeded":
                    await asyncio.sleep(float(data.get("retry_after", 4)) + 1)
                    continue

                # İlk kararın tüm alanlarını göster
                if data.get("decisions"):
                    print("İlk kararın tüm alanları:")
                    print(json.dumps(data["decisions"][0], ensure_ascii=False, indent=2))
                break

if __name__ == "__main__":
    asyncio.run(main())