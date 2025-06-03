import json
import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=api_key)
client = genai

def expand_query_kb(original_query: str, relevant_documents: list[str]) -> dict:
    """
    Use Gemini to perform query expansion based on relevant documents.
    
    Args:
        original_query (str): The original search query.
        relevant_documents (List[str]): A list of relevant document texts.
        
    Returns:
        dict: Contains the expanded query as {"expanded-query": "..."}
    """
    prompt = f"""
Gunakan pengetahuanmu untuk melakukan *query expansion* terhadap kalimat berikut agar pencarian dokumen menjadi lebih maksimal dan relevan.

⚠️ PENTING:
- Fokus untuk **memperluas** query asli dengan menambahkan konteks atau detail yang relevan dari dokumen, **bukan mengganti seluruh isi query asli**.
- Jika ada bagian dari query yang kurang relevan atau tidak ditemukan di dokumen, barulah boleh diganti dengan sesuatu yang lebih tepat.
- Hanya berikan **1 query hasil ekspansi terbaik**.
- JANGAN gunakan operator boolean seperti AND, OR, atau tanda kutip ganda (" ").
- Jangan berikan daftar sinonim atau alternatif dalam satu string.
- Hindari bentuk seperti: "<opsi 1>" OR "<opsi 2>".
- Hasilkan query dalam bentuk kalimat natural, eksplisit, dan spesifik seolah-olah pengguna tahu persis apa yang dicari.

🔍 Tujuan:
Jadikan query lebih fokus, kontekstual, dan sesuai dengan isi dokumen relevan tanpa kehilangan maksud aslinya.

Query asli:
{original_query}

Beberapa dokumen yang relevan:
{chr(10).join(f"- {doc}" for doc in relevant_documents)}

Berikan hasil dalam format JSON:
{{
  "expanded-query": "<query hasil ekspansi>"
}}

Contoh:
Query asli: "aturan asuransi"
Dokumen: menyebut "peraturan klaim BPJS Kesehatan 2022"
Hasil ekspansi:
{{
  "expanded-query": "aturan klaim BPJS Kesehatan tahun 2022"
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",  # Or use "gemini-1.5-pro" for more accuracy
            contents=[{"role": "user", "parts": [prompt]}],
        )
        content = response.text.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to expand query: {e}")
