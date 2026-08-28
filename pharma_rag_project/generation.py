import json
import urllib.request
from typing import List, Dict, Any

class OllamaGenerator:
    def __init__(self, model_name: str = "mistral", api_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "⚠️ [NO DATA] No matching clinical records found in the database."

        context_str = "\n---\n".join([str(c) for c in contexts])
        
        system_instruction = (
            "You are a clinical decision support assistant. "
            "Always respond strictly in English regardless of the language of the source documents. "
            "Synthesize the answer using ONLY the context provided below. "
            "If context fields contain French or another language, translate the meaning accurately into English. "
            "Do not invent medical information not supported by context."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"Retrieved Clinical Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"English Summary Answer:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama local API call failed: {str(e)}")