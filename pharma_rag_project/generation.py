import json
import re
import urllib.request
from typing import List, Dict, Any

from summary_writer import NO_RESULTS_MESSAGE, format_records_for_prompt

SYSTEM_INSTRUCTION = """You are PharmaRAG, a friendly clinical decision-support assistant.
Explain the retrieved medicine records to the user in clear, simple, professional English,
as a helpful pharmacist would, not as a database printing its fields.

GROUNDING RULES (most important):
- Use ONLY facts found in the retrieved records below. Never add uses, doses, warnings,
  interactions, brand facts, or side effects that are not in the records.
- If a field says NOT RECORDED, clearly say that this information is not available in the
  records and cannot be confirmed here. Never guess or fill it in.
- If the records do not answer the user's question, say so plainly.
- Translate any non-English record text into English.

STYLE RULES:
- Write complete, natural sentences in short paragraphs (1-3 sentences each).
- Do NOT use "Field: value" lines, raw labels, dictionary-style output, or the text "[NO DATA]".
- Turn list items into flowing sentences, e.g. "pain relief; treatment of fever" becomes
  "relieving pain and helping manage fever".
- You may briefly explain a medical term in everyday words in brackets,
  e.g. "palpitations (a fast or pounding heartbeat)", without adding new facts about the medicine.
- Avoid robotic or repetitive phrasing. Keep it calm and professional; no emojis.
- Do not add a medical disclaimer; the application shows one separately.
- Keep the answer under about 180 words.

FORMAT (Markdown):
1. Start with a 1-2 sentence overview naming the main medicine in **bold** and what the records say it is used for.
2. Then use only the relevant short headings written as "##### Heading", chosen from:
   "##### Possible side effects", "##### Dosage and safety",
   "##### What the records don't cover", "##### Other matching medicines".
3. Under "What the records don't cover", combine all missing details into one sentence.

EXAMPLE STYLE:
**Fever-X Tablet** is listed in the available records for relieving pain and helping manage fever.

##### Possible side effects
The recorded side effects include stomach pain, nausea, and vomiting.

##### What the records don't cover
The available records do not include dosage or contraindication information for this medicine, so these details cannot be confirmed here."""

_ECHO_PREFIX_RE = re.compile(r"^\s*(?:english\s+)?(?:summary\s+)?answer\s*:\s*", re.IGNORECASE)


class OllamaGenerator:
    def __init__(
        self,
        model_name: str = "mistral",
        api_url: str = "http://localhost:11434/api/generate",
        timeout_s: float = 120.0,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.timeout_s = timeout_s

    @staticmethod
    def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"RETRIEVED RECORDS:\n{format_records_for_prompt(contexts)}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Write the answer now, following every rule above.\n\nANSWER:"
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        text = _ECHO_PREFIX_RE.sub("", (text or "").strip())
        return text.strip()

    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return NO_RESULTS_MESSAGE

        payload = {
            "model": self.model_name,
            "prompt": self.build_prompt(query, contexts),
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Ollama local API call failed: {str(e)}")

        answer = self._clean_output(result.get("response", ""))
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer
