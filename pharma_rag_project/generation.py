import requests
import json
from typing import List, Dict

class OllamaGenerator:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initializes connection to the local Ollama instance running at T=0.0.
        """
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def format_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Formats retrieved passages into a strict context-bounded prompt.
        """
        formatted_context = ""
        for idx, doc in enumerate(contexts, 1):
            formatted_context += (
                f"--- Context Entry {idx} ---\n"
                f"Drug Name: {doc.get('Nom', 'N/A')}\n"
                f"Active Ingredients: {doc.get('Composition', 'N/A')}\n"
                f"Indications/Uses: {doc.get('Prescription', 'N/A')}\n"
                f"Dosage: {doc.get('Posologie', '[NO DATA]')}\n"
                f"Contraindications: {doc.get('Contrindications', '[NO DATA]')}\n\n"
            )

        prompt = (
            "You are a clinical decision support system. Answer the user query using ONLY the provided contexts below. "
            "Do NOT use external knowledge. If the answer is not contained in the context, output '[NO DATA]'.\n\n"
            f"RETIREVED CONTEXT:\n{formatted_context}\n"
            f"USER QUERY: {query}\n\n"
            "CLINICAL ANSWER:"
        )
        return prompt

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """
        Sends the formatted prompt to local Ollama API with Temperature = 0.0.
        """
        if not contexts:
            return "⚠️ [NO DATA] No relevant clinical records were found to answer your request."

        prompt = self.format_prompt(query, contexts)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature to enforce context-bounded determinism
                "top_p": 0.1
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"⚠️ Error from Ollama API (Status {response.status_code}): Could not generate response."
        except requests.exceptions.ConnectionError:
            return "⚠️ Error: Local Ollama service is not running. Please start Ollama using `ollama run mistral`."


if __name__ == "__main__":
    # Test execution
    llm = OllamaGenerator(model_name="mistral")
    sample_contexts = [{
        "Nom": "PARACETAMOL",
        "Composition": "Paracétamol 500mg",
        "Prescription": "Mild to moderate pain relief and fever reduction.",
        "Posologie": "1-2 tablets every 4 to 6 hours.",
        "Contrindications": "Severe hepatic impairment."
    }]
    answer = llm.generate("What is the dosage for Paracetamol?", sample_contexts)
    print("Generated LLM Answer:\n", answer)