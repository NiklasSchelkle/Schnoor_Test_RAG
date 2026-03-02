import requests
from typing import Union, Generator
import json

class Pipe:
    def __init__(self):
        self.type = "pipe"
        self.id = "schnoor_hybrid_rag"
        self.name = "SCHNOOR Hybrid RAG"

    def pipe(self, body: dict, __user__: dict = None) -> Union[str, Generator]:
        # Die gesamte Nachrichten-Historie an das Backend senden
        messages = body.get("messages", [])

        try:
            # Verbindung zum Docker-Backend mit Streaming-Support
            # stream=True ist wichtig, damit die Antwort Stück für Stück kommt
            response = requests.post(
                "http://rag_backend:8050/query",
                json={"question": messages},
                timeout=120,
                stream=True,
            )
            
            # Falls die API einen Fehler wirft (z.B. 500)
            if response.status_code != 200:
                return f"Fehler vom Backend: {response.status_code} - {response.text}"

            # Generator-Funktion, um das Streaming an die Open WebUI weiterzureichen
            def stream_response():
                # Wir iterieren über die Chunks, die von der FastAPI (api.py) kommen
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

            return stream_response()

        except Exception as e:
            return f"Verbindungsfehler zum RAG-Backend: {str(e)}"