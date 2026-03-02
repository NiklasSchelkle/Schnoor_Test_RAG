from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Any
import uvicorn
from engine import search_hybrid_graph, search_documents_only, llm, fast_llm
import os 
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse
import json
import asyncio
import time
import uuid # Neu für Request-Tracking

app = FastAPI(title="SCHNOOR Hybrid RAG API")

# 1. CORS SETUP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DATEI-PFAD
INTERNAL_STORAGE_PATH = "/app/data/ingest"
if os.path.exists(INTERNAL_STORAGE_PATH):
    app.mount("/download", StaticFiles(directory=INTERNAL_STORAGE_PATH), name="download")
    print(f"✅ Lokaler File-Server auf {INTERNAL_STORAGE_PATH} aktiv.")
else:
    print(f"⚠️ Warnung: Pfad {INTERNAL_STORAGE_PATH} nicht gefunden!")

class ChatQuery(BaseModel):
    question: Any 

@app.post("/query")
async def handle_query(query: ChatQuery):
    start_total = time.time()
    req_id = str(uuid.uuid4())[:8] # ID für dieses spezifische Logging
    
    # A. Daten aus Open WebUI sortieren
    if isinstance(query.question, list):
        messages = query.question
        last_user_message = messages[-1]["content"] if messages else ""
    else:
        messages = []
        last_user_message = query.question

    print(f"\n{'='*80}\n[REQ-ID: {req_id}] EINGANG: {last_user_message}\n{'='*80}")


    # B. System-Schleuse
    technical_keywords = ["### Task:", "JSON format:", "concise", "summarizing", "title with", "Generate a short title", "Summarize the conversation", "Return only the title", "based on the previous" ]
    if any(k in last_user_message for k in technical_keywords):
        print(f"[{req_id}] ⚡ System-Task erkannt.")
        fast_res = fast_llm.invoke([("user", last_user_message)])
        return StreamingResponse(iter([fast_res.content]), media_type="text/plain")

    # C. QUERY REWRITING & INTENT CHECK
    start_intent = time.time()
    history_context = ""
    if len(messages) > 1:
        for msg in messages[-4:-1]:
            clean_content = msg['content'].split("### Referenzen:")[0].strip()
            history_context += f"{msg['role']}: {clean_content}\n"

    rewrite_prompt = f"""
    Du bist der Klassifizierer für Anfragen und entscheidest, ob die Frage INTERNE SCHNOOR-Infos braucht oder ALLGEMEIN ist.
    Die Schnoor Industrieelektronik GmbH & Co. KG ist ein Unternehmen, welches auf Objektfunk / BOS-Funk (analoge & digitale Systeme, Maritime Kommunikationslösungen und Transport-, Industri-, und Energiekommunikation spezialisiert ist.
    Außerdem musst  du Neben der Klassifizierung perfekte Suche Begriffe für ein RAGSystem generieren.
    Gib NUR das folgende Format zurück:
    TYPE: [SMALLTALK, RAG oder SEARCH]
    QUERY: [Präzise Suchbegriffe]

    ### SCHRITT 1: KLASSIFIZIERUNG (TYPE)

    - SMALLTALK: Begrüßung, Danke, Geplänkel oder allgemeine Fragen ohne SCHNOOR-Bezug.
    - SEARCH: Nutzer möchte explizit Dokumente finden oder suchen, eine Liste von Dateien sehen oder fragt "Welche Dokumente/PDFs gibt es zu...".
    - RAG: Fachfrage zu Inhalten, Projekten oder Wissen, die eine erklärende Antwort aus den Schnoor-Dokumenten benötigt.
      Wichtig ist zudem. Dass die Schnoor-Dokumente nicht nur technisch sind, sondern auch Fragen zu Mitarbeitern,Unternehmenssoftware, Untenehmenspersonen und Abläufen beihnalten können.
    - Fokussiere dich hierbei auf die Aktuelle Frage.

    ### SCHRITT 2: SUCHBEGRIFF-OPTIMIERUNG (QUERY)

    - Die QUERY darf NUR aus den Kern-Suchbegriffen bestehen.
    - KEINE URLs, KEINE Markdown-Links, KEIN "history:", KEINE Sonderzeichen wie & oder |.
    - Die Query wird direkt an eine Datenbank-Suche gereicht. Halte sie kurz und präzise.
    - Die QUERY sollte NUR den Namen der Person und oder das konkrete Fachthema enthalten.
    - Keine Füllwörter, keine Höflichkeitsfloskeln, keine Kontext-Infos. Nur die nackten Suchbegriffe. Keine Ganzen Sätze!
    - Entferne den Firmennamen "Schnoor", da die Suche bereits im Schnoor-Archiv stattfindet.
    - WICHTIG: Wenn die aktuelle Frage ein neues Thema anspricht, ignoriere den Inhalt des Verlaufs komplett!
    - Nutze den Verlauf nur, wenn nach dem einem bestimmten vorherigen Inhalt gefragt wurde!

    ### Antworte Streng in diesem FORMAT (STRENG EINHALTEN)
    TYPE: [Hier nur SMALLTALK, SEARCH oder RAG einsetzen]
    QUERY: [Hier nur die nackten Suchbegriffe einsetzen]


    VERLAUF:
    {history_context}

    AKTUELLE FRAGE:
    {last_user_message}

    """
    
    try:
        rewrite_res = fast_llm.invoke([("user", rewrite_prompt)])
        res_text = rewrite_res.content.strip()
        intent_type = "SEARCH" if "TYPE: SEARCH" in res_text else "SMALLTALK" if "TYPE: SMALLTALK" in res_text else "RAG"
        search_query = res_text.split("QUERY:")[1].strip() if "QUERY:" in res_text else last_user_message
        print(f"[{req_id}] ⏱️ Intent: {intent_type} | Query: {search_query} ({time.time() - start_intent:.2f}s)")
    except Exception as e:
        print(f"[{req_id}] ⚠️ Intent-Fehler: {e}")
        intent_type, search_query = "RAG", last_user_message

    # D. HYBRID SUCHE (Engine-Aufruf)
    start_search = time.time()
    if intent_type == "SMALLTALK":
        context, graph, all_found_docs = "", "", []
    elif intent_type == "SEARCH":
        print(f"[{req_id}] 🔍 SEARCH-MODUS aktiv.")
        context, graph, all_found_docs = search_documents_only(search_query)
    else:
        print(f"[{req_id}] 🧠 RAG-MODUS aktiv.")
        context, graph, all_found_docs = search_hybrid_graph(search_query)

    print(f"[{req_id}] ⏱️ Suche abgeschlossen: {time.time() - start_search:.2f}s")
    
    # --- DEEP LOGGING: Gefundener Kontext ---
    if intent_type != "SMALLTALK":
        print(f"\n--- [DEBUG: RETRIEVED CONTEXT FOR {req_id}] ---")
        print(context if context else "KEIN KONTEXT GEFUNDEN!")
        print("--- [END CONTEXT] ---\n")

    # E. SYSTEM PROMPTS
    if intent_type == "SEARCH":
        if not all_found_docs:
            current_system_prompt = "Informiere den Nutzer höflich darüber, dass das Archiv zu dieser Suche aktuell keine Treffer liefert. Erkläre, dass es daran liegen könnte, dass noch keine entsprechenden Dokumente eingelesen wurden."
        else:
            current_system_prompt = "Du bist der SCHNOOR-Archivar. Bestätige kurz, dass Dokumente im Archiv gefunden wurden. Liste die DOKUMENTE und Quellen NICHT auf!Sie erscheinen automatisch."
            
    elif intent_type == "SMALLTALK":
        current_system_prompt = "Du bist ein freundlicher SCHNOOR-Assistent. Antworte charmant auf Begrüßungen oder die Smalltalkfrage. Weise darauf hin, dass du dem User bei Fragen zu SCHNOOR-Projekten, internem Wissen und bei Dokumentsuchen helfen kannst."
        
    else: # RAG-Modus
        current_system_prompt = f"""
        ### DEINE ROLLE ###
        Du bist der offizielle Wissensexperte von Schnoor. Antworte basierend auf der Datengrundlage.
        Dein Ziel: Maximale Vollständigkeit und Korrektheit.

        Nur zu deiner Information: Die Schnoor Industrieelektronik GmbH & Co. KG ist ein Unternehmen, welches auf Objektfunk / BOS-Funk (analoge & digitale Systeme, Maritime Kommunikationslösungen und Transport-, Industri-, und Energiekommunikation spezialisiert ist.
        
        ### VERHALTEN ###
        Nutze die DATENGRUNDLAGE (Summaries & Text) als absolute Priorität.
        Nutze die DOKUMENT-INFO (Zusammenfassung), um den Kontext der Informationen zu verstehen.
        BEGRÜẞUNG & SMALLTALK: Wenn der Nutzer dich grüßt oder allgemeine Fragen stellt (z.B. "Wie geht es dir?", "Wer bist du?"), antworte charmant und hilfsbereit mit deinem eigenen Wissen.
    
        
        ### DATENGRUNDLAGE ###
        {context}

        ### ARBEITSANWEISUNG ###
        - Antworte direkt, präzise und fachlich korrekt.
        - Nutze alle relevanten Informationen aus dem TEXT-KONTEXT. Vermeide es, Informationen zu erfinden, die nicht in den Dokumenten stehen.
        - Erstelle KEINE eigene Quellenliste.
        - Falls keine Daten vorliegen: "Dazu liegen keine internen Dokumente vor, aber allgemein bekannt ist...". In solch einem Fall darfst du mit deinem allgemeinen Wissen antworten. Aber keine Informationen Erfinden! Weise gerne darauf hin, dass noch kein Dokument zu diesem Thema eingelesen wurde.
        """

    # --- DEEP LOGGING: Finaler Prompt ---
    print(f"[{req_id}] 📝 FINAL SYSTEM PROMPT SENT TO LLM:")
    print("-" * 40)
    print(current_system_prompt)
    print("-" * 40 + "\n")

    # F. LLM MESSAGES & STREAMING
    llm_messages = [("system", current_system_prompt)]
    for msg in messages[:-1]:
        role = "user" if msg.get("role") == "user" else "assistant"
        llm_messages.append((role, msg.get("content", "")))
    llm_messages.append(("user", last_user_message))

    async def response_generator():
        print(f"[{req_id}] 🚀 LLM Stream startet...")
        try:
            for chunk in llm.stream(llm_messages):
                if chunk.content:
                    yield chunk.content

            if intent_type in ["RAG", "SEARCH"] and all_found_docs:
                yield "\n\n---\n### 🔍 Referenzen:\n"
                seen = set()
                for doc in all_found_docs:
                    if doc['title'] not in seen:
                        yield f"- [{doc['title']}]({doc['url']})\n"
                        seen.add(doc['title'])
            
            print(f"[{req_id}] ✅ Fertig. Gesamtzeit: {time.time() - start_total:.2f}s")
        except Exception as e:
            print(f"[{req_id}] ❌ Fehler: {str(e)}")
            yield f"\n\n[Fehler im Stream: {str(e)}]"

    return StreamingResponse(response_generator(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
