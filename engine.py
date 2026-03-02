import os
import uuid
import psycopg2
import urllib.parse
import json
import time
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv

# LangChain & KI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import CrossEncoder 

# 1. NLTK SETUP
# Einmaliger Download der Stopwords für die Schlagwort-Extraktion
nltk.download('stopwords')
GERMAN_STOPWORDS = set(stopwords.words('german'))

# 2. SETUP & MODEL LOADING
load_dotenv()

def load_models():
    # Basis-URL für alle Ollama-Aufrufe (Docker-Netzwerk)
    OLLAMA_URL = "http://ollama:11434"
    
    # keep_alive: -1 sorgt dafür, dass die Modelle im VRAM bleiben
    common_args = {"base_url": OLLAMA_URL, "keep_alive": -1, "temperature": 0}

    # NEU: Qwen3-Reranker auf GPU
    ranker = CrossEncoder(
        "Qwen/Qwen3-Reranker-0.6B",
        max_length=8192,
        device="cuda"   # GPU dank docker-compose Änderung
    )

    # Haupt-LLM: Mistral (12k Kontext für tiefe Analysen)
    llm = ChatOllama(
        model="mistral-small:22b", 
        num_ctx=12288, 
        **common_args
    )

    # Schnelles LLM: Qwen/Llama (2k Kontext für Klassifizierung)
    fast_llm = ChatOllama(
        model="qwen3:14b", 
        num_ctx=4096, 
        **common_args,
    )

    # Embeddings: Qwen3 (2k Kontext für semantische Suche)
    embeddings_model = OllamaEmbeddings(
        model="qwen3-embedding:4b", 
        base_url=OLLAMA_URL,
        num_ctx=2048 
    )

    return ranker, llm, embeddings_model, fast_llm

# Globale Instanzen
ranker, llm, embeddings_model, fast_llm = load_models()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

# 3. HILFSFUNKTION FÜR SCHLAGWORTE
def get_search_terms(question):
    words = question.lower().split()
    terms = [
        w.strip("?!.,") for w in words 
        if w.strip("?!.,") not in GERMAN_STOPWORDS and len(w.strip("?!.,")) > 1
    ]
    return terms if terms else [question.lower()]

# 4. HAUPTFUNKTION: HYBRID RAG (FÜR FACHFRAGEN)
def search_hybrid_graph(question):
    """
    Findet kleine Chunks, liefert aber den großen 
    Parent-Volltext (1750 Zeichen) an das LLM für maximalen Kontext.
    """
    search_terms = get_search_terms(question)
    sql_keywords = [f"%{t}%" for t in search_terms]
    ts_query_string = " ".join(search_terms)

    # Vektor erstellen
    query_vector = embeddings_model.embed_query(question)
    
    conn = get_connection()
    cur = conn.cursor()

    # STUFE 1: Hybrid-Suche (Vektor + Keyword Boost + Summary)

    start_db = time.time()

    MIN_SIMILARITY = 0.60

    cur.execute("""
        WITH vector_matches AS (
            -- 🔵  Feste Quote - Top 25 Vektortreffer
            SELECT 
                p.id as p_id,
                (1 - (c.embedding <=> %s::halfvec)) as score
            FROM document_chunks c
            JOIN parent_documents p ON c.parent_id = p.id
            WHERE (1 - (c.embedding <=> %s::halfvec)) > %s
            ORDER BY score DESC
            LIMIT 25
        ),
        keyword_matches AS (
            -- 🟡  Feste Quote - Top 15 Keywordtreffer
            SELECT 
                p.id as p_id,
                ts_rank_cd(to_tsvector('german', c.content), plainto_tsquery('german', %s)) as score
            FROM document_chunks c
            JOIN parent_documents p ON c.parent_id = p.id
            WHERE (to_tsvector('german', c.content) @@ plainto_tsquery('german', %s)
                OR p.full_text ILIKE ANY(%s))
            ORDER BY score DESC
            LIMIT 15
        ),
        combined_matches AS (
            -- Ergebnisse zusammenführen
            SELECT p_id, score FROM vector_matches
            UNION ALL
            SELECT p_id, score FROM keyword_matches
        ),
        best_unique_matches AS (
            -- Dubletten entfernen (falls in beiden Suchen gefunden)
            SELECT p_id, MAX(score) as final_score
            FROM combined_matches
            GROUP BY p_id
        )
        -- Finale Daten für Python abholen
        SELECT 
            p.full_text, p.title, p.source_url, p.id, 
            'no_content_needed', p.summary
        FROM best_unique_matches b
        JOIN parent_documents p ON b.p_id = p.id
        ORDER BY b.final_score DESC;
    """, (query_vector, query_vector, MIN_SIMILARITY, ts_query_string, ts_query_string, sql_keywords))
    
    rows = cur.fetchall()
    db_duration = time.time() - start_db # Zeitmessung Ende DB

    passages = []
    unique_docs_map = {} 
    
    for r in rows:
        # Zuordnung basierend auf deinem SQL-Query:
        # r[0]=parent_text, r[1]=title, r[2]=url, r[3]=p_id, r[4]=chunk_content, r[5]=summary
        parent_text, title, url, summary, content = r[0], r[1], r[2], r[5], r[4]  # content ist der kleine Chunk, parent_text ist der große Kontext
        
        if title not in unique_docs_map:
            unique_docs_map[title] = urllib.parse.quote(url, safe=':/?&=')

        # Wir packen alles in die Passages für Flashrank
        passages.append({
            "id": str(len(passages)),
            "text": parent_text, # Flashrank bewertet die Relevanz basierend auf dem parent_text. Wenn ich hier den content nehme, wird der kleine Chunkbewertet. 
            "meta": {
                "title": title,
                "url": url,
                "summary": summary,
                "parent_text": parent_text # Hier speichern wir den 1750er Block für später
            }
        })
    
    if not passages:
        cur.close()
        conn.close()
        return "Keine relevanten SCHNOOR-Dokumente gefunden.", "", []

    # STUFE 2: Re-Ranking (Flashrank)
    start_rerank = time.time()
    
    instruction = "Finde relevante Abschnitte aus Schnoor-Dokumenten für die folgende Frage:"
    pairs = [
        (f"Instruct: {instruction}\nQuery: {question}", p["meta"]["parent_text"])
    for p in passages
    ]

    scores = ranker.predict(pairs, show_progress_bar=False, batch_size=1)
    for i, p in enumerate(passages):
        p["_score"] = float(scores[i])

    top_results = sorted(passages, key=lambda x: x["_score"], reverse=True)[:6]

    rerank_duration = time.time() - start_rerank # Zeitmessung Ende Reranking

    # LOGGING der Zeiten in die Konsole
    print(f"--- ⏱️ ZEITMESSUNG SCHNOOR-ENGINE ---")
    print(f"DB-Suche & Deduplizierung: {db_duration:.3f}s")
    print(f"Reranking ({len(passages)} große Blöcke): {rerank_duration:.3f}s")
    print(f"Gesamt-Latenz Engine: {db_duration + rerank_duration:.3f}s")
    print(f"------------------------------------")

    context_text = ""
    found_documents_list = []
    seen_titles = set()

    for res in top_results:
        meta = res['meta']
        safe_url = urllib.parse.quote(meta['url'], safe=':/?&=')
        
        # PROMPT-GEBÄUDE: Hier bekommt das LLM den Dokument-Anker (Summary)
        context_text += f"\n### QUELLE: {meta['title']}\n"
        context_text += f"DOKUMENT-INFO (Zusammenfassung): {meta.get('summary', 'Keine Info')}\n"
        context_text += f"RELEVANTER ABSCHNITT:\n{meta.get('parent_text', res['text'])}\n"  # hier wird der froße Chunk aus den metadaten genommen. Obwohl er halt auch in res['text'] ist. text ist aber nur ein Fallback. es wird also nix doppelt injected.  
        
        if meta['title'] not in seen_titles:
            found_documents_list.append({"title": meta['title'], "url": safe_url})
            seen_titles.add(meta['title'])

    cur.close()
    conn.close()

    # Rückgabe: context, graph (jetzt leer), doc_list
    return context_text, "", found_documents_list


def search_documents_only(question):
    """
    Spezialisierte Suche für den SEARCH-Modus.
    Nutzt Embeddings der Chunks, holt aber die Links vom Parent-Dokument.
    """
    
    # 1. Zentraler Aufruf der Hilfsfunktion
    search_terms = get_search_terms(question)
    sql_keywords = [f"%{t}%" for t in search_terms]
    
    # 2. Embedding erstellen (Semantische Suche über Qwen3)
    query_vector = embeddings_model.embed_query(question)
    
    conn = get_connection()
    cur = conn.cursor()

    # 3. Hybrid-Suche mit JOIN und Gruppierung
    # Da ein Dokument mehrere Chunks hat, nehmen wir den besten Treffer (MAX score)
    # und gruppieren nach dem Titel, damit jedes PDF nur einmal erscheint.
    MIN_SIMILARITY = 0.65 

    cur.execute("""
        WITH vector_matches AS (
            SELECT 
                p.title, 
                p.source_url, 
                MAX(1 - (c.embedding <=> %s::halfvec)) as score
            FROM document_chunks c
            JOIN parent_documents p ON c.parent_id = p.id
            WHERE (1 - (c.embedding <=> %s::halfvec)) > %s
            GROUP BY p.title, p.source_url
        ),
        keyword_matches AS (
            SELECT title, source_url, 1.0 as score
            FROM parent_documents
            WHERE title ILIKE ANY(%s)
            GROUP BY title, source_url
        )
        -- Ergebnisse kombinieren, Duplikate entfernen und nach Relevanz sortieren
        SELECT title, source_url 
        FROM (
            SELECT title, source_url, score FROM vector_matches
            UNION ALL
            SELECT title, source_url, score FROM keyword_matches
        ) AS combined
        ORDER BY score DESC
        LIMIT 35;
    """, (query_vector, query_vector, MIN_SIMILARITY, sql_keywords))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 4. Ergebnisse für die API aufbereiten
    found_docs = []
    seen = set()
    for r in rows:
        title, url = r[0], r[1]
        if title not in seen:
            safe_url = urllib.parse.quote(url, safe=':/?&=')
            found_docs.append({"title": title, "url": safe_url})
            seen.add(title)

    return "Dokument-Suche abgeschlossen.","", found_docs


