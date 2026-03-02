import os
import uuid
import json
import psycopg2
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 1. SETUP
load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", 
    ".asc", ".md", ".txt", ".png", ".jpg", ".jpeg", ".tiff"
}

# Basis-URL für den späteren Zugriff über WebUI (Standardmäßig localhost für Tests)
# In der .env Datei kannst du FILE_SERVER_URL=https://dein-server.de/files/ setzen
FILE_SERVER_BASE_URL = "https://schnoorki.knowladgebaseai.space/download/"

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

# KI-Modelle

# OLLAMA LOKAL 
embeddings_model = OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://ollama:11434", num_ctx=2048) 

llm = ChatOllama(model="mistral-small:22b",base_url="http://ollama:11434", temperature=0,num_ctx=16384)


# 2 . Generiert eine prägnante Zusammenfassung des gesamten Dokuments, um den Kontext der Chunks zu verstehen.
def generate_document_summary(text):
    """
    Erstellt eine prägnante Zusammenfassung des gesamten Dokuments.
    Dies dient als 'Anker' für die KI, um den Kontext der Chunks zu verstehen.
    """
    prompt = f"""
    Du bist der Wissensmanager der Schnoor Industrieelektronik GmbH. 
    Fasse das vorliegende Dokument in genau 3 Sätzen zusammen.
    Konzentriere dich auf:
    1. Den Hauptzweck des Dokuments. Was ist es für ein Dokument? (z.B. Wartungsplan, Projektbericht, Produktdatenblatt, Einweisung ins Unternehemen etc...).
    2. Das konkrete Projekt oder Produkt (falls genannt).
    3. Die wichtigste Kernaussage.

    Text: {text[:20000]} 
    """
    try:
        # Mistral-Modell für die inhaltliche Zusammenfassung
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ Fehler bei Summary-Erstellung: {e}")
        return "Keine Zusammenfassung verfügbar."

# 3. VERARBEITUNGSLOGIK
def ingest_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS: return

    filename = os.path.basename(file_path)
    print(f"\n🚀 Verarbeite Dokument: {filename}")
    
    try:
        # A. Konvertierung mit Docling  
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        # B. Zusammenfassung erstellen (Der neue "Anker")
        print(f"   📝 Generiere Dokument-Zusammenfassung...")
        doc_summary = generate_document_summary(markdown_text)
        
        # C. Chunking
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1750, chunk_overlap=250)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        
        conn = get_connection()
        cur = conn.cursor()
        
        parent_chunks = parent_splitter.split_text(markdown_text)
        web_url = f"{FILE_SERVER_BASE_URL}{filename}"
        
        for idx, p_text in enumerate(parent_chunks):
            p_id = str(uuid.uuid4())
            
            # 1. Parent speichern (JETZT MIT SUMMARY)
            cur.execute("""
                INSERT INTO parent_documents (id, title, full_text, source_url, summary) 
                VALUES (%s, %s, %s, %s, %s)
            """, (p_id, filename, p_text, web_url, doc_summary))
            
            # 2. Vektor-Speicherung der Child-Chunks
            child_chunks = child_splitter.split_text(p_text)
            if child_chunks:
                vectors = embeddings_model.embed_documents(child_chunks)
                for c_text, vec in zip(child_chunks, vectors):
                    cur.execute("""
                        INSERT INTO document_chunks (id, parent_id, content, embedding) 
                        VALUES (%s, %s, %s, %s::halfvec)
                    """, (str(uuid.uuid4()), p_id, c_text, vec))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Fertig indexiert: {filename}")
        
    except Exception as e:
        print(f"❌ Schwerer Fehler bei {file_path}: {e}")

# 4. STARTPUNKT (Inkrementelles Update)
if __name__ == "__main__":
    doc_dir = os.getenv("DOC_DIR")
    if not doc_dir or not os.path.exists(doc_dir):
        print("Fehler: DOC_DIR nicht konfiguriert.")
    else:
        try:
            conn = get_connection()
            cur = conn.cursor()
            # Nur Dateien laden, die noch nicht in der Datenbank existieren
            cur.execute("SELECT DISTINCT title FROM parent_documents")
            indexed_files = {row[0] for row in cur.fetchall()}
            cur.close()
            conn.close()
            print(f"Status: {len(indexed_files)} Dokumente bereits indexiert.")
        except:
            indexed_files = set()

        files = [f for f in os.listdir(doc_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        for filename in files:
            if filename in indexed_files:
                print(f"⏩ Überspringe: {filename} (Bereits vorhanden)")
                continue
            ingest_document(os.path.join(doc_dir, filename))


