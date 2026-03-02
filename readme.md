Dieses Repository enthält eine vollständige Pipeline für ein Retrieval-Augmented Generation (RAG) System, basierend auf Docker, Ollama und Open WebUI.

🚀 Installation & Setup

Folge diesen Schritten, um die Umgebung auf deinem Server aufzusetzen.
1. Repository klonen und vorbereiten

Zuerst das Repository herunterladen und in das Verzeichnis wechseln:

    Bash
    git clone https://github.com/NiklasSchelkle/RAG
    cd RAG
    
Oder: 

    Bash
    scp -r "C:\Users\Dell\SchnoorfinalRAG\Verda 7" root@31.22.104.187:/root/schnoor_rag
    cd schnoor_rag

2. Umgebungsvariablen konfigurieren

Erstelle oder editiere die .env Datei und füge deinen Cloudflare Tunnel-Token hinzu:

    Bash
    nano .env

3. Container starten

Baue und starte die Docker-Container im Hintergrund:

    Bash
    docker compose up -d --build

5. LLMs herunterladen

Lade die benötigten Modelle direkt in den Ollama-Container:

    Bash
    docker exec -it ollama ollama pull mistral-small:22b
    docker exec -it ollama ollama pull llama3.2:3b
    docker exec -it ollama ollama pull qwen3-embedding:4b

📂 Daten-Ingestion (RAG)

Um eigene Dokumente für das System verfügbar zu machen:

Verzeichnis erstellen:
    Bash

    mkdir -p files_to_embed

Dateien hochladen: (Beispiel von lokalem Windows-System auf den Server)
    Bash

    scp -r "PFAD*" root@ServerIP:~/RAG/files_to_embed/

Ingestion-Skript ausführen:
Verarbeite die Dokumente, um sie in die Vektordatenbank zu laden:
    Bash

    docker exec -it rag_backend python ingestion.py

🌐 Web-Interface & Konfiguration

Öffne die Weboberfläche: schnoorki.knowladgebaseai.space

Gehe in den Admin-Bereich -> Funktionen.

Füge dort den Inhalt der Datei apiopenwebui.py ein.

⚠️ Wichtige Performance-Hinweise

[!CAUTION]
VRAM Management: Wenn du die LLMs ohne spezifische Funktionen nutzt, muss der Kontext via num_ctx begrenzt werden.

Andernfalls drohen Abstürze der Modelle, da der Grafikspeicher (VRAM) überläuft!

Alle Skripte ziehen: 
     Bash
     scp -r root@31.22.104.187:/root/schnoor_rag "C:\Users\Dell\SchnoorfinalRAG\Verda 8"
