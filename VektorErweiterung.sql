-- =================================================================
-- 1. ERWEITERUNGEN
-- =================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =================================================================
-- 2. TABELLEN-STRUKTUR (OPTIMIERT FÜR CONTEXTUAL RAG)
-- =================================================================

-- A. Elterndokumente (Mit dem neuen Dokument-Anker: summary)
CREATE TABLE IF NOT EXISTS parent_documents (
    id UUID PRIMARY KEY,
    title TEXT,
    full_text TEXT,
    metadata JSONB,
    source_url TEXT,
    summary TEXT, -- <== DER NEUE ANKER FÜR DEN KONTEXT
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- B. Dokument-Chunks (Vektorsuche)
-- Nutzt halfvec(2560) für das Qwen3-Embedding Modell
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
    content TEXT,
    embedding halfvec(2560) 
);

-- =================================================================
-- 3. PERFORMANCE-OPTIMIERUNG (INDIZES)
-- =================================================================

-- Vektor-Index (IVFFlat oder HNSW)
-- Da du 2560 Dimensionen nutzt, kannst du auch HNSW probieren (schneller).
-- Hier bleiben wir bei IVFFlat für maximale Kompatibilität:
CREATE INDEX IF NOT EXISTS idx_chunks_vector_search ON document_chunks 
USING ivfflat (embedding halfvec_cosine_ops) WITH (lists = 100);

-- GIN-Index für die klassische Keyword-Suche (Deutsch)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector ON document_chunks 
USING gin (to_tsvector('german', content));

-- GIN-Index für Titel-Suche (Wichtig für den SEARCH-Modus)
CREATE INDEX IF NOT EXISTS idx_parent_title_tsvector ON parent_documents 
USING gin (to_tsvector('german', title));
