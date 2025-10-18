"""
M14 - Multimodales RAG Modul mit LlamaIndex (Version 4 - Vereinfacht)

Verwendung:
    # System initialisieren
    rag = init_rag_system()

    # Verzeichnis verarbeiten
    stats = process_directory(rag, './files', auto_describe_images=True)

    # Multimodale Suche
    result = multimodal_search(rag, "Roboter")

    # Bild → Bild Suche
    similar_images = search_similar_images(rag, "./query_image.jpg", k=5)

    # Bild → Text Suche
    text_results = search_text_by_image(rag, "./query_image.jpg", k=3)

Autor: Enhanced by Claude with LlamaIndex
Datum: Oktober 2025
Version: 4.0 - Vereinfachung durch LlamaIndex Framework
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import shutil

# LlamaIndex Core
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument, TextNode

# Vector Stores
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Embeddings & LLMs
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# Additional utilities
from PIL import Image


# ============================================================================
# KONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Zentrale Konfiguration für das RAG-System"""
    # Modell-Konfiguration
    text_embed_model: str = 'text-embedding-3-small'
    clip_model: str = 'openai/clip-vit-base-patch32'
    llm_model: str = 'gpt-4o-mini'
    vision_model: str = 'gpt-4o-mini'

    # Retrieval-Konfiguration
    similarity_top_k: int = 3
    image_similarity_top_k: int = 3

    # Datenbank-Konfiguration
    db_path: str = './multimodal_rag_db_v4'
    text_collection: str = 'text_store'
    image_collection: str = 'image_store'

    # LLM-Konfiguration
    temperature: float = 0.0


@dataclass
class RAGSystem:
    """Container für das LlamaIndex RAG-System"""
    index: MultiModalVectorStoreIndex
    retriever: any
    query_engine: any
    chroma_client: chromadb.PersistentClient
    config: RAGConfig


# ============================================================================
# SYSTEM-INITIALISIERUNG
# ============================================================================

def init_rag_system(config: Optional[RAGConfig] = None) -> RAGSystem:
    """
    Initialisiert das LlamaIndex-basierte multimodale RAG-System

    Args:
        config: Optional - RAGConfig Instanz

    Returns:
        RAGSystem mit Index, Retriever und Query Engine
    """
    if config is None:
        config = RAGConfig()

    print(f"🚀 Initialisiere LlamaIndex RAG-System in {config.db_path}")

    # Global Settings konfigurieren
    Settings.llm = OpenAI(model=config.llm_model, temperature=config.temperature)
    Settings.embed_model = OpenAIEmbedding(model=config.text_embed_model)

    print("✅ LLM und Text-Embeddings konfiguriert")

    # CLIP für Bild-Embeddings
    clip_embed_model = ClipEmbedding(model_name=config.clip_model)
    print("✅ CLIP-Modell geladen")

    # ChromaDB Client initialisieren
    Path(config.db_path).mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=config.db_path)

    # Vector Stores für Text und Bilder
    text_collection = chroma_client.get_or_create_collection(config.text_collection)
    image_collection = chroma_client.get_or_create_collection(config.image_collection)

    text_store = ChromaVectorStore(chroma_collection=text_collection)
    image_store = ChromaVectorStore(chroma_collection=image_collection)

    # Storage Context mit separaten Stores
    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store
    )

    print("✅ ChromaDB Collections erstellt")

    # MultiModal Index erstellen (leer initialisiert)
    index = MultiModalVectorStoreIndex.from_documents(
        [],
        storage_context=storage_context,
        image_embed_model=clip_embed_model
    )

    # Retriever und Query Engine erstellen
    retriever = index.as_retriever(
        similarity_top_k=config.similarity_top_k,
        image_similarity_top_k=config.image_similarity_top_k
    )

    query_engine = index.as_query_engine(
        llm=OpenAI(model=config.llm_model, temperature=config.temperature),
        image_qa_prompt="Beantworte die Frage basierend auf den bereitgestellten Dokumenten und Bildern auf Deutsch."
    )

    print("✅ Index, Retriever und Query Engine initialisiert\n")

    return RAGSystem(
        index=index,
        retriever=retriever,
        query_engine=query_engine,
        chroma_client=chroma_client,
        config=config
    )


# ============================================================================
# DOKUMENT-VERARBEITUNG
# ============================================================================

def process_directory(
    rag_system: RAGSystem,
    directory: str,
    include_images: bool = True,
    auto_describe_images: bool = True,
    text_extensions: Optional[List[str]] = None,
    image_extensions: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Verarbeitet alle Dateien in einem Verzeichnis mit LlamaIndex

    Args:
        rag_system: RAG-System-Instanz
        directory: Verzeichnispfad
        include_images: Bilder verarbeiten
        auto_describe_images: Automatische Bildbeschreibungen mit Vision-LLM
        text_extensions: Unterstützte Text-Dateiformate
        image_extensions: Unterstützte Bildformate

    Returns:
        Dictionary mit Statistiken
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Verzeichnis nicht gefunden: {directory}")
        return {"texts": 0, "images": 0, "total": 0}

    # Standard-Dateitypen
    if text_extensions is None:
        text_extensions = ['.pdf', '.docx', '.txt', '.md', '.html']
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    print(f"📂 Verarbeite Verzeichnis: {directory}\n")

    # Text-Dokumente mit SimpleDirectoryReader laden
    text_docs = []
    if text_extensions:
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(dir_path),
                required_exts=text_extensions,
                recursive=True
            )
            text_docs = reader.load_data()
            print(f"✅ {len(text_docs)} Text-Dokumente geladen")
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von Text-Dokumenten: {e}")

    # Bilder separat verarbeiten mit Vision-LLM Beschreibungen
    image_docs = []
    if include_images:
        image_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in image_extensions]
        print(f"🖼️ {len(image_files)} Bilder gefunden\n")

        if auto_describe_images:
            vision_llm = OpenAIMultiModal(
                model=rag_system.config.vision_model,
                temperature=0.0
            )

            for img_path in image_files:
                try:
                    print(f"   📸 {img_path.name}")

                    # Vision-LLM Beschreibung generieren
                    image_doc = ImageDocument(image_path=str(img_path))

                    # Beschreibung mit Vision-LLM
                    description_prompt = """Analysiere dieses Bild detailliert auf Deutsch.

Beschreibe:
1. Hauptobjekte und -personen
2. Farben und Stimmung
3. Komposition und Setting
4. Besondere Details oder Merkmale

Halte die Beschreibung prägnant aber informativ (2-4 Sätze)."""

                    response = vision_llm.complete(
                        prompt=description_prompt,
                        image_documents=[image_doc]
                    )

                    description = response.text.strip()

                    # Metadaten erweitern
                    image_doc.metadata = {
                        'file_path': str(img_path),
                        'file_name': img_path.name,
                        'description': description,
                        'doc_type': 'image'
                    }
                    image_doc.text = f"Bild: {img_path.name} - {description}"

                    image_docs.append(image_doc)
                    print(f"      ✅ Beschreibung: {description[:80]}...")

                except Exception as e:
                    print(f"      ❌ Fehler bei {img_path.name}: {e}")
        else:
            # Ohne Beschreibung - nur Dateiname
            for img_path in image_files:
                image_doc = ImageDocument(
                    image_path=str(img_path),
                    metadata={
                        'file_path': str(img_path),
                        'file_name': img_path.name,
                        'doc_type': 'image'
                    }
                )
                image_docs.append(image_doc)

    # Alle Dokumente zum Index hinzufügen
    all_docs = text_docs + image_docs

    if all_docs:
        print(f"\n📥 Füge {len(all_docs)} Dokumente zum Index hinzu...")

        # LlamaIndex verarbeitet automatisch Text und Bilder
        for doc in all_docs:
            rag_system.index.insert(doc)

        print("✅ Alle Dokumente erfolgreich indexiert\n")

    return {
        "texts": len(text_docs),
        "images": len(image_docs),
        "total": len(all_docs)
    }


# ============================================================================
# SUCHFUNKTIONEN
# ============================================================================

def multimodal_search(
    rag_system: RAGSystem,
    query: str,
    k_text: Optional[int] = None,
    k_images: Optional[int] = None
) -> str:
    """
    Führt multimodale Suche mit LlamaIndex Query Engine durch

    Args:
        rag_system: RAG-System-Instanz
        query: Suchanfrage
        k_text: Optional - Anzahl Text-Ergebnisse (überschreibt Config)
        k_images: Optional - Anzahl Bild-Ergebnisse (überschreibt Config)

    Returns:
        Formatierte Antwort mit Quellen
    """
    print(f"\n{'='*70}")
    print(f"🔍 Multimodale Suche: {query}")
    print(f"{'='*70}\n")

    # Retriever mit benutzerdefinierten k-Werten erstellen falls angegeben
    if k_text or k_images:
        retriever = rag_system.index.as_retriever(
            similarity_top_k=k_text or rag_system.config.similarity_top_k,
            image_similarity_top_k=k_images or rag_system.config.image_similarity_top_k
        )
    else:
        retriever = rag_system.retriever

    # Retrieval durchführen
    retrieved_nodes = retriever.retrieve(query)

    # Ergebnisse nach Typ trennen
    text_sources = []
    image_sources = []

    for node in retrieved_nodes:
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        score = node.score if hasattr(node, 'score') else 0.0

        doc_type = metadata.get('doc_type', 'text')

        if doc_type == 'image':
            image_sources.append({
                'filename': metadata.get('file_name', 'Unbekannt'),
                'path': metadata.get('file_path', ''),
                'description': metadata.get('description', ''),
                'similarity': round(score, 3)
            })
        else:
            text_sources.append({
                'filename': metadata.get('file_name', 'Unbekannt'),
                'similarity': round(score, 3)
            })

    # Query Engine für Antwortgenerierung verwenden
    response = rag_system.query_engine.query(query)

    # Formatierte Ausgabe
    result = f"📄 ANTWORT:\n{'-'*70}\n{response.response}\n\n"

    if text_sources:
        result += f"📚 Text-Quellen ({len(text_sources)}):\n"
        for i, src in enumerate(text_sources, 1):
            result += f"   {i}. {src['filename']} (Ähnlichkeit: {src['similarity']})\n"
        result += "\n"

    if image_sources:
        result += f"🖼️ Relevante Bilder ({len(image_sources)}):\n"
        for i, img in enumerate(image_sources, 1):
            result += f"   {i}. {img['filename']} (Ähnlichkeit: {img['similarity']})\n"
            if img['description']:
                result += f"      📝 {img['description'][:300]}...\n"

    return result


def search_similar_images(
    rag_system: RAGSystem,
    query_image_path: str,
    k: int = 5
) -> List[Dict]:
    """
    Bild → Bild Suche: Findet visuell ähnliche Bilder

    Args:
        rag_system: RAG-System-Instanz
        query_image_path: Pfad zum Query-Bild
        k: Anzahl ähnlicher Bilder

    Returns:
        Liste von ähnlichen Bildern mit Metadaten
    """
    path = Path(query_image_path)

    if not path.exists():
        print(f"❌ Query-Bild nicht gefunden: {query_image_path}")
        return []

    print(f"\n{'='*70}")
    print(f"🔍 Bild → Bild Suche für: {path.name}")
    print(f"{'='*70}\n")

    try:
        # Query-Bild als ImageDocument
        query_image_doc = ImageDocument(image_path=str(path))

        # Image Retriever mit custom k
        image_retriever = rag_system.index.as_retriever(
            similarity_top_k=0,  # Keine Text-Ergebnisse
            image_similarity_top_k=k
        )

        # Suche durchführen (mit Bild-Query)
        # LlamaIndex unterstützt leider nicht direkt Bild-zu-Bild-Queries
        # Workaround: Verwende CLIP-Embedding direkt
        from llama_index.embeddings.clip import ClipEmbedding

        clip_embed_model = ClipEmbedding(model_name=rag_system.config.clip_model)

        # Query-Image Embedding erstellen
        image = Image.open(path).convert('RGB')
        query_embedding = clip_embed_model.get_image_embedding(str(path))

        # Manuelle Suche im image_store
        image_collection = rag_system.chroma_client.get_collection(rag_system.config.image_collection)

        results = image_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'distances']
        )

        # Ergebnisse formatieren
        similar_images = []
        if results['ids'] and results['ids'][0]:
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                similarity = max(0, 1 - (distance / 2))  # L2 zu Ähnlichkeit

                similar_images.append({
                    'filename': metadata.get('file_name', 'Unbekannt'),
                    'path': metadata.get('file_path', ''),
                    'description': metadata.get('description', ''),
                    'similarity': round(similarity, 3)
                })

        print(f"✅ {len(similar_images)} ähnliche Bilder gefunden\n")

        for i, img in enumerate(similar_images, 1):
            print(f"{i}. {img['filename']} (Ähnlichkeit: {img['similarity']})")
            if img['description']:
                print(f"   📝 {img['description'][:200]}...\n")

        return similar_images

    except Exception as e:
        print(f"❌ Fehler bei Bild-zu-Bild-Suche: {e}")
        return []


def search_text_by_image(
    rag_system: RAGSystem,
    query_image_path: str,
    k: int = 3
) -> str:
    """
    Bild → Text Suche: Findet ähnliche Bilder und gibt Textinformationen zurück

    Args:
        rag_system: RAG-System-Instanz
        query_image_path: Pfad zum Query-Bild
        k: Anzahl Ergebnisse

    Returns:
        Formatierter String mit Textinformationen zu ähnlichen Bildern
    """
    path = Path(query_image_path)

    if not path.exists():
        return f"❌ Query-Bild nicht gefunden: {query_image_path}"

    print(f"\n{'='*70}")
    print(f"🔍 Bild → Text Suche für: {path.name}")
    print(f"{'='*70}\n")

    # 1. Ähnliche Bilder finden
    similar_images = search_similar_images(rag_system, query_image_path, k)

    if not similar_images:
        return "❌ Keine ähnlichen Bilder gefunden"

    # 2. Beschreibungen sammeln
    descriptions = []
    for img in similar_images:
        if img['description']:
            descriptions.append(
                f"Bild: {img['filename']} (Ähnlichkeit: {img['similarity']})\n"
                f"{img['description']}"
            )

    if not descriptions:
        return "❌ Keine Textbeschreibungen verfügbar"

    # 3. LLM-Zusammenfassung erstellen
    context = "\n\n---\n\n".join(descriptions)

    prompt = f"""Du hast ein Query-Bild erhalten und folgende visuell ähnliche Bilder wurden gefunden.
Fasse die Gemeinsamkeiten und wichtigsten Merkmale zusammen.

GEFUNDENE BILDER UND IHRE BESCHREIBUNGEN:
{context}

ZUSAMMENFASSUNG:"""

    try:
        llm = OpenAI(model=rag_system.config.llm_model, temperature=0.0)
        response = llm.complete(prompt)

        # Formatierte Ausgabe
        result = f"📄 ZUSAMMENFASSUNG:\n{'-'*70}\n{response.text}\n\n"
        result += f"🖼️ GEFUNDENE ÄHNLICHE BILDER ({len(similar_images)}):\n{'-'*70}\n"

        for i, img in enumerate(similar_images, 1):
            result += f"   {i}. {img['filename']} (Ähnlichkeit: {img['similarity']})\n"
            if img['description']:
                result += f"      📝 {img['description'][:300]}...\n"

        return result

    except Exception as e:
        print(f"❌ Fehler bei Textgenerierung: {e}")
        return f"❌ Fehler bei der Verarbeitung: {e}"


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def get_system_status(rag_system: RAGSystem) -> Dict:
    """
    Gibt System-Status zurück

    Args:
        rag_system: RAG-System-Instanz

    Returns:
        Dictionary mit Statistiken
    """
    text_collection = rag_system.chroma_client.get_collection(rag_system.config.text_collection)
    image_collection = rag_system.chroma_client.get_collection(rag_system.config.image_collection)

    text_count = text_collection.count()
    image_count = image_collection.count()

    return {
        "text_chunks": text_count,
        "images": image_count,
        "total_documents": text_count + image_count,
        "db_path": rag_system.config.db_path
    }


def cleanup_database(db_path: str = './multimodal_rag_db_v4'):
    """
    Löscht die Datenbank komplett

    Args:
        db_path: Pfad zur Datenbank
    """
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        print(f"🗑️ Datenbank gelöscht: {db_path}")
    else:
        print(f"⚠️ Datenbank existiert nicht: {db_path}")

