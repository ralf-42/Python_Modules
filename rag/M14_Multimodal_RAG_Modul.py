"""
M14a - Standalone Multimodales RAG mit Bildbeschreibungen
Vollständig eigenständige Version - benötigt KEIN M14.ipynb

Verwendung:
    from M14a_standalone import *

    # System initialisieren
    rag = init_rag_system_enhanced()

    # Verzeichnis verarbeiten
    process_directory(rag, './files', auto_describe_images=True)

    # Suchen
    result = multimodal_search(rag, "Roboter")

Autor: Enhanced by Claude
Datum: Oktober 2025
"""

from pathlib import Path
import uuid
import base64
import shutil
from dataclasses import dataclass

from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from PIL import Image
import chromadb


# ============================================================================
# KONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Zentrale Konfiguration für das RAG-System"""
    chunk_size: int = 200
    chunk_overlap: int = 20
    text_threshold: float = 1.2
    image_threshold: float = 0.8
    clip_model: str = 'clip-ViT-B-32'
    text_model: str = 'text-embedding-3-small'
    llm_model: str = 'gpt-4o-mini'
    vision_model: str = 'gpt-4o-mini'
    db_path: str = './multimodal_rag_db_enhanced'


@dataclass
class RAGComponents:
    """Container für alle RAG-System-Komponenten"""
    text_embeddings: OpenAIEmbeddings
    clip_model: SentenceTransformer
    llm: ChatOpenAI
    vision_llm: ChatOpenAI
    text_splitter: RecursiveCharacterTextSplitter
    markitdown: MarkItDown
    chroma_client: chromadb.PersistentClient
    text_collection: Chroma
    image_collection: any
    config: RAGConfig


# ============================================================================
# SYSTEM-INITIALISIERUNG
# ============================================================================

def init_rag_system_enhanced(config=None):
    """
    Initialisiert das vollständige RAG-System mit Vision-LLM

    Args:
        config: Optional - RAGConfig Instanz

    Returns:
        RAGComponents mit allen Komponenten
    """
    if config is None:
        config = RAGConfig()

    print(f"🚀 Initialisiere Enhanced RAG-System in {config.db_path}")

    # KI-Modelle laden
    text_embeddings = OpenAIEmbeddings(model=config.text_model)
    print("✅ OpenAI Text-Embeddings initialisiert")

    print("🖼️ Lade CLIP-Modell...")
    clip_model = SentenceTransformer(config.clip_model)
    print("✅ CLIP-Modell geladen")

    llm = ChatOpenAI(model=config.llm_model, temperature=0)
    vision_llm = ChatOpenAI(model=config.vision_model, temperature=0)
    print("✅ LLMs initialisiert (Text + Vision)")

    # Text-Verarbeitung
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    markitdown = MarkItDown()

    # Datenbank einrichten
    Path(config.db_path).mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=config.db_path)

    # Text-Collection (für Text-Dokumente UND Bildbeschreibungen)
    text_collection = Chroma(
        collection_name="texts",
        embedding_function=text_embeddings,
        persist_directory=config.db_path
    )

    # Bild-Collection (für CLIP-Embeddings)
    collections = [c.name for c in chroma_client.list_collections()]
    if "images" in collections:
        image_collection = chroma_client.get_collection("images")
    else:
        image_collection = chroma_client.create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )

    print("✅ Collections initialisiert\n")

    return RAGComponents(
        text_embeddings, clip_model, llm, vision_llm, text_splitter,
        markitdown, chroma_client, text_collection, image_collection, config
    )


# ============================================================================
# BILDBESCHREIBUNGS-GENERIERUNG
# ============================================================================

def generate_image_description(vision_llm, image_path):
    """
    Generiert eine detaillierte Beschreibung eines Bildes mit GPT-4o-mini

    Args:
        vision_llm: ChatOpenAI Instanz mit Vision-Unterstützung
        image_path: Pfad zum Bild

    Returns:
        String mit Bildbeschreibung oder Fallback bei Fehler
    """
    try:
        # Bild laden und in Base64 konvertieren
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Dateiendung ermitteln
        image_extension = Path(image_path).suffix.lower().replace('.', '')
        if image_extension == 'jpg':
            image_extension = 'jpeg'

        # Prompt für detaillierte Bildbeschreibung
        prompt = [
            {
                "type": "text",
                "text": """Analysiere dieses Bild detailliert und erstelle eine präzise Beschreibung auf Deutsch.

Beschreibe:
1. Hauptobjekte und -personen
2. Farben und Stimmung
3. Komposition und Setting
4. Besondere Details oder Merkmale
5. Möglichen Kontext oder Zweck

Halte die Beschreibung prägnant aber informativ (2-4 Sätze)."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_extension};base64,{image_data}"
                }
            }
        ]

        # Beschreibung generieren
        response = vision_llm.invoke([{"role": "user", "content": prompt}])
        description = response.content.strip()

        print(f"📝 Bildbeschreibung generiert: {description[:100]}...")
        return description

    except Exception as e:
        print(f"❌ Fehler bei Bildbeschreibung für {Path(image_path).name}: {e}")
        # Fallback: Dateiname als Beschreibung
        return Path(image_path).stem.replace('_', ' ').replace('-', ' ')


# ============================================================================
# DOKUMENT-VERARBEITUNG
# ============================================================================

def add_text_document(components, file_path):
    """
    Fügt ein Text-Dokument zur Datenbank hinzu

    Args:
        components: RAG-System-Komponenten
        file_path: Pfad zum Dokument

    Returns:
        bool - Erfolg
    """
    path = Path(file_path).absolute()

    # Duplikatsprüfung
    if components.text_collection.get(where={"source": str(path)})['ids']:
        print(f"⚠️ {path.name} bereits vorhanden")
        return False

    try:
        # Dokument mit MarkItDown konvertieren
        result = components.markitdown.convert(str(path))
        if not result or not result.text_content.strip():
            print(f"⚠️ {path.name} enthält keinen Text")
            return False

        # Text in Chunks aufteilen
        chunks = components.text_splitter.split_text(result.text_content)
        documents = [
            Document(
                page_content=chunk.strip(),
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "chunk_id": i,
                    "doc_type": "text_document"
                }
            ) for i, chunk in enumerate(chunks) if chunk.strip()
        ]

        # Zur Datenbank hinzufügen
        if documents:
            components.text_collection.add_documents(documents)
            print(f"✅ {len(documents)} Chunks von '{path.name}' hinzugefügt")
            return True

    except Exception as e:
        print(f"❌ Fehler bei {path.name}: {e}")

    return False


def add_image_with_description(components, image_path, auto_describe=True):
    """
    Fügt ein Bild mit automatischer Beschreibung zur Datenbank hinzu

    Args:
        components: RAG-System-Komponenten
        image_path: Pfad zum Bild
        auto_describe: Automatische Beschreibung mit GPT-4o-mini

    Returns:
        Tuple (success: bool, text_doc_id: str oder None)
    """
    path = Path(image_path).absolute()

    if not path.exists():
        print(f"❌ Bild nicht gefunden: {path}")
        return False, None

    # Duplikatsprüfung
    if components.image_collection.get(where={"source": str(path)})['ids']:
        print(f"⚠️ Bild bereits vorhanden: {path.name}")
        return False, None

    try:
        # Bild laden und CLIP-Embedding erstellen
        image = Image.open(path).convert('RGB')
        clip_embedding = components.clip_model.encode(image).tolist()
        print(f"🖼️ Bild-Embedding erstellt für {path.name}")

        # Bildbeschreibung generieren
        description = ""
        if auto_describe:
            description = generate_image_description(components.vision_llm, str(path))
        else:
            description = path.stem.replace('_', ' ').replace('-', ' ')

        # IDs generieren
        image_doc_id = f"img_{uuid.uuid4().hex[:8]}_{path.name}"
        text_doc_id = f"img_desc_{uuid.uuid4().hex[:8]}_{path.stem}"

        # 1. Bildbeschreibung in Text-Collection speichern
        text_document = Document(
            page_content=f"Bildbeschreibung für {path.name}: {description}",
            metadata={
                "source": str(path),
                "filename": path.name,
                "doc_type": "image_description",
                "image_doc_id": image_doc_id,
                "description": description,
                "has_clip_embedding": True
            }
        )
        components.text_collection.add_documents([text_document], ids=[text_doc_id])
        print(f"✅ Bildbeschreibung in Text-Collection gespeichert")

        # 2. Bild in Bild-Collection speichern mit Cross-Reference
        components.image_collection.add(
            ids=[image_doc_id],
            embeddings=[clip_embedding],
            documents=[f"Bild: {path.name} - {description}"[:1000]],
            metadatas=[{
                "source": str(path),
                "filename": path.name,
                "description": description[:500],
                "text_doc_id": text_doc_id
            }]
        )

        print(f"✅ Bild '{path.name}' mit Cross-References hinzugefügt\n")
        return True, text_doc_id

    except Exception as e:
        print(f"❌ Fehler bei Bild {path.name}: {e}")
        return False, None


def process_directory(components, directory, include_images=True, auto_describe_images=True):
    """
    Verarbeitet alle Dateien in einem Verzeichnis

    Args:
        components: RAG-System-Komponenten
        directory: Verzeichnispfad
        include_images: Bilder verarbeiten
        auto_describe_images: Automatische Bildbeschreibungen

    Returns:
        Dictionary mit Statistiken
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Verzeichnis nicht gefunden: {directory}")
        return {"texts": 0, "images": 0, "image_descriptions": 0}

    # Unterstützte Dateitypen
    text_extensions = {'.pdf', '.docx', '.txt', '.md', '.html'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

    # Dateien sammeln
    text_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in text_extensions]
    image_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in image_extensions] if include_images else []

    print(f"📊 Gefunden: {len(text_files)} Text-Dateien, {len(image_files)} Bilder\n")

    # Text-Dateien verarbeiten
    text_count = 0
    for file_path in text_files:
        print(f"📄 {file_path.name}")
        if add_text_document(components, str(file_path)):
            text_count += 1

    print()  # Leerzeile

    # Bild-Dateien verarbeiten
    image_count = 0
    image_desc_count = 0
    for img_path in image_files:
        print(f"🖼️ {img_path.name}")
        success, text_doc_id = add_image_with_description(
            components,
            str(img_path),
            auto_describe=auto_describe_images
        )
        if success:
            image_count += 1
            if text_doc_id:
                image_desc_count += 1

    return {
        "texts": text_count,
        "images": image_count,
        "image_descriptions": image_desc_count
    }


# ============================================================================
# SUCHFUNKTIONEN
# ============================================================================

def search_texts(components, query, k=3, include_image_descriptions=True):
    """
    Durchsucht Text-Dokumente inkl. Bildbeschreibungen

    Args:
        components: RAG-System-Komponenten
        query: Suchanfrage
        k: Anzahl Ergebnisse
        include_image_descriptions: Bildbeschreibungen einschließen

    Returns:
        Formatierter String mit Ergebnissen
    """
    if not components.text_collection.get()['ids']:
        return "❌ Keine Text-Dokumente gefunden"

    # Ähnlichkeitssuche durchführen
    docs_with_scores = components.text_collection.similarity_search_with_score(query, k=k*2)
    if not docs_with_scores:
        return "❌ Keine relevanten Dokumente gefunden"

    # Score in Ähnlichkeit umwandeln
    docs_with_similarity = []
    for doc, score in docs_with_scores:
        similarity = max(0, min(1, 2.0 / (1 + score)))
        docs_with_similarity.append((doc, similarity))

    # Nach Ähnlichkeit sortieren
    docs_with_similarity.sort(key=lambda x: x[1], reverse=True)

    # Filtern nach Mindest-Ähnlichkeit
    min_similarity = 0.3
    relevant_docs = [(doc, sim) for doc, sim in docs_with_similarity[:k]
                     if sim >= min_similarity]

    if not relevant_docs:
        return "❌ Keine ausreichend ähnlichen Dokumente gefunden"

    # Dokumente nach Typ trennen
    text_docs = []
    image_desc_docs = []

    for doc, sim in relevant_docs:
        doc_type = doc.metadata.get("doc_type", "text_document")
        if doc_type == "image_description" and include_image_descriptions:
            image_desc_docs.append((doc, sim))
        elif doc_type == "text_document":
            text_docs.append((doc, sim))

    # Kontext für LLM zusammenstellen
    all_docs = text_docs + image_desc_docs
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in all_docs])

    # Quellen sammeln
    sources = [
        {
            "filename": doc.metadata.get("filename", "Unbekannt"),
            "similarity": round(sim, 3),
            "type": doc.metadata.get("doc_type", "text_document"),
            "image_doc_id": doc.metadata.get("image_doc_id")
        }
        for doc, sim in all_docs
    ]

    # LLM-Antwort generieren
    prompt = f"""Beantworte die Frage präzise basierend auf dem Kontext.

KONTEXT:
{context}

FRAGE: {query}

ANTWORT:"""

    response = components.llm.invoke(prompt).content

    # Ausgabe mit separaten Quellenlisten
    text_sources = [s for s in sources if s['type'] == 'text_document']
    image_sources = [s for s in sources if s['type'] == 'image_description']

    result = response

    if text_sources:
        result += f"\n\n📚 Text-Quellen ({len(text_sources)}): " + "\n".join([
            f"   • {src['filename']} (Ähnlichkeit: {src['similarity']})"
            for src in text_sources
        ])

    if image_sources:
        result += f"\n\n🖼️ Relevante Bilder ({len(image_sources)}): " + "\n".join([
            f"   • {src['filename']} (Ähnlichkeit: {src['similarity']})"
            for src in image_sources
        ])

    return result


def search_images(components, query, k=3):
    """
    Durchsucht Bilder mit Text-Query über CLIP

    Args:
        components: RAG-System-Komponenten
        query: Suchanfrage
        k: Anzahl Ergebnisse

    Returns:
        Liste von Bildern mit Metadaten
    """
    if components.image_collection.count() == 0:
        return []

    # Text-Query in Bild-Embedding-Raum umwandeln
    query_embedding = components.clip_model.encode(query).tolist()

    # Suche in Bild-Collection
    results = components.image_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k*2, components.image_collection.count()),
        include=['documents', 'metadatas', 'distances']
    )

    if not results['ids'][0]:
        return []

    # Ergebnisse filtern und formatieren
    return [
        {
            "filename": metadata.get("filename", "Unbekannt"),
            "path": metadata.get("source", ""),
            "description": metadata.get("description", ""),
            "text_doc_id": metadata.get("text_doc_id", ""),
            "similarity": round(max(0, 1 - distance), 3)
        }
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0])
        if distance < components.config.image_threshold
    ]


def find_related_images_from_text(components, text_doc_ids, k=3):
    """
    Findet Bilder über ihre Textbeschreibungen (Cross-Modal-Retrieval)

    Args:
        components: RAG-System-Komponenten
        text_doc_ids: Liste von Text-Dokument-IDs
        k: Maximale Anzahl Bilder

    Returns:
        Liste von verwandten Bildern
    """
    related_images = []

    for text_id in text_doc_ids:
        try:
            doc_data = components.text_collection.get(ids=[text_id])
            if not doc_data['ids']:
                continue

            metadata = doc_data['metadatas'][0]

            # Prüfe ob es eine Bildbeschreibung ist
            if metadata.get('doc_type') == 'image_description':
                image_doc_id = metadata.get('image_doc_id')

                if image_doc_id:
                    # Hole das zugehörige Bild
                    image_data = components.image_collection.get(ids=[image_doc_id])

                    if image_data['ids']:
                        img_metadata = image_data['metadatas'][0]
                        related_images.append({
                            'filename': img_metadata.get('filename', 'Unbekannt'),
                            'path': img_metadata.get('source', ''),
                            'description': img_metadata.get('description', ''),
                            'source': 'cross_modal_retrieval'
                        })
        except Exception as e:
            print(f"⚠️ Fehler beim Cross-Modal-Retrieval: {e}")
            continue

    return related_images[:k]


def multimodal_search(components, query, k_text=3, k_images=3, enable_cross_modal=True):
    """
    Führt erweiterte multimodale Suche durch

    Kombiniert:
    - Text-Suche (inkl. Bildbeschreibungen)
    - CLIP-basierte Bildsuche
    - Cross-Modal-Retrieval (Text → Bild über Beschreibungen)

    Args:
        components: RAG-System-Komponenten
        query: Suchanfrage
        k_text: Anzahl Text-Ergebnisse
        k_images: Anzahl Bild-Ergebnisse
        enable_cross_modal: Cross-Modal-Retrieval aktivieren

    Returns:
        Formatierter String mit allen Ergebnissen
    """
    print(f"\n{'='*70}")
    print(f"🔍 Multimodale Suche: {query}")
    print(f"{'='*70}\n")

    # 1. Text-Suche (inkl. Bildbeschreibungen)
    text_results = search_texts(components, query, k_text, include_image_descriptions=True)

    # 2. Direkte Bild-Suche über CLIP
    image_results = search_images(components, query, k_images)

    # 3. Cross-Modal-Retrieval
    cross_modal_images = []
    if enable_cross_modal:
        docs_with_scores = components.text_collection.similarity_search_with_score(query, k=k_text*2)
        image_desc_ids = [
            components.text_collection.get(where={"source": doc.metadata['source']})['ids'][0]
            for doc, _ in docs_with_scores
            if doc.metadata.get('doc_type') == 'image_description'
        ]

        if image_desc_ids:
            cross_modal_images = find_related_images_from_text(components, image_desc_ids, k_images)

    # Ergebnisse zusammenfassen
    result = f"📄 TEXT-ERGEBNISSE:\n{'-'*70}\n{text_results}\n\n"

    if image_results:
        result += f"🖼️ BILD-ERGEBNISSE via CLIP ({len(image_results)} gefunden):\n{'-'*70}\n"
        for i, img in enumerate(image_results, 1):
            result += f"   {i}. {img['filename']} (Ähnlichkeit: {img['similarity']})\n"
            if img['description']:
                result += f"      📝 {img['description'][:600]}...\n"
    else:
        result += f"🖼️ Keine relevanten Bilder via CLIP gefunden.\n"

    if cross_modal_images:
        result += f"\n🔗 CROSS-MODAL RETRIEVAL ({len(cross_modal_images)} Bilder via Textsuche):\n{'-'*70}\n"
        for i, img in enumerate(cross_modal_images, 1):
            result += f"   {i}. {img['filename']}\n"
            if img['description']:
                result += f"      📝 {img['description'][:600]}...\n"

    return result


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def get_system_status(components):
    """Gibt System-Status zurück"""
    text_data = components.text_collection.get()
    all_text_count = len(text_data['ids'])

    text_docs_count = sum(1 for meta in text_data['metadatas']
                         if meta.get('doc_type') == 'text_document')
    image_desc_count = sum(1 for meta in text_data['metadatas']
                          if meta.get('doc_type') == 'image_description')

    image_count = components.image_collection.count()

    return {
        "text_chunks": text_docs_count,
        "image_descriptions": image_desc_count,
        "images": image_count,
        "total_text_entries": all_text_count,
        "total_documents": all_text_count + image_count
    }


def cleanup_database(db_path='./multimodal_rag_db_enhanced'):
    """Löscht die Datenbank komplett"""
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        print(f"🗑️ Datenbank gelöscht: {db_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("M14a - Standalone Multimodales RAG mit Bildbeschreibungen")
    print("="*70)
    print("\n✅ Modul geladen - keine Abhängigkeiten zu M14.ipynb\n")
    print("Verfügbare Funktionen:")
    print("  • init_rag_system_enhanced()        - System initialisieren")
    print("  • generate_image_description()       - Einzelne Bildbeschreibung")
    print("  • add_text_document()                - Text-Dokument hinzufügen")
    print("  • add_image_with_description()       - Bild mit Beschreibung")
    print("  • process_directory()                - Verzeichnis verarbeiten")
    print("  • search_texts()                     - Text-Suche")
    print("  • search_images()                    - Bild-Suche (CLIP)")
    print("  • find_related_images_from_text()    - Cross-Modal-Retrieval")
    print("  • multimodal_search()                - Kombinierte Suche")
    print("  • get_system_status()                - System-Status")
    print("  • cleanup_database()                 - Datenbank löschen")
    print("\n" + "="*70)
