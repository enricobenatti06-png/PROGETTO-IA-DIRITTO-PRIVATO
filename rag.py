import os
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import streamlit as st

from huggingface_hub import list_repo_files, hf_hub_download
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIGURAZIONE
# ============================================================

HF_REPO = "enricobenatti06/manuale_galgano"

HF_REPO_TYPE = "dataset"

HF_FOLDER = "Manuale Galgano"

# Modello per gli embedding.
# Multilingua: adatto anche ai testi italiani.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modello LLM.
# Può essere modificato dai Secrets di Streamlit.
DEFAULT_LLM_MODEL = "gpt-6-luna"

# Numero massimo di documenti recuperati.
DEFAULT_TOP_K = 5

# Dimensione massima del chunk.
CHUNK_SIZE = 1800

# Sovrapposizione tra chunk.
CHUNK_OVERLAP = 250


# ============================================================
# OPENAI
# ============================================================

@st.cache_resource
def get_openai_client():

    api_key = st.secrets.get(
        "OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non configurata."
        )

    return OpenAI(
        api_key=api_key
    )


def get_llm_model():

    return st.secrets.get(
        "OPENAI_MODEL",
        DEFAULT_LLM_MODEL
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# DOWNLOAD / LETTURA DATASET
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_repository_files():

    try:

        files = list_repo_files(
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE
        )

    except Exception as e:

        raise RuntimeError(
            f"Impossibile collegarsi al repository "
            f"Hugging Face '{HF_REPO}': {e}"
        )


    # Manteniamo soltanto i file contenuti
    # nella cartella del Manuale Galgano.

    prefix = HF_FOLDER.rstrip("/") + "/"

    relevant_files = []

    for file in files:

        if not file.startswith(prefix):
            continue

        extension = Path(file).suffix.lower()

        if extension in {
            ".txt",
            ".md",
            ".markdown"
        }:

            relevant_files.append(file)


    return sorted(
        relevant_files
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def download_text_file(
    file_path
):

    try:

        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=file_path,
            repo_type=HF_REPO_TYPE
        )

    except Exception as e:

        raise RuntimeError(
            f"Errore durante il download di "
            f"'{file_path}': {e}"
        )


    path = Path(local_path)


    try:

        return path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        return path.read_text(
            encoding="utf-8",
            errors="replace"
        )


# ============================================================
# PULIZIA TESTO
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Normalizza gli spazi.

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Elimina spazi multipli.

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Elimina troppe righe vuote.

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Divide il testo in blocchi mantenendo una
    sovrapposizione tra i blocchi.

    Il chunking cerca prima di tutto di interrompere
    il testo in corrispondenza di paragrafi.
    """

    text = clean_text(text)

    if not text:
        return []


    paragraphs = re.split(
        r"\n\s*\n",
        text
    )


    chunks = []

    current = ""


    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue


        # Se il paragrafo può essere aggiunto
        # senza superare la dimensione prevista.

        if len(current) + len(paragraph) + 2 <= chunk_size:

            if current:

                current += "\n\n" + paragraph

            else:

                current = paragraph

            continue


        # Salviamo il chunk precedente.

        if current:

            chunks.append(
                current.strip()
            )


        # Se il singolo paragrafo è troppo grande,
        # lo dividiamo ulteriormente.

        if len(paragraph) > chunk_size:

            start = 0

            while start < len(paragraph):

                end = start + chunk_size

                piece = paragraph[start:end]

                if piece.strip():

                    chunks.append(
                        piece.strip()
                    )

                start = end - overlap

            current = ""

        else:

            current = paragraph


    if current:

        chunks.append(
            current.strip()
        )


    return chunks


# ============================================================
# COSTRUZIONE CORPUS
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def build_corpus():

    files = get_repository_files()

    documents = []


    for file_path in files:

        try:

            text = download_text_file(
                file_path
            )

        except Exception:

            continue


        text = clean_text(
            text
        )


        # Ignoriamo i file vuoti.

        if not text:
            continue


        chunks = split_text(
            text
        )


        for index, chunk in enumerate(
            chunks
        ):

            documents.append({

                "text": chunk,

                "source": file_path,

                "chunk_id": index,

                "title": Path(
                    file_path
                ).stem,

                "path": file_path

            })


    return documents


# ============================================================
# EMBEDDING CORPUS
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def build_embeddings(
    texts
):

    model = get_embedding_model()


    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# CARICAMENTO INDICE
# ============================================================

@st.cache_resource
def get_rag_index():

    documents = build_corpus()


    if not documents:

        raise RuntimeError(
            "Il repository Hugging Face non contiene "
            "testi utilizzabili."
        )


    texts = [
        document["text"]
        for document in documents
    ]


    embeddings = build_embeddings(
        texts
    )


    return documents, embeddings


# ============================================================
# EMBEDDING DELLA QUERY
# ============================================================

def embed_query(
    query
):

    model = get_embedding_model()


    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )


    return np.asarray(
        embedding[0],
        dtype=np.float32
    )


# ============================================================
# RICERCA SEMANTICA
# ============================================================

def search_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict]:

    documents, embeddings = get_rag_index()


    if not documents:

        return []


    query_embedding = embed_query(
        query
    )


    # Poiché gli embedding sono normalizzati,
    # il prodotto scalare equivale alla cosine similarity.

    scores = embeddings @ query_embedding


    # Ordina dal documento più pertinente
    # al meno pertinente.

    ranked_indices = np.argsort(
        scores
    )[::-1]


    results = []


    for index in ranked_indices[:top_k]:

        document = dict(
            documents[index]
        )


        document["score"] = float(
            scores[index]
        )


        results.append(
            document
        )


    return results


# ============================================================
# FILTRO DI RILEVANZA
# ============================================================

def filter_relevant_documents(
    documents,
    minimum_score=0.25
):

    relevant = []

    for document in documents:

        score = document.get(
            "score",
            0
        )


        if score >= minimum_score:

            relevant.append(
                document
            )


    # Se la soglia è troppo severa,
    # manteniamo almeno il primo risultato.

    if not relevant and documents:

        relevant = [
            documents[0]
        ]


    return relevant


# ============================================================
# COSTRUZIONE CONTESTO
# ============================================================

def build_context(
    documents
):

    if not documents:

        return (
            "Nessun documento pertinente "
            "è stato recuperato dal Manuale Galgano."
        )


    context_parts = []


    for number, document in enumerate(
        documents,
        start=1
    ):

        source = document.get(
            "source",
            "Fonte non specificata"
        )

        text = document.get(
            "text",
            ""
        )

        score = document.get(
            "score"
        )


        if score is not None:

            score_text = (
                f"Pertinenza: {score:.3f}"
            )

        else:

            score_text = ""


        context_parts.append(
            f"""
[FONTE {number}]
Percorso: {source}
{score_text}

{text}
""".strip()
        )


    return "\n\n====================\n\n".join(
        context_parts
    )


# ============================================================
# PROMPT GIURIDICO
# ============================================================

def build_prompt(
    question,
    context
):

    return f"""
Sei un assistente di diritto privato destinato allo studio
universitario.

Il tuo compito è rispondere alla domanda dell'utente
utilizzando il materiale giuridico recuperato dal
Manuale Galgano.

PRINCIPI DA RISPETTARE:

1. Non inventare norme, articoli, sentenze o citazioni.

2. Non attribuire al Manuale Galgano affermazioni che
   non risultano dal contesto fornito.

3. Quando richiesto o pertinente, indica gli articoli
   del Codice civile.

4. Distingui la disposizione normativa dalla sua
   interpretazione dottrinale.

5. Se il materiale recuperato non è sufficiente,
   dichiaralo espressamente.

6. Non trattare il materiale recuperato come se fosse
   automaticamente una fonte normativa.

7. Rispondi in italiano.

8. Usa terminologia giuridica precisa.

9. Per domande universitarie privilegia una spiegazione
   ragionata e strutturata, non una semplice definizione.

10. Non aggiungere informazioni non necessarie soltanto
    per rendere la risposta più lunga.

MATERIALE RECUPERATO DAL DATABASE:

{context}

DOMANDA DELL'UTENTE:

{question}

Fornisci una risposta giuridicamente precisa,
chiara e strutturata.
""".strip()


# ============================================================
# GENERAZIONE LLM
# ============================================================

def generate_with_llm(
    prompt
):

    client = get_openai_client()

    model = get_llm_model()


    try:

        response = client.responses.create(

            model=model,

            instructions=(
                "Sei un assistente universitario "
                "specializzato in diritto privato italiano."
            ),

            input=prompt

        )


        answer = response.output_text


        if not answer:

            raise RuntimeError(
                "Il modello non ha restituito testo."
            )


        return answer.strip()


    except Exception as e:

        raise RuntimeError(
            f"Errore durante la generazione della risposta: {e}"
        )


# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================

def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K
) -> Dict:

    question = question.strip()


    if not question:

        return {
            "answer": "Inserisci una domanda.",
            "documents": [],
            "context": ""
        }


    # --------------------------------------------------------
    # 1. RICERCA
    # --------------------------------------------------------

    documents = search_documents(
        question,
        top_k=top_k
    )


    # --------------------------------------------------------
    # 2. FILTRO
    # --------------------------------------------------------

    documents = filter_relevant_documents(
        documents
    )


    # --------------------------------------------------------
    # 3. CONTESTO
    # --------------------------------------------------------

    context = build_context(
        documents
    )


    # --------------------------------------------------------
    # 4. PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        context
    )


    # --------------------------------------------------------
    # 5. LLM
    # --------------------------------------------------------

    answer = generate_with_llm(
        prompt
    )


    # --------------------------------------------------------
    # 6. RISULTATO
    # --------------------------------------------------------

    return {

        "answer": answer,

        "documents": documents,

        "context": context

    }


# ============================================================
# INFORMAZIONI SUL DATABASE
# ============================================================

def get_database_info():

    documents, embeddings = get_rag_index()


    sources = set(
        document["source"]
        for document in documents
    )


    return {

        "repository": HF_REPO,

        "folder": HF_FOLDER,

        "documents": len(documents),

        "sources": len(sources),

        "embedding_model": EMBEDDING_MODEL,

        "llm_model": get_llm_model()

    }
