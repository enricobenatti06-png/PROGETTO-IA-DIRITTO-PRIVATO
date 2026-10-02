import re

import numpy as np
import streamlit as st

from groq import Groq
from huggingface_hub import list_repo_files, hf_hub_download
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAZIONE
# ============================================================

HF_REPO_ID = "enricobenatti06/manuale_galgano"
HF_REPO_TYPE = "dataset"

# Modello utilizzato per trasformare i testi in vettori
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Modello LLM servito da Groq
GROQ_MODEL = "openai/gpt-oss-120b"

# Cartella principale del dataset
HF_FOLDER = "Manuale Galgano"

# Dimensione dei blocchi di testo
CHUNK_SIZE = 1800

# Sovrapposizione tra blocchi
CHUNK_OVERLAP = 250

# Numero massimo di documenti recuperati
DEFAULT_TOP_K = 5

# Soglia minima di similarità
MINIMUM_SCORE = 0.25


# ============================================================
# CLIENT GROQ
# ============================================================

@st.cache_resource
def get_groq_client():

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# MODELLO EMBEDDING
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# PULIZIA TESTO
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Elimina spazi multipli
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Elimina troppe righe vuote
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# LETTURA FILE HUGGING FACE
# ============================================================

@st.cache_data(show_spinner=False)
def get_hf_files():

    files = list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE
    )

    allowed_extensions = (
        ".txt",
        ".md",
        ".markdown"
    )

    selected_files = []

    for file_path in files:

        # Consideriamo soltanto i file della cartella
        if not file_path.startswith(
            HF_FOLDER + "/"
        ):
            continue

        if not file_path.lower().endswith(
            allowed_extensions
        ):
            continue

        selected_files.append(
            file_path
        )

    return sorted(
        selected_files
    )


# ============================================================
# DOWNLOAD FILE
# ============================================================

@st.cache_data(show_spinner=False)
def download_hf_file(file_path):

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=file_path,
        repo_type=HF_REPO_TYPE
    )

    with open(
        local_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end]

        # Se non siamo alla fine, cerchiamo di
        # chiudere il chunk in corrispondenza
        # di una frase o di uno spazio.
        if end < text_length:

            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            last_space = chunk.rfind(" ")

            cut_position = max(
                last_period,
                last_newline,
                last_space
            )

            if cut_position > chunk_size * 0.60:

                end = start + cut_position + 1

                chunk = text[
                    start:end
                ]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# COSTRUZIONE CORPUS
# ============================================================

@st.cache_data(show_spinner=False)
def build_corpus():

    files = get_hf_files()

    documents = []

    for file_path in files:

        try:

            raw_text = download_hf_file(
                file_path
            )

        except Exception:
            continue

        cleaned = clean_text(
            raw_text
        )

        if not cleaned:
            continue

        chunks = chunk_text(
            cleaned
        )

        for index, chunk in enumerate(
            chunks
        ):

            documents.append(
                {
                    "text": chunk,
                    "source": file_path,
                    "chunk_index": index
                }
            )

    return documents


# ============================================================
# EMBEDDING DEL CORPUS
# ============================================================

@st.cache_resource(show_spinner=True)
def build_embeddings():

    documents = build_corpus()

    if not documents:
        raise Exception(
            "Nessun documento trovato "
            "nel dataset Hugging Face."
        )

    model = get_embedding_model()

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return documents, embeddings


# ============================================================
# RICERCA SEMANTICA
# ============================================================

def retrieve_documents(
    question,
    top_k=DEFAULT_TOP_K,
    minimum_score=MINIMUM_SCORE
):

    documents, embeddings = (
        build_embeddings()
    )

    model = get_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Con embedding normalizzati,
    # il prodotto scalare equivale
    # alla cosine similarity.
    scores = np.dot(
        embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        if score < minimum_score:
            continue

        document = dict(
            documents[index]
        )

        document["score"] = score

        results.append(
            document
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# COSTRUZIONE DEL CONTESTO
# ============================================================

def build_context(documents):

    if not documents:
        return (
            "Nessun passaggio rilevante "
            "è stato recuperato dal corpus."
        )

    context_parts = []

    for index, document in enumerate(
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
            "score",
            0
        )

        context_parts.append(
            f"""
[FONTE {index}]
File: {source}
Rilevanza: {score:.3f}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# PROMPT GIURIDICO
# ============================================================

SYSTEM_PROMPT = """
Sei un assistente universitario specializzato
in diritto privato italiano.

Il tuo compito è assistere nello studio del diritto
privato attraverso le fonti e i materiali messi a
disposizione dal sistema.

REGOLE FONDAMENTALI:

1. Utilizza prioritariamente il contesto giuridico
   fornito dal sistema.

2. Non inventare articoli del Codice civile,
   sentenze, orientamenti giurisprudenziali,
   definizioni dottrinali o riferimenti bibliografici.

3. Se il contesto non contiene informazioni
   sufficienti per rispondere con sicurezza,
   dichiaralo espressamente.

4. Non presentare come certa un'informazione
   che non è supportata dal contesto.

5. Utilizza una terminologia giuridica italiana
   precisa.

6. Quando il contesto contiene riferimenti
   normativi, riportali correttamente.

7. Distingui, quando possibile, tra:
   - disposizione normativa;
   - istituto giuridico;
   - interpretazione;
   - dottrina;
   - giurisprudenza.

8. La risposta deve privilegiare il ragionamento
   giuridico e i rapporti tra gli istituti, non
   una semplice riproduzione meccanica del testo.

9. Non attribuire al Manuale Galgano contenuti
   che non risultano dal contesto recuperato.

10. Se la domanda è ambigua, esplicita
    l'interpretazione utilizzata.

STRUTTURA DELLA RISPOSTA:

Quando opportuno:
- individua l'istituto;
- indica le norme rilevanti;
- spiega il funzionamento;
- evidenzia presupposti, effetti e limiti;
- collega gli istituti tra loro.

L'obiettivo è fornire una risposta utile
allo studio universitario del diritto privato.
"""


# ============================================================
# GENERAZIONE RISPOSTA CON GROQ
# ============================================================

def generate_answer(
    question,
    context
):

    client = get_groq_client()

    user_prompt = f"""
CONTESTO GIURIDICO RECUPERATO
================================

{context}


DOMANDA DELL'UTENTE
================================

{question}


ISTRUZIONE

Rispondi alla domanda utilizzando
il contesto giuridico recuperato.

Se il contesto non è sufficiente,
indicalo chiaramente invece di inventare
informazioni.
"""

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_tokens=4000
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# FUNZIONE PRINCIPALE DEL RAG
# ============================================================

def answer_question(
    question,
    top_k=DEFAULT_TOP_K
):

    question = question.strip()

    if not question:

        return {
            "answer": (
                "Inserisci una domanda."
            ),
            "documents": [],
            "context": ""
        }

    # 1. Recupero semantico
    documents = retrieve_documents(
        question,
        top_k=top_k
    )

    # 2. Costruzione contesto
    context = build_context(
        documents
    )

    # 3. Generazione risposta
    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "documents": documents,
        "context": context
    }
