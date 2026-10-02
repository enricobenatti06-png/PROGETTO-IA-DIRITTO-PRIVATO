import os
from typing import List, Dict, Optional


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Modello utilizzato per generare la risposta.
# Verrà configurato successivamente tramite Streamlit Secrets.
MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "gemma3:1b"
)


# ============================================================
# RICERCA DOCUMENTALE
# ============================================================

def search_documents(
    query: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Cerca nei documenti giuridici quelli più pertinenti
    rispetto alla domanda dell'utente.

    Per ora restituisce una lista vuota.
    In seguito verrà collegata a ChromaDB.
    """

    # TODO:
    # 1. trasformare query in embedding
    # 2. interrogare ChromaDB
    # 3. recuperare i documenti più pertinenti
    # 4. restituire testo + metadati

    return []


# ============================================================
# COSTRUZIONE DEL CONTESTO
# ============================================================

def build_context(
    documents: List[Dict]
) -> str:
    """
    Costruisce il contesto giuridico da fornire al modello.
    """

    if not documents:
        return ""

    context_parts = []

    for document in documents:

        text = document.get("text", "")

        source = document.get(
            "source",
            "Fonte non specificata"
        )

        context_parts.append(
            f"FONTE: {source}\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# COSTRUZIONE DEL PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str
) -> str:
    """
    Costruisce il prompt destinato al modello linguistico.
    """

    if context:

        return f"""
Sei un assistente di diritto privato.

Devi rispondere alla domanda dell'utente utilizzando
principalmente il materiale giuridico fornito nel contesto.

Non inventare norme, articoli, sentenze o citazioni.

Distingui sempre:
- disposizione normativa;
- interpretazione dottrinale;
- orientamento giurisprudenziale;
- eventuali ricostruzioni interpretative.

Quando è pertinente, indica gli articoli del Codice civile.

CONTESTO GIURIDICO:

{context}

DOMANDA DELL'UTENTE:

{question}

FORMULA UNA RISPOSTA CHIARA, PRECISA E GIURIDICAMENTE ARGOMENTATA.
"""

    return f"""
Sei un assistente di diritto privato.

Rispondi alla seguente domanda in modo chiaro e preciso.

Non inventare riferimenti normativi o giurisprudenziali.

Se non disponi di una fonte sufficiente per verificare
un'affermazione, dichiaralo.

DOMANDA:

{question}
"""


# ============================================================
# GENERAZIONE DELLA RISPOSTA
# ============================================================

def generate_with_llm(
    prompt: str
) -> str:
    """
    Invia il prompt al modello linguistico.

    Per ora utilizza un fallback.
    Questa funzione verrà collegata a Ollama/Groq/OpenAI
    quando configureremo definitivamente il modello.
    """

    # TODO:
    # Collegamento al modello LLM.

    return (
        "Il motore RAG è configurato, ma il modello linguistico "
        "non è ancora collegato."
    )


# ============================================================
# FUNZIONE PRINCIPALE DEL RAG
# ============================================================

def answer_question(
    question: str,
    top_k: int = 5
) -> Dict:
    """
    Funzione principale utilizzata dalla chat.

    Pipeline:

    domanda
       ↓
    ricerca documenti
       ↓
    costruzione contesto
       ↓
    costruzione prompt
       ↓
    LLM
       ↓
    risposta
    """

    # --------------------------------------------------------
    # 1. Ricerca
    # --------------------------------------------------------

    documents = search_documents(
        question,
        top_k=top_k
    )


    # --------------------------------------------------------
    # 2. Contesto
    # --------------------------------------------------------

    context = build_context(
        documents
    )


    # --------------------------------------------------------
    # 3. Prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        context
    )


    # --------------------------------------------------------
    # 4. LLM
    # --------------------------------------------------------

    answer = generate_with_llm(
        prompt
    )


    # --------------------------------------------------------
    # 5. Risultato
    # --------------------------------------------------------

    return {
        "answer": answer,
        "documents": documents,
        "context": context
    }
