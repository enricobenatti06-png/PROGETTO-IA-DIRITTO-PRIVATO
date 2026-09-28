"""
IA DIRITTO PRIVATO – Manuale Galgano
Interfaccia Streamlit + RAG BM25

Struttura:
    app.py
    fonti/
    └── manuale_galgano/
        ├── <capitolo>/
        │   ├── <sezione>/
        │   │   ├── <argomento>.txt
        │   │   └── ...
        │   └── ...
        └── ...

Dipendenze:
    pip install streamlit groq huggingface_hub
"""

import os
import re
import math
import pickle
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import streamlit as st
from groq import Groq


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

HF_REPO = "enricobenatti06/manuale_galgano"

DOCS_DIR = Path("/tmp/manuale_galgano")
INDEX_FILE = Path("/tmp/.rag_index.pkl")
HASH_FILE = Path("/tmp/.rag_hash.txt")
CONVERSATIONS_FILE = Path("/tmp/conversazioni.json")

MAX_RESULTS = 4

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "openai/gpt-oss-120b"

LEVEL_BOOST = {
    0: 0.5,
    1: 1.0,
    2: 1.5,
    3: 2.0,
}


# ============================================================================
# SINONIMI GIURIDICI
# ============================================================================

LEGAL_SYNONYMS: dict[str, list[str]] = {

    "contratto": [
        "accordo",
        "negozio",
        "patto",
        "convenzione",
    ],

    "proprietà": [
        "dominio",
        "diritto reale",
        "titolarità",
    ],

    "responsabilità": [
        "illecito",
        "danno",
        "risarcimento",
        "colpa",
        "dolo",
    ],

    "successione": [
        "eredità",
        "testamento",
        "eredi",
        "legato",
        "mortis causa",
    ],

    "obbligazione": [
        "debito",
        "credito",
        "prestazione",
        "adempimento",
        "inadempimento",
    ],

    "nullità": [
        "invalidità",
        "inefficacia",
        "annullabilità",
        "vizio",
    ],

    "risarcimento": [
        "indennizzo",
        "ristoro",
        "riparazione",
    ],

    "possesso": [
        "detenzione",
        "animus",
        "corpus",
        "possessore",
    ],

    "usucapione": [
        "prescrizione acquisitiva",
        "acquisto originario",
    ],

    "locazione": [
        "affitto",
        "conduttore",
        "locatore",
        "canone",
    ],

    "persona": [
        "soggetto",
        "capacità",
        "personalità giuridica",
    ],

    "famiglia": [
        "matrimonio",
        "coniuge",
        "filiazione",
        "parentela",
    ],

    "trust": [
        "fiducia",
        "gestione patrimoniale",
    ],

    "garanzia": [
        "pegno",
        "ipoteca",
        "fideiussione",
        "cauzione",
    ],

    "rappresentanza": [
        "mandato",
        "procura",
        "agente",
        "preponente",
    ],
}


# ============================================================================
# TOKENIZZAZIONE
# ============================================================================

STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "a", "da", "in", "con", "su",
    "per", "tra", "fra",
    "e", "o", "ma", "che", "non",
    "si", "del", "della", "dei", "degli", "delle",
    "al", "alla", "ai", "agli", "alle",
    "dal", "dalla", "dai", "dagli", "dalle",
    "nel", "nella", "nei", "negli", "nelle",
    "sul", "sulla", "sui", "sugli", "sulle",
    "col", "come", "anche", "già", "più",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "sono", "essere", "avere", "fare",
    "può", "deve", "hanno", "aveva",
    "sarà", "suo", "sua", "suoi", "sue",
    "loro", "tutto", "tutti",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(
        r"\b[a-zàèéìòùA-ZÀÈÉÌÒÙ]{3,}\b",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


# ============================================================================
# GERARCHIA DEI DOCUMENTI
# ============================================================================

def parse_hierarchy(path: Path, base: Path) -> dict:

    rel = path.relative_to(base)

    parts = list(rel.parts)

    folders = parts[:-1]

    filename = parts[-1].replace(".txt", "")

    return {
        "folders": folders,
        "filename": filename,
        "depth": len(folders),
        "topic_tokens": tokenize(" ".join(parts)),
    }


# ============================================================================
# DOWNLOAD HUGGING FACE
# ============================================================================

def download_from_hf() -> None:

    from huggingface_hub import snapshot_download

    hf_token = st.secrets.get("HF_TOKEN", None)

    if (
        not DOCS_DIR.exists()
        or not any(DOCS_DIR.rglob("*.txt"))
    ):

        st.info(
            "⬇️ Download del Manuale Galgano "
            "(solo al primo avvio)..."
        )

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(DOCS_DIR),
            ignore_patterns=[
                "*.json",
                "*.md",
                ".gitattributes",
            ],
            token=hf_token,
        )


# ============================================================================
# CARICAMENTO DOCUMENTI
# ============================================================================

def load_all_chunks(base_dir: Path) -> list[dict]:

    docs = []

    for path in sorted(base_dir.rglob("*.txt")):

        try:
            content = path.read_text(
                encoding="utf-8"
            ).strip()

        except Exception:
            continue

        if not content:
            continue

        meta = parse_hierarchy(
            path,
            base_dir
        )

        docs.append({
            "path": str(path),
            "content": content,
            "meta": meta,
            "tokens": tokenize(content),
        })

    return docs


# ============================================================================
# HASH CORPUS
# ============================================================================

def corpus_hash(base_dir: Path) -> str:

    h = hashlib.md5()

    for path in sorted(
        base_dir.rglob("*.txt")
    ):

        h.update(
            str(path).encode()
        )

        h.update(
            str(path.stat().st_mtime).encode()
        )

    return h.hexdigest()


# ============================================================================
# BM25
# ============================================================================

class BM25:

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):

        self.k1 = k1
        self.b = b
        self.n = len(corpus)

        self.avgdl = (
            sum(len(d) for d in corpus)
            / max(self.n, 1)
        )

        self.df: dict[str, int] = defaultdict(int)

        self.tf: list[
            dict[str, float]
        ] = []

        for doc in corpus:

            freq = Counter(doc)

            self.tf.append(freq)

            for term in freq:
                self.df[term] += 1

        self.idf = {

            term: math.log(
                (self.n - df + 0.5)
                / (df + 0.5)
                + 1
            )

            for term, df in self.df.items()
        }

    def get_scores(
        self,
        query: list[str]
    ) -> list[float]:

        scores = []

        for tf in self.tf:

            dl = sum(tf.values())

            score = 0.0

            for term in query:

                if term not in tf:
                    continue

                idf = self.idf.get(
                    term,
                    0
                )

                f = tf[term]

                score += (
                    idf
                    * (
                        f * (self.k1 + 1)
                    )
                    / (
                        f
                        + self.k1
                        * (
                            1
                            - self.b
                            + self.b
                            * dl
                            / self.avgdl
                        )
                    )
                )

            scores.append(score)

        return scores


def build_index(
    chunks: list[dict]
) -> BM25:

    corpus = [
        chunk["tokens"]
        for chunk in chunks
    ]

    return BM25(corpus)


# ============================================================================
# CARICAMENTO INDICE
# ============================================================================

@st.cache_resource(
    show_spinner="📚 Indicizzazione del corpus..."
)
def load_index():

    download_from_hf()

    current_hash = corpus_hash(
        DOCS_DIR
    )

    if (
        INDEX_FILE.exists()
        and HASH_FILE.exists()
    ):

        if (
            HASH_FILE.read_text().strip()
            == current_hash
        ):

            with open(
                INDEX_FILE,
                "rb"
            ) as f:

                data = pickle.load(f)

            return (
                data["chunks"],
                data["bm25"],
            )

    chunks = load_all_chunks(
        DOCS_DIR
    )

    if not chunks:

        st.error(
            "Nessun file .txt trovato "
            "nel dataset Hugging Face."
        )

        st.stop()

    bm25 = build_index(
        chunks
    )

    with open(
        INDEX_FILE,
        "wb"
    ) as f:

        pickle.dump(
            {
                "chunks": chunks,
                "bm25": bm25,
            },
            f
        )

    HASH_FILE.write_text(
        current_hash
    )

    return chunks, bm25


# ============================================================================
# QUERY EXPANSION
# ============================================================================

def expand_query(
    query: str
) -> list[str]:

    words = tokenize(query)

    expanded = list(words)

    for word in words:

        for key, synonyms in (
            LEGAL_SYNONYMS.items()
        ):

            if (
                word == key
                or word in synonyms
            ):

                expanded += [
                    key
                ]

                expanded += synonyms

    return list(
        dict.fromkeys(expanded)
    )


# ============================================================================
# RETRIEVAL
# ============================================================================

def retrieve(
    query: str,
    chunks: list[dict],
    bm25: BM25,
    max_results: int = MAX_RESULTS,
) -> list[tuple]:

    query_tokens = expand_query(
        query
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    boosted = []

    for i, (
        chunk,
        score
    ) in enumerate(
        zip(chunks, bm25_scores)
    ):

        meta = chunk["meta"]

        topic_match = sum(
            1
            for token in query_tokens
            if token in meta["topic_tokens"]
        )

        depth_weight = LEVEL_BOOST.get(
            min(meta["depth"], 3),
            2.0
        )

        final_score = (
            score
            + topic_match * depth_weight
        )

        if final_score > 0:

            boosted.append(
                (
                    final_score,
                    i,
                    chunk
                )
            )

    boosted.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    seen = defaultdict(int)

    results = []

    for score, _, chunk in boosted:

        path = chunk["path"]

        if seen[path] < 2:

            results.append(
                (
                    score,
                    chunk
                )
            )

            seen[path] += 1

        if len(results) >= max_results:
            break

    return results


# ============================================================================
# CONTEXT
# ============================================================================

def build_context(
    results: list[tuple]
) -> str:

    parts = []

    for score, chunk in results:

        meta = chunk["meta"]

        breadcrumb = " > ".join(
            meta["folders"]
            + [meta["filename"]]
        )

        parts.append(
            f"[{breadcrumb}]\n"
            f"{chunk['content']}"
        )

    return "\n\n---\n\n".join(
        parts
    )


# ============================================================================
# GROQ
# ============================================================================

def ask_groq(
    messages: list[dict]
) -> str:

    client = Groq(
        api_key=GROQ_API_KEY
    )

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=messages,

        max_tokens=2048,
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """
Sei un assistente universitario specializzato
in diritto privato italiano.

La tua banca dati principale è costituita
dal Manuale Galgano fornito nel contesto.

Regole:

1. Usa prioritariamente le informazioni
   contenute nel contesto fornito.

2. Non inventare contenuti giuridici
   che non risultano dal contesto.

3. Se il contesto non è sufficiente,
   dichiaralo esplicitamente.

4. Mantieni terminologia giuridica precisa.

5. Quando possibile, collega logicamente
   istituti, concetti e norme menzionate
   nel contesto.

6. Non attribuire al Manuale Galgano
   informazioni che non risultano presenti
   nei documenti recuperati.

7. Rispondi in italiano.

8. Per una domanda di studio, privilegia
   una spiegazione discorsiva ma schematica,
   adatta alla preparazione universitaria.
"""


MODE_INSTRUCTIONS = {

    "Chat": """
Rispondi normalmente alla domanda.
Costruisci una spiegazione chiara,
tecnica e ragionata.
""",

    "Schema": """
Trasforma il materiale recuperato
in uno schema ordinato per lo studio.

Struttura preferenziale:

Definizione
→ Fondamento
→ Elementi
→ Disciplina
→ Effetti
→ Limiti/eccezioni
→ Collegamenti

Non aggiungere informazioni non presenti
nel contesto.
""",

    "Flashcard": """
Crea 10 flashcard.

Formato:

FRONT: domanda tecnica
BACK: risposta precisa e concisa

Usa esclusivamente il contesto.
""",

    "Quiz": """
Crea un quiz composto da:

- 5 domande a risposta multipla A/B/C/D
- 3 vero/falso con spiegazione
- 2 domande aperte con risposta modello

Usa esclusivamente il contesto.
""",
}


# ============================================================================
# CONVERSAZIONI
# ============================================================================

def load_conversations() -> list:

    if not CONVERSATIONS_FILE.exists():
        return []

    try:

        return json.loads(
            CONVERSATIONS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []


def save_conversations(
    conversations: list
) -> None:

    try:

        CONVERSATIONS_FILE.write_text(
            json.dumps(
                conversations,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    except Exception:
        pass


def create_conversation() -> dict:

    now = datetime.now()

    return {
        "id": hashlib.md5(
            now.isoformat().encode()
        ).hexdigest()[:12],

        "title": "Nuova conversazione",

        "created_at": now.strftime(
            "%d/%m/%Y %H:%M"
        ),

        "updated_at": now.strftime(
            "%d/%m/%Y %H:%M"
        ),

        "messages": [],
    }


def save_current_conversation():

    conversation = (
        st.session_state.conversation
    )

    conversations = load_conversations()

    found = False

    for i, existing in enumerate(
        conversations
    ):

        if (
            existing["id"]
            == conversation["id"]
        ):

            conversations[i] = conversation

            found = True

            break

    if not found:

        conversations.append(
            conversation
        )

    conversations = conversations[-100:]

    save_conversations(
        conversations
    )


def load_conversation(
    conversation_id: str
):

    conversations = load_conversations()

    for conversation in conversations:

        if (
            conversation["id"]
            == conversation_id
        ):

            return conversation

    return None


# ============================================================================
# SESSION STATE
# ============================================================================

if "conversation" not in st.session_state:

    st.session_state.conversation = (
        create_conversation()
    )


if "last_sources" not in st.session_state:

    st.session_state.last_sources = {}


if "show_sources" not in st.session_state:

    st.session_state.show_sources = True


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="IA Diritto Privato",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================================
# CSS
# ============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 0.7rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# CARICAMENTO DATABASE
# ============================================================================

chunks, bm25 = load_index()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        "## ⚖️ IA Diritto Privato"
    )

    st.caption(
        "Assistente RAG — Manuale Galgano"
    )

    st.divider()

    # ------------------------------------------------------------
    # NUOVA CHAT
    # ------------------------------------------------------------

    if st.button(
        "＋ Nuova conversazione",
        use_container_width=True
    ):

        st.session_state.conversation = (
            create_conversation()
        )

        st.session_state.last_sources = {}

        st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # MODALITÀ
    # ------------------------------------------------------------

    st.markdown(
        "### 🎓 Modalità"
    )

    mode = st.selectbox(
        "Modalità di risposta",
        [
            "Chat",
            "Schema",
            "Flashcard",
            "Quiz",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # ------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------

    with st.expander(
        "⚙️ Retrieval"
    ):

        max_results = st.slider(
            "Documenti recuperati",
            2,
            10,
            MAX_RESULTS
        )

        st.session_state.show_sources = (
            st.checkbox(
                "Mostra fonti",
                value=True
            )
        )

        show_scores = st.checkbox(
            "Mostra punteggio BM25",
            value=False
        )

    st.divider()

    # ------------------------------------------------------------
    # CONVERSAZIONI PRECEDENTI
    # ------------------------------------------------------------

    st.markdown(
        "### 🕘 Conversazioni"
    )

    conversations = load_conversations()

    if conversations:

        for conversation in reversed(
            conversations[-10:]
        ):

            title = conversation.get(
                "title",
                "Conversazione"
            )

            if len(title) > 35:
                title = title[:35] + "…"

            if st.button(
                title,
                key=f"conversation_{conversation['id']}",
                use_container_width=True
            ):

                loaded = load_conversation(
                    conversation["id"]
                )

                if loaded:

                    st.session_state.conversation = (
                        loaded
                    )

                    st.session_state.last_sources = {}

                    st.rerun()

    else:

        st.caption(
            "Nessuna conversazione salvata."
        )

    st.divider()

    st.caption(
        f"📚 {len(chunks)} documenti indicizzati"
    )


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    '<div class="main-title">⚖️ IA Diritto Privato</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Manuale Galgano · sistema RAG</div>',
    unsafe_allow_html=True
)


# ============================================================================
# CHAT
# ============================================================================

conversation = st.session_state.conversation

messages = conversation["messages"]


# ============================================================================
# MESSAGGI
# ============================================================================

for index, message in enumerate(messages):

    role = message["role"]

    if role == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            st.write(
                message["content"]
            )

            sources = message.get(
                "sources",
                []
            )

            if (
                st.session_state.show_sources
                and sources
            ):

                with st.expander(
                    f"📚 Fonti utilizzate ({len(sources)})"
                ):

                    for source in sources:

                        breadcrumb = source[
                            "breadcrumb"
                        ]

                        score = source[
                            "score"
                        ]

                        preview = source[
                            "preview"
                        ]

                        if show_scores:

                            st.markdown(
                                f"**{breadcrumb}**  "
                                f"`score: {score:.3f}`"
                            )

                        else:

                            st.markdown(
                                f"**{breadcrumb}**"
                            )

                        st.caption(
                            preview
                        )

                        st.divider()


# ============================================================================
# INPUT CHAT
# ============================================================================

query = st.chat_input(
    "Scrivi una domanda di diritto privato..."
)


# ============================================================================
# ELABORAZIONE DOMANDA
# ============================================================================

if query:

    # ------------------------------------------------------------
    # AGGIUNGI MESSAGGIO UTENTE
    # ------------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # Titolo automatico
    if conversation["title"] == "Nuova conversazione":

        title = query.strip()

        if len(title) > 55:
            title = title[:55] + "…"

        conversation["title"] = title

    conversation["updated_at"] = (
        datetime.now().strftime(
            "%d/%m/%Y %H:%M"
        )
    )

    # ------------------------------------------------------------
    # VISUALIZZA DOMANDA
    # ------------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(query)

    # ------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Ricerca nella banca dati..."
        ):

            results = retrieve(
                query,
                chunks,
                bm25,
                max_results=max_results
            )

        if not results:

            output = (
                "Non ho trovato documenti "
                "sufficientemente rilevanti "
                "nella banca dati del Manuale "
                "Galgano per rispondere alla domanda."
            )

            sources = []

        else:

            context = build_context(
                results
            )

            # ----------------------------------------------------
            # COSTRUZIONE DEL PROMPT
            # ----------------------------------------------------

            prompt = f"""
{SYSTEM_PROMPT}

MODALITÀ:
{mode}

ISTRUZIONI DELLA MODALITÀ:
{MODE_INSTRUCTIONS[mode]}

CONTESTO RECUPERATO DAL MANUALE GALGANO:
{context}

DOMANDA DELL'UTENTE:
{query}

Rispondi ora.
"""

            # ----------------------------------------------------
            # GENERAZIONE
            # ----------------------------------------------------

            with st.spinner(
                "⚖️ Elaborazione della risposta..."
            ):

                output = ask_groq(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )

            # ----------------------------------------------------
            # PREPARA FONTI
            # ----------------------------------------------------

            sources = []

            for score, chunk in results:

                meta = chunk["meta"]

                breadcrumb = " > ".join(
                    meta["folders"]
                    + [meta["filename"]]
                )

                preview = chunk[
                    "content"
                ]

                if len(preview) > 400:

                    preview = (
                        preview[:400]
                        + "…"
                    )

                sources.append(
                    {
                        "breadcrumb": breadcrumb,
                        "score": score,
                        "preview": preview,
                    }
                )

        # --------------------------------------------------------
        # MOSTRA RISPOSTA
        # --------------------------------------------------------

        st.write(output)

        if (
            st.session_state.show_sources
            and sources
        ):

            with st.expander(
                f"📚 Fonti utilizzate ({len(sources)})"
            ):

                for source in sources:

                    if show_scores:

                        st.markdown(
                            f"**{source['breadcrumb']}**  "
                            f"`score: {source['score']:.3f}`"
                        )

                    else:

                        st.markdown(
                            f"**{source['breadcrumb']}**"
                        )

                    st.caption(
                        source["preview"]
                    )

                    st.divider()

    # ------------------------------------------------------------
    # SALVA RISPOSTA
    # ------------------------------------------------------------

    messages.append(
        {
            "role": "assistant",
            "content": output,
            "mode": mode,
            "sources": sources,
            "timestamp": datetime.now().strftime(
                "%d/%m/%Y %H:%M"
            ),
        }
    )

    conversation["updated_at"] = (
        datetime.now().strftime(
            "%d/%m/%Y %H:%M"
        )
    )

    save_current_conversation()

    st.rerun()
