"""
RAG – Manuale Galgano  (app.py)
Struttura attesa:
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
    pip install streamlit ollama
"""

import os
import re
import math
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
HF_REPO      = "enricobenatti06/manuale_galgano"   # dataset Hugging Face
DOCS_DIR     = Path("/tmp/manuale_galgano")         # cartella locale temporanea
INDEX_FILE   = Path("/tmp/.rag_index.pkl")
HASH_FILE    = Path("/tmp/.rag_hash.txt")

MAX_RESULTS  = 4
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]          # chiave da Streamlit Secrets
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Pesi boosting per livello gerarchico (più è profondo, più è specifico)
LEVEL_BOOST = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}   # livello 0 = root

# Sinonimi giuridici per query expansion
LEGAL_SYNONYMS: dict[str, list[str]] = {
    "contratto":       ["accordo", "negozio", "patto", "convenzione"],
    "proprietà":       ["dominio", "diritto reale", "titolarità"],
    "responsabilità":  ["illecito", "danno", "risarcimento", "colpa", "dolo"],
    "successione":     ["eredità", "testamento", "eredi", "legato", "mortis causa"],
    "obbligazione":    ["debito", "credito", "prestazione", "adempimento", "inadempimento"],
    "nullità":         ["invalidità", "inefficacia", "annullabilità", "vizio"],
    "risarcimento":    ["indennizzo", "ristoro", "riparazione"],
    "possesso":        ["detenzione", "animus", "corpus", "possessore"],
    "usucapione":      ["prescrizione acquisitiva", "acquisto originario"],
    "locazione":       ["affitto", "conduttore", "locatore", "canone"],
    "persona":         ["soggetto", "capacità", "personalità giuridica"],
    "famiglia":        ["matrimonio", "coniuge", "filiazione", "parentela"],
    "trust":           ["fiducia", "gestione patrimoniale"],
    "garanzia":        ["pegno", "ipoteca", "fideiussione", "cauzione"],
    "rappresentanza":  ["mandato", "procura", "agente", "preponente"],
}


# ---------------------------------------------------------------------------
# Tokenizzazione
# ---------------------------------------------------------------------------
STOPWORDS = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","con",
    "su","per","tra","fra","e","o","ma","che","non","si","del","della","dei",
    "degli","delle","al","alla","ai","agli","alle","dal","dalla","dai","dagli",
    "dalle","nel","nella","nei","negli","nelle","sul","sulla","sui","sugli",
    "sulle","col","come","anche","già","più","questo","questa","questi","queste",
    "quello","quella","quelli","quelle","sono","essere","avere","fare","può",
    "deve","hanno","aveva","sarà","suo","sua","suoi","sue","loro","tutto","tutti",
}

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zàèéìòùA-ZÀÈÉÌÒÙ]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Caricamento documenti — un file = un documento (i file sono già chunk)
# ---------------------------------------------------------------------------
def parse_hierarchy(path: Path, base: Path) -> dict:
    rel      = path.relative_to(base)
    parts    = list(rel.parts)          # es. [cap, sez, sottosez, argomento, file.txt]
    folders  = parts[:-1]               # tutto tranne il file
    filename = parts[-1].replace(".txt", "")
    return {
        "folders":      folders,        # lista completa delle cartelle, qualsiasi profondità
        "filename":     filename,
        "depth":        len(folders),
        "topic_tokens": tokenize(" ".join(parts)),  # tutti i livelli come token per il boosting
    }


def download_from_hf() -> None:
    """Scarica i file dal dataset Hugging Face se non già presenti."""
    from huggingface_hub import snapshot_download
    hf_token = st.secrets.get("HF_TOKEN", None)
    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*.txt")):
        st.info("⬇️ Download documenti da Hugging Face… (solo al primo avvio)")
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(DOCS_DIR),
            ignore_patterns=["*.json", "*.md", ".gitattributes"],
            token=hf_token,
        )


def load_all_chunks(base_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(base_dir.rglob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        meta = parse_hierarchy(path, base_dir)
        docs.append({
            "path":    str(path),
            "content": content,
            "meta":    meta,
            "tokens":  tokenize(content),
        })
    return docs


# ---------------------------------------------------------------------------
# Hash del corpus (per capire se rindexare)
# ---------------------------------------------------------------------------
def corpus_hash(base_dir: Path) -> str:
    h = hashlib.md5()
    for path in sorted(base_dir.rglob("*.txt")):
        h.update(str(path).encode())
        h.update(str(path.stat().st_mtime).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# BM25 nativo (senza dipendenze esterne)
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.n  = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[dict[str, float]] = []
        for doc in corpus:
            freq = Counter(doc)
            self.tf.append(freq)
            for term in freq:
                self.df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for i, tf in enumerate(self.tf):
            dl    = sum(tf.values())
            score = 0.0
            for term in query:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0)
                f   = tf[term]
                score += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores.append(score)
        return scores


def build_index(chunks: list[dict]) -> BM25:
    corpus = [c["tokens"] for c in chunks]
    return BM25(corpus)


@st.cache_resource(show_spinner="📚 Indicizzazione corpus… (solo al primo avvio)")
def load_index():
    download_from_hf()

    current_hash = corpus_hash(DOCS_DIR)

    # Usa indice cached se il corpus non è cambiato
    if INDEX_FILE.exists() and HASH_FILE.exists():
        if HASH_FILE.read_text().strip() == current_hash:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
            return data["chunks"], data["bm25"]

    # Rindexazione
    chunks = load_all_chunks(DOCS_DIR)
    if not chunks:
        st.error("Nessun file .txt trovato nel dataset Hugging Face.")
        st.stop()

    bm25 = build_index(chunks)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "bm25": bm25}, f)
    HASH_FILE.write_text(current_hash)

    return chunks, bm25


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
def expand_query(query: str) -> list[str]:
    words = tokenize(query)
    expanded = list(words)
    for word in words:
        for key, syns in LEGAL_SYNONYMS.items():
            if word == key or word in syns:
                expanded += [key] + syns
    return list(dict.fromkeys(expanded))


# ---------------------------------------------------------------------------
# Retrieval BM25 + boosting gerarchico
# ---------------------------------------------------------------------------
def retrieve(query: str, chunks: list[dict], bm25: BM25, max_results: int = MAX_RESULTS) -> list[dict]:
    query_tokens = expand_query(query)

    # Punteggi BM25 base
    bm25_scores = bm25.get_scores(query_tokens)

    # Boosting gerarchico: premia chunk il cui path contiene termini della query
    boosted = []
    for i, (chunk, score) in enumerate(zip(chunks, bm25_scores)):
        meta         = chunk["meta"]
        topic_match  = sum(1 for t in query_tokens if t in meta["topic_tokens"])
        depth_weight = LEVEL_BOOST.get(min(meta["depth"], 3), 2.0)
        final_score  = score + topic_match * depth_weight

        if final_score > 0:
            boosted.append((final_score, i, chunk))

    boosted.sort(reverse=True, key=lambda x: x[0])

    # Deduplicazione: max 2 chunk per file
    seen: dict[str, int] = defaultdict(int)
    results = []
    for score, _, chunk in boosted:
        p = chunk["path"]
        if seen[p] < 2:
            results.append((score, chunk))
            seen[p] += 1
        if len(results) >= max_results:
            break

    return results   # lista di (score, chunk)


# ---------------------------------------------------------------------------
# Costruzione contesto per il prompt
# ---------------------------------------------------------------------------
def build_context(results: list[tuple]) -> str:
    parts = []
    for score, chunk in results:
        meta = chunk["meta"]
        breadcrumb = " > ".join(meta["folders"] + [meta["filename"]])
        parts.append(f"[{breadcrumb}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def ask_ollama(prompt: str, model: str = GROQ_MODEL) -> str:
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPTS = {
    "Chat": """Sei un assistente esperto di diritto privato italiano (manuale Galgano).
Rispondi alla domanda usando SOLO il contesto fornito. Sii preciso e tecnico.
Se il contesto non contiene informazioni sufficienti, dillo esplicitamente.

CONTESTO:
{context}

DOMANDA:
{query}

RISPOSTA:""",

    "Schema": """Sei un assistente esperto di diritto privato italiano (manuale Galgano).
Trasforma il contenuto in uno schema ordinato per lo studio universitario.
Struttura: Definizione → Fondamento normativo → Elementi essenziali → Effetti → Eccezioni/Limiti.
Non inventare informazioni non presenti nel contesto.

CONTESTO:
{context}

ARGOMENTO:
{query}

SCHEMA:""",

    "Flashcard": """Sei un assistente esperto di diritto privato italiano (manuale Galgano).
Crea 10 flashcard di studio sull'argomento usando SOLO il contesto.

Formato rigoroso:
FRONT: [domanda tecnica]
BACK: [risposta concisa e precisa]

CONTESTO:
{context}

ARGOMENTO:
{query}

FLASHCARD:""",

    "Quiz": """Sei un assistente esperto di diritto privato italiano (manuale Galgano).
Crea un quiz sull'argomento usando SOLO il contesto.

Genera:
- 5 domande a risposta multipla (A/B/C/D) con risposta corretta indicata
- 3 vero/falso con spiegazione
- 2 domande aperte brevi con risposta modello

CONTESTO:
{context}

ARGOMENTO:
{query}

QUIZ:""",
}


# ---------------------------------------------------------------------------
# Storico — salvataggio e lettura (completo: domanda, risposta, fonti, modalità, timestamp)
# ---------------------------------------------------------------------------
import json as _json
from datetime import datetime

LOG_PATH = Path("/tmp/storico.json")

BADGE_COLORS = {
    "Chat":      "#4F8EF7",
    "Schema":    "#9B59B6",
    "Flashcard": "#27AE60",
    "Quiz":      "#E67E22",
}

def save_log(query: str, mode: str, output: str, fonti: list[dict]) -> None:
    try:
        existing = load_log()
        entry = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "modalita":  mode,
            "domanda":   query,
            "risposta":  output,
            "fonti": [
                {
                    "breadcrumb": " > ".join(d["meta"]["folders"] + [d["meta"]["filename"]]),
                    "anteprima":  d["content"][:300],
                }
                for _, d in fonti
            ],
        }
        existing.append(entry)
        existing = existing[-300:]
        LOG_PATH.write_text(_json.dumps(existing, ensure_ascii=False, indent=2))
        st.session_state["_log_cache"] = existing
    except Exception:
        pass


def load_log() -> list:
    if "._log_cache" in st.session_state:
        return st.session_state["_log_cache"]
    if LOG_PATH.exists():
        try:
            data = _json.loads(LOG_PATH.read_text())
            st.session_state["_log_cache"] = data
            return data
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# CSS personalizzato — UI moderna e pulita
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Font e sfondo */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Header principale */
.main-header {
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}
.main-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 0.85rem; color: #6b7280; margin: 0.2rem 0 0 0; }

/* Card risposta */
.answer-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}
.answer-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

/* Badge modalità */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 99px;
    color: white;
    letter-spacing: 0.03em;
}

/* Fonte pill */
.fonte-pill {
    display: inline-block;
    font-size: 0.75rem;
    background: #eef2ff;
    color: #4338ca;
    border-radius: 6px;
    padding: 3px 8px;
    margin: 2px 3px 2px 0;
}

/* Storico card */
.log-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background: white;
}
.log-card-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.log-card-q {
    font-size: 0.95rem;
    font-weight: 500;
    color: #111827;
    margin-bottom: 0.5rem;
}
</style>
"""


# ---------------------------------------------------------------------------
# App Streamlit
# ---------------------------------------------------------------------------
chunks, bm25 = load_index()

st.set_page_config(
    page_title="IA Studio – Diritto Privato",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚖️ IA Studio")
    st.markdown("*Manuale Galgano — Diritto Privato*")
    st.divider()
    pagina = st.radio("Navigazione", ["🎓 Assistente", "📋 Storico"], label_visibility="collapsed")
    st.divider()
    st.caption(f"📚 {len(chunks)} documenti indicizzati")

# ---------------------------------------------------------------------------
# PAGINA: STORICO
# ---------------------------------------------------------------------------
if pagina == "📋 Storico":
    st.markdown('<div class="main-header"><h1>📋 Storico sessioni</h1><p>Tutte le domande e risposte salvate</p></div>', unsafe_allow_html=True)

    password = st.text_input("Password di accesso", type="password", placeholder="Inserisci la password…")
    PASSWORD_CORRETTA = st.secrets.get("STORICO_PASSWORD", "galgano2024")

    if password != PASSWORD_CORRETTA:
        st.warning("Inserisci la password per accedere allo storico.")
        st.stop()

    logs = load_log()
    if not logs:
        st.info("Nessuna sessione registrata ancora.")
        st.stop()

    # Barra in alto: contatore + filtri + export
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**{len(logs)} sessioni registrate**")
    with col2:
        modalita_filter = st.selectbox("Filtra per modalità", ["Tutte", "Chat", "Schema", "Flashcard", "Quiz"], label_visibility="collapsed")
    with col3:
        st.download_button(
            "⬇️ Esporta JSON",
            data=_json.dumps(logs, ensure_ascii=False, indent=2),
            file_name="storico_diritto_privato.json",
            mime="application/json",
        )

    st.divider()

    filtered = [e for e in reversed(logs) if modalita_filter == "Tutte" or e["modalita"] == modalita_filter]

    for entry in filtered:
        color = BADGE_COLORS.get(entry["modalita"], "#6b7280")
        badge = f'<span class="badge" style="background:{color}">{entry["modalita"]}</span>'
        fonti_pills = "".join(
            f'<span class="fonte-pill">{f["breadcrumb"]}</span>'
            for f in entry.get("fonti", [])
        )
        with st.expander(f"{entry['timestamp']}  —  {entry['domanda'][:70]}{'…' if len(entry['domanda'])>70 else ''}"):
            st.markdown(
                f'<div class="log-card-meta">{badge} &nbsp; 🕐 {entry["timestamp"]}</div>'
                f'<div class="log-card-q">❓ {entry["domanda"]}</div>',
                unsafe_allow_html=True
            )
            if fonti_pills:
                st.markdown(f"**Fonti:** {fonti_pills}", unsafe_allow_html=True)
            st.markdown("---")
            st.write(entry["risposta"])

# ---------------------------------------------------------------------------
# PAGINA: ASSISTENTE
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="main-header"><h1>⚖️ IA Studio – Diritto Privato</h1><p>Assistente basato sul Manuale Galgano</p></div>', unsafe_allow_html=True)

    # Controlli sidebar
    with st.sidebar:
        mode = st.selectbox("📌 Modalità", ["Chat", "Schema", "Flashcard", "Quiz"])
        with st.expander("⚙️ Impostazioni retrieval"):
            max_results  = st.slider("Documenti recuperati", 2, 10, MAX_RESULTS)
            show_sources = st.checkbox("Mostra fonti", value=True)
            show_scores  = st.checkbox("Mostra punteggi BM25", value=False)

    # Input query
    query = st.text_input(
        "Inserisci argomento o domanda",
        placeholder="es. responsabilità extracontrattuale, usucapione, nullità del contratto…",
        label_visibility="collapsed",
    )

    col_btn, col_tip = st.columns([1, 5])
    with col_btn:
        genera = st.button("Genera →", type="primary", use_container_width=True)
    with col_tip:
        st.caption("Prova: *nullità del contratto*, *successione testamentaria*, *possesso e usucapione*")

    if genera and query:
        with st.spinner("Ricerca nel corpus…"):
            results = retrieve(query, chunks, bm25, max_results=max_results)

        if not results:
            st.warning("⚠️ Nessun documento rilevante trovato. Prova con termini più specifici o sinonimi.")
            st.stop()

        context = build_context(results)
        prompt  = PROMPTS[mode].format(context=context, query=query)

        with st.spinner("Generazione risposta…"):
            output = ask_ollama(prompt)

        # Salva nello storico (con fonti)
        save_log(query, mode, output, results)

        # Mostra risposta in card
        color = BADGE_COLORS.get(mode, "#6b7280")
        badge = f'<span class="badge" style="background:{color}">{mode}</span>'
        st.markdown(
            f'<div class="answer-card">'
            f'<div class="answer-card-header">{badge} <span style="font-size:0.85rem;color:#6b7280">Risposta generata</span></div>',
            unsafe_allow_html=True
        )
        st.write(output)
        st.markdown('</div>', unsafe_allow_html=True)

        # Fonti
        if show_sources:
            with st.expander(f"📄 Fonti utilizzate ({len(results)} documenti)"):
                for score, chunk in results:
                    meta       = chunk["meta"]
                    breadcrumb = " > ".join(meta["folders"] + [meta["filename"]])
                    header     = f"**{breadcrumb}**"
                    if show_scores:
                        header += f"  `score: {score:.3f}`"
                    st.markdown(header)
                    st.caption(chunk["content"][:400] + ("…" if len(chunk["content"]) > 400 else ""))
                    st.divider()
