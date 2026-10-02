import streamlit as st

st.set_page_config(
    page_title="Test configurazione",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Test configurazione")

st.write("Sto verificando i moduli del progetto...")

# =========================
# TEST AUTH
# =========================

try:
    from auth import (
        is_logged_in,
        get_current_user,
        show_auth_page,
        show_user_sidebar,
        load_user_session,
        supabase
    )

    st.success("✅ auth.py importato correttamente")

except Exception as e:
    st.error("❌ ERRORE IMPORTANDO auth.py")
    st.exception(e)


# =========================
# TEST DATABASE
# =========================

try:
    import database

    st.success("✅ database.py importato correttamente")

except Exception as e:
    st.error("❌ ERRORE IMPORTANDO database.py")
    st.exception(e)


# =========================
# TEST CHAT
# =========================

try:
    import chat

    st.success("✅ chat.py importato correttamente")

except Exception as e:
    st.error("❌ ERRORE IMPORTANDO chat.py")
    st.exception(e)


# =========================
# TEST RAG
# =========================

try:
    import rag

    st.success("✅ rag.py importato correttamente")

except Exception as e:
    st.error("❌ ERRORE IMPORTANDO rag.py")
    st.exception(e)


# =========================
# TEST LIBRERIE
# =========================

st.divider()

st.subheader("Test librerie installate")


try:
    import supabase

    st.success("✅ Supabase installato")

except Exception as e:
    st.error("❌ Supabase non disponibile")
    st.exception(e)


try:
    import groq

    st.success("✅ Groq installato")

except Exception as e:
    st.error("❌ Groq non disponibile")
    st.exception(e)


try:
    import huggingface_hub

    st.success("✅ Hugging Face Hub installato")

except Exception as e:
    st.error("❌ Hugging Face Hub non disponibile")
    st.exception(e)


try:
    from sentence_transformers import SentenceTransformer

    st.success("✅ Sentence Transformers installato")

except Exception as e:
    st.error("❌ Sentence Transformers non disponibile")
    st.exception(e)


try:
    import torch

    st.success("✅ PyTorch installato")

except Exception as e:
    st.error("❌ PyTorch non disponibile")
    st.exception(e)


try:
    import numpy

    st.success("✅ NumPy installato")

except Exception as e:
    st.error("❌ NumPy non disponibile")
    st.exception(e)


st.divider()
