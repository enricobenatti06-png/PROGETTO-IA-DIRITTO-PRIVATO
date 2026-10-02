import streamlit as st

st.set_page_config(
    page_title="Assistente di Diritto Privato",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Test configurazione")

try:
    import supabase
    st.success("✅ Supabase installato correttamente")
except Exception as e:
    st.error("❌ Errore importando Supabase")
    st.code(str(e))

try:
    import groq
    st.success("✅ Groq installato correttamente")
except Exception as e:
    st.error("❌ Errore importando Groq")
    st.code(str(e))

try:
    import huggingface_hub
    st.success("✅ Hugging Face installato correttamente")
except Exception as e:
    st.error("❌ Errore importando Hugging Face")
    st.code(str(e))

try:
    from sentence_transformers import SentenceTransformer
    st.success("✅ Sentence Transformers installato correttamente")
except Exception as e:
    st.error("❌ Errore importando Sentence Transformers")
    st.code(str(e))
