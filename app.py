import streamlit as st
import os

st.set_page_config(
    page_title="Diagnostica progetto",
    page_icon="⚖️"
)

st.title("🔎 Diagnostica progetto")

st.subheader("Cartella di esecuzione")

st.code(os.getcwd())

st.subheader("File presenti")

files = os.listdir(".")

for file in sorted(files):
    st.write(f"• {file}")

st.divider()

st.subheader("Verifica file Python")

for filename in ["app.py", "auth.py", "database.py", "chat.py", "rag.py"]:
    if os.path.exists(filename):
        st.success(f"✅ {filename} trovato")
    else:
        st.error(f"❌ {filename} NON trovato")

st.divider()

st.subheader("Test import auth")

try:
    import auth
    st.success("✅ `auth` importato correttamente")
    st.write(auth.__file__)
except Exception as e:
    st.error("❌ `auth` NON importabile")
    st.exception(e)
