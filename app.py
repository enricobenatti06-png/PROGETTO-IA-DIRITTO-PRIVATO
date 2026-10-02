import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="Test registrazione",
    page_icon="⚖️"
)

st.title("⚖️ Test registrazione")

# ============================================================
# CLIENT SUPABASE
# ============================================================

url = st.secrets["SUPABASE_URL"].strip()
key = st.secrets["SUPABASE_KEY"].strip()

st.caption("Project URL utilizzato:")
st.code(url)

supabase = create_client(url, key)

st.success("✅ Client Supabase creato")


# ============================================================
# REGISTRAZIONE
# ============================================================

st.divider()

email = st.text_input(
    "Email",
    placeholder="nuova-email@example.com"
)

password = st.text_input(
    "Password",
    type="password"
)

if st.button(
    "Crea account",
    use_container_width=True
):

    if not email or not password:
        st.error("Inserisci email e password.")
        st.stop()

    try:

        response = supabase.auth.sign_up(
            {
                "email": email.strip(),
                "password": password
            }
        )

        st.success("✅ REGISTRAZIONE RIUSCITA")

        st.write(response)

    except Exception as e:

        st.error("❌ ERRORE")

        st.write("Tipo errore:")
        st.code(type(e).__name__)

        st.write("Messaggio:")
        st.code(str(e))

        st.write("URL utilizzato:")
        st.code(url)
