import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="Test Supabase",
    page_icon="🔧"
)

st.title("🔧 Test Supabase Auth")

st.write("URL configurato:")
st.code(st.secrets["SUPABASE_URL"])

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

    st.success("✅ Client Supabase creato correttamente")

except Exception as e:
    st.error("❌ Errore nella creazione del client")
    st.exception(e)
    st.stop()


st.divider()

st.subheader("Test registrazione")

email = st.text_input(
    "Email",
    placeholder="test@example.com"
)

password = st.text_input(
    "Password",
    type="password"
)

if st.button("Test registrazione"):

    if not email or not password:
        st.warning("Inserisci email e password.")
        st.stop()

    try:

        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        st.success("✅ Richiesta di registrazione eseguita.")

        st.write("Risposta Supabase:")
        st.write(response)

    except Exception as e:

        st.error("❌ ERRORE SUPABASE AUTH")

        st.exception(e)
