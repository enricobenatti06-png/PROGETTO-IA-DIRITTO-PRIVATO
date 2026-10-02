import streamlit as st
from supabase import create_client


# ============================================================
# SUPABASE
# ============================================================

def create_supabase_client():

    url = st.secrets["SUPABASE_URL"].strip()
    key = st.secrets["SUPABASE_KEY"].strip()

    return create_client(url, key)


supabase = create_supabase_client()


# ============================================================
# SESSIONE
# ============================================================

def get_current_user():

    return st.session_state.get("user")


def get_current_role():

    return st.session_state.get("role")


def is_logged_in():

    return (
        "user" in st.session_state
        and st.session_state.user is not None
    )


def logout():

    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    st.session_state.clear()
    st.rerun()


# ============================================================
# RECUPERO PROFILO
# ============================================================

def get_profile(user_id):

    response = (
        supabase
        .table("profiles")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )

    return response.data


def load_user_session(user):

    profile = get_profile(user.id)

    if profile is None:
        profile = {"role": "user"}

    st.session_state.user = user
    st.session_state.profile = profile
    st.session_state.role = profile.get("role", "user")

    return profile


# ============================================================
# LOGIN
# ============================================================

def login(email, password):

    try:

        client = create_supabase_client()

        response = client.auth.sign_in_with_password({
            "email": email.strip(),
            "password": password
        })

        user = response.user

        if user is None:
            return False, "Login non riuscito."

        load_user_session(user)

        return True, "Login effettuato."

    except Exception as e:

        return False, f"{type(e).__name__}: {str(e)}"


# ============================================================
# REGISTRAZIONE
# ============================================================

def register(email, password):

    try:

        # Creiamo un client NUOVO ad ogni registrazione.
        client = create_supabase_client()

        clean_email = email.strip()

        response = client.auth.sign_up({
            "email": clean_email,
            "password": password
        })

        user = response.user

        if user is None:
            return False, "Supabase non ha restituito un utente."

        return True, (
            "Registrazione completata. "
            "Controlla la tua email per confermare l'account."
        )

    except Exception as e:

        return False, f"{type(e).__name__}: {str(e)}"


# ============================================================
# INTERFACCIA LOGIN
# ============================================================

def show_auth_page():

    st.title("⚖️ Assistente di Diritto Privato")

    st.write(
        "Accedi al sistema per utilizzare le conversazioni "
        "e l'assistente giuridico."
    )

    login_tab, register_tab = st.tabs([
        "Accedi",
        "Registrati"
    ])


    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        with st.form("login_form"):

            email = st.text_input(
                "Email",
                placeholder="nome@email.com"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Accedi",
                use_container_width=True
            )

        if submitted:

            if not email or not password:

                st.error("Inserisci email e password.")

            else:

                success, message = login(
                    email,
                    password
                )

                if success:

                    st.success(message)
                    st.rerun()

                else:

                    st.error(message)


    # ========================================================
    # REGISTRAZIONE
    # ========================================================

    with register_tab:

        with st.form("register_form"):

            email = st.text_input(
                "Email",
                key="register_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="register_password"
            )

            password_confirm = st.text_input(
                "Conferma password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Crea account",
                use_container_width=True
            )

        if submitted:

            if not email or not password:

                st.error("Compila tutti i campi.")

            elif password != password_confirm:

                st.error("Le password non coincidono.")

            elif len(password) < 6:

                st.error(
                    "La password deve contenere almeno 6 caratteri."
                )

            else:

                success, message = register(
                    email,
                    password
                )

                if success:

                    st.success(message)

                else:

                    st.error(message)


# ============================================================
# SIDEBAR UTENTE
# ============================================================

def show_user_sidebar():

    if not is_logged_in():
        return

    user = get_current_user()
    role = get_current_role()

    with st.sidebar:

        st.divider()

        st.caption(
            f"👤 {user.email}"
        )

        st.caption(
            f"Ruolo: **{role}**"
        )

        if st.button(
            "Esci",
            use_container_width=True
        ):

            logout()
