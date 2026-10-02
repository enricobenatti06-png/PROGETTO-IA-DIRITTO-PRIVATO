import streamlit as st

from database import (
    get_conversations,
    create_conversation,
    get_messages,
    create_message,
    update_conversation_title
)


def show_chat(user):

    user_id = user.id

    # =====================================
    # INIZIALIZZAZIONE SESSION STATE
    # =====================================

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


    # =====================================
    # SIDEBAR CONVERSAZIONI
    # =====================================

    with st.sidebar:

        st.title("Conversazioni")

        if st.button(
            "＋ Nuova conversazione",
            use_container_width=True
        ):

            conversation = create_conversation(
                user_id,
                "Nuova conversazione"
            )

            st.session_state.conversation_id = conversation["id"]

            st.rerun()


        st.divider()


        conversations = get_conversations(user_id)


        for conversation in conversations:

            title = conversation.get(
                "title",
                "Nuova conversazione"
            )

            conversation_id = conversation["id"]


            if st.button(
                title,
                key=f"conversation_{conversation_id}",
                use_container_width=True
            ):

                st.session_state.conversation_id = conversation_id

                st.rerun()


    # =====================================
    # NESSUNA CONVERSAZIONE SELEZIONATA
    # =====================================

    if st.session_state.conversation_id is None:

        st.title("Assistente di Diritto Privato")

        st.write(
            "Seleziona una conversazione oppure creane una nuova."
        )

        return


    # =====================================
    # MESSAGGI
    # =====================================

    conversation_id = st.session_state.conversation_id

    messages = get_messages(conversation_id)


    st.title("Assistente di Diritto Privato")


    for message in messages:

        role = message["role"]
        content = message["content"]

        if role == "user":

            with st.chat_message("user"):
                st.write(content)

        else:

            with st.chat_message("assistant"):
                st.write(content)


    # =====================================
    # INPUT
    # =====================================

    prompt = st.chat_input(
        "Scrivi la tua domanda..."
    )


    if prompt:

        # -----------------------------
        # SALVA DOMANDA
        # -----------------------------

        create_message(
            conversation_id,
            "user",
            prompt
        )


        # -----------------------------
        # GENERAZIONE RISPOSTA
        # -----------------------------

        answer = generate_answer(prompt)


        # -----------------------------
        # SALVA RISPOSTA
        # -----------------------------

        create_message(
            conversation_id,
            "assistant",
            answer
        )


        # -----------------------------
        # TITOLO AUTOMATICO
        # -----------------------------

        if len(messages) == 0:

            title = prompt[:50]

            update_conversation_title(
                conversation_id,
                title
            )


        st.rerun()


# =====================================
# TEMPORANEO
# =====================================

def generate_answer(prompt):

    return (
        "Questa è una risposta di prova. "
        "Qui verrà collegato il motore RAG "
        "giuridico."
    )
