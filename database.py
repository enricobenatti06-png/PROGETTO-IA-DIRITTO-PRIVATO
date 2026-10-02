import streamlit as st
from supabase import create_client


@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


supabase = get_supabase()


# =========================
# CONVERSAZIONI
# =========================

def get_conversations(user_id):
    response = (
        supabase
        .table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data


def create_conversation(user_id, title="Nuova conversazione"):
    response = (
        supabase
        .table("conversations")
        .insert({
            "user_id": user_id,
            "title": title
        })
        .execute()
    )

    return response.data[0]


def get_conversation(conversation_id, user_id):
    response = (
        supabase
        .table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    return response.data


# =========================
# MESSAGGI
# =========================

def get_messages(conversation_id):
    response = (
        supabase
        .table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )

    return response.data


def create_message(conversation_id, role, content):
    response = (
        supabase
        .table("messages")
        .insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        })
        .execute()
    )

    return response.data[0]


def update_conversation_title(conversation_id, title):
    response = (
        supabase
        .table("conversations")
        .update({
            "title": title
        })
        .eq("id", conversation_id)
        .execute()
    )

    return response.data
