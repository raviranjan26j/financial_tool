import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(page_title="FinAI Main", layout="wide")

# --- 1. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR (Status Only) ---
with st.sidebar:
    st.title("🏦 Analyst Panel")
    st.success("✅ System Online")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.rerun()

# --- 3. AGENT INITIALIZATION ---
if "agent" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    st.session_state.memory = MemorySaver()
    st.session_state.agent = create_agent(model, [], checkpointer=st.session_state.memory)

# --- 4. MAIN DISPLAY AREA ---
st.title("Financial Analysis Dashboard")

# Create a container for messages so they don't push the form too far down
chat_placeholder = st.container()

with chat_placeholder:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# --- 5. MAIN AREA FORM (At the bottom) ---
# We use a form here to ensure the "Send" button is always visible and functional
with st.form(key="main_chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your financial question:", placeholder="e.g. Analyze NVDA stock...")
    submit_button = st.form_submit_button(label="🚀 Send to AI")

# --- 6. LOGIC ---
if submit_button and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Immediately show user message
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": "main_session"}}
            response = st.session_state.agent.invoke({"messages": [("human", user_query)]}, config)
            ans = response["messages"][-1].content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
    
    # Rerun to sync state
    st.rerun()