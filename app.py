import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Black & White Minimalist Design (CSS)
st.set_page_config(page_title="LensGPT", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;900&display=swap');
    
    html, body, [class*="css"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #000000; color: #ffffff;
        border-radius: 0px; border: 2px solid #000000;
        font-weight: 900; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ffffff; color: #000000; }
    input { border-radius: 0px !important; border: 2px solid #000000 !important; }
    .stTextInput>div>div>input { color: black; }
    .sidebar .sidebar-content { background-color: #f0f0f0; border-right: 3px solid #000000; }
    </style>
    """, unsafe_allow_mode=True)

# 2. Authentication Logic
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.markdown("# LENSGPT / LOGIN")
    user = st.text_input("USERNAME")
    pw = st.text_input("PASSWORD", type="password")
    if st.button("ACCESS SYSTEM"):
        if user == "admin" and pw == "admin": # يمكنك تغييرها لاحقاً لربطها بقاعدة بيانات
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

# 3. Core AI Engine
def process_docs(files):
    raw_text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# 4. Main Application Interface
def main_app():
    # Sidebar
    st.sidebar.markdown("# LENSGPT")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("MENU", ["Workspace", "Subscription", "Settings"])
    
    if st.sidebar.button("LOGOUT"):
        st.session_state.authenticated = False
        st.rerun()

    if menu == "Workspace":
        st.markdown("# SMART WORKSPACE")
        files = st.file_uploader("UPLOAD SOURCES (PDF)", accept_multiple_files=True)
        
        if files:
            if "vectorstore" not in st.session_state:
                with st.spinner("Lens is scanning..."):
                    st.session_state.vectorstore = process_docs(files)
            
            query = st.text_input("Ask LensGPT about your documents:")
            if query:
                qa = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model="gpt-4o", openai_api_key=st.secrets["OPENAI_API_KEY"]),
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                response = qa.run(query)
                st.markdown(f"### Result:\n{response}")

    elif menu == "Subscription":
        st.markdown("# PRICING PLANS")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### BASIC\n$0/mo\n- 2 Files\n- GPT-3.5")
            st.button("Current Plan", disabled=True)
        with col2:
            st.markdown("### PRO\n$20/mo\n- Unlimited Files\n- GPT-4o Power")
            # ضع هنا رابط Stripe الخاص بك
            st.link_button("UPGRADE NOW", "https://buy.stripe.com/test_links")

# 5. App Router
if not st.session_state.authenticated:
    login()
else:
    main_app()
