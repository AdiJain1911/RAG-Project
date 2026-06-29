import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Dynamic RAG Chatbot", page_icon="📄")
st.title("Upload & Chat with Documents 📄")
st.caption("Upload a text file, let it process, and start asking questions—100% locally.")

# --- 1. Load AI Models Efficiently ---
# @st.cache_resource prevents Streamlit from reloading the model on every click,
# which fixes the "client closed" threading error!
@st.cache_resource
def load_models():
    # local_files_only=True stops it from pinging the internet and crashing
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
    # Load Ollama here as well so it doesn't re-initialize on every chat message
    llm = ChatOllama(model="gemma4:latest", temperature=0)
    
    return embeddings, llm

# Load them immediately when the app starts
embedding_model, model = load_models()


# --- 2. Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# --- 3. File Upload Interface (Sidebar) ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    
    if uploaded_file and not st.session_state.file_processed:
        with st.spinner("Processing document..."):
            file_content = uploaded_file.read().decode("utf-8")
            
            doc = Document(
                page_content=file_content, 
                metadata={"source": uploaded_file.name}
            )
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents([doc])
            
            # Create an IN-MEMORY vector store using the cached embedding model
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            st.session_state.vectorstore = vectorstore
            st.session_state.file_processed = True
            
        st.success("✅ File processed and ready for chat!")
    
    if st.session_state.file_processed:
        if st.button("Upload a different file"):
            st.session_state.vectorstore = None
            st.session_state.file_processed = False
            st.session_state.chat_history = []
            st.session_state.ui_messages = []
            st.rerun()

# --- 4. Chat Interface (Main Area) ---
if st.session_state.file_processed:
    st.header(f"Chatting with: {uploaded_file.name}")
    
    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_question := st.chat_input("Ask a question about the uploaded document..."):
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        st.session_state.ui_messages.append({"role": "user", "content": user_question})
        
        with st.spinner("Searching document & thinking..."):
            
            if st.session_state.chat_history:
                context_messages = [
                    SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
                ] + st.session_state.chat_history + [
                    HumanMessage(content=f"New question: {user_question}")
                ]
                
                result = model.invoke(context_messages)
                search_question = result.content.strip()
            else:
                search_question = user_question
                
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(search_question)
            
            docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])
            combined_input = f"""Based on the following documents, please answer this question: {user_question}

            Documents:
            {docs_text}

            Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
            """
            
            final_messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
            ] + st.session_state.chat_history + [
                HumanMessage(content=combined_input)
            ]
            
            response = model.invoke(final_messages)
            answer = response.content
            
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=answer))

        with st.chat_message("assistant"):
            st.markdown(answer)
            
            with st.expander("Show Retrieved Context"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Chunk {i}:** {doc.page_content[:300]}...")

        st.session_state.ui_messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a .txt file in the sidebar to start chatting.")