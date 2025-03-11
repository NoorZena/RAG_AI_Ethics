import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from mistralai.models import SDKError  # Import error handling for Mistral

# ✅ Retrieve API key from Streamlit secrets
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
except KeyError:
    st.error("❌ API key not found. Please set MISTRAL_API_KEY in Streamlit Secrets.")
    st.stop()

# ✅ Explicitly set the embedding model
try:
    Settings.llm = MistralAI(api_key=MISTRAL_API_KEY)
    Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=MISTRAL_API_KEY)
except SDKError as e:
    st.error(f"❌ Mistral API Error: {e}")
    st.stop()

# Streamlit UI
st.title("📚 AI Ethics RAG Agent")
st.write("This app retrieves and summarizes information from an AI ethics research paper.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load document
        documents = SimpleDirectoryReader(input_files=["uploaded.pdf"]).load_data()

        # Split document
        splitter = SentenceSplitter(chunk_size=512)
        nodes = splitter.get_nodes_from_documents(documents)

        # ✅ Use explicit embedding settings
        summary_index = SummaryIndex(nodes[:10])
        vector_index = VectorStoreIndex(nodes[:10])  

        # Query Engines
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
        vector_query_engine = vector_index.as_query_engine()

        # ✅ Multi-query Support using Session State
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User input for queries
        query = st.text_input("Ask a question about the paper:")

        if query:
            try:
                if "summarize" in query.lower():
                    response = summary_query_engine.query(query)
                else:
                    response = vector_query_engine.query(query)

                # Store query and response in session state
                st.session_state.chat_history.append((query, response.response))

                # Clear the input box for new queries
                st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ Error while processing query: {e}")

        # Display past queries and responses
        st.write("### Chat History:")
        for q, r in st.session_state.chat_history:
            with st.expander(f"🔹 {q}"):
                st.write(r)

    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
