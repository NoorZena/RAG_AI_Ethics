import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

# ✅ Retrieve API key from Streamlit secrets
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# ✅ Explicitly set the embedding model
Settings.llm = MistralAI(api_key=MISTRAL_API_KEY)
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=MISTRAL_API_KEY)

# Streamlit UI
st.title("📚 AI Ethics RAG Agent")
st.write("This app retrieves and summarizes information from an AI ethics research paper.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load document
    documents = SimpleDirectoryReader(input_files=["uploaded.pdf"]).load_data()

    # Split document
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    # ✅ Use explicit embedding settings
    summary_index = SummaryIndex(nodes[:10])
    vector_index = VectorStoreIndex(nodes[:10])  # No more default model errors

    # Query Engines
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    vector_query_engine = vector_index.as_query_engine()

    # User input for queries
    query = st.text_input("Ask a question about the paper:")
    if query:
        if "summarize" in query.lower():
            response = summary_query_engine.query(query)
        else:
            response = vector_query_engine.query(query)

        st.write("### Response:")
        st.write(response)
