import streamlit as st
import os
from rag_pipeline import create_rag_pipeline

# Title
st.markdown(
    "<h4 style='text-align: center;'>🧠 ContextAI - RAG Chatbot</h4>",
    unsafe_allow_html=True
)

# Upload file
uploaded_file = st.file_uploader("📂 Upload a .txt file", type=["txt"])

if uploaded_file:
    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    # Create pipeline
    retriever, llm = create_rag_pipeline(file_path)

    # Input query
    query = st.text_input("Ask something from your document:")

    if query:
        docs = retriever.get_relevant_documents(query)

        if docs:
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)
            st.write("🤖 Answer:", response)

        else:
            st.write("❌ No relevant information found")