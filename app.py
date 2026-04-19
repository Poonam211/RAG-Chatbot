import streamlit as st
from rag_pipeline import create_rag_pipeline

st.title("🧠 ContextAI RAG-Chatbot")

retriever, llm = create_rag_pipeline()

query = st.text_input("Ask something:")

if query:
    docs = retriever.get_relevant_documents(query)

    st.write("📄 Retrieved Docs:", docs)  # DEBUG

    if docs:
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer based on this:\n{context}\n\nQuestion: {query}"

        response = llm.invoke(prompt)
        st.write("🤖 Answer:", response)
    else:
        st.write("❌ No relevant documents found")