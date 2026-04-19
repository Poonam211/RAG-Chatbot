from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


def create_rag_pipeline():
    # Load data
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    print("Loaded documents:", documents)  # DEBUG

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    print("Split docs:", len(docs))  # DEBUG

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chroma_db"
    )

    # Retriever
    retriever = vectorstore.as_retriever()

    # LLM
    llm = Ollama(model="phi")

    return retriever, llm