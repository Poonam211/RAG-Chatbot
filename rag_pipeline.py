from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import uuid


def create_rag_pipeline(file_path):
    # Load uploaded file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 🔥 Unique DB path (fix for permission error)
    db_path = f"chroma_db_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=db_path
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # LLM
    llm = Ollama(model="phi")

    return retriever, llm