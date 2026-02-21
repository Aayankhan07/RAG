import os
import sys

PROJECT_PACKAGES_DIR = os.path.join(os.path.dirname(__file__), ".packages")

if os.path.isdir(PROJECT_PACKAGES_DIR) and PROJECT_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PACKAGES_DIR)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    print("Starting ingestion pipeline...")

    if not os.path.isdir(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' does not exist. Please create it and add PDFs.")
        sys.exit(1)

    print(f"Loading PDFs from '{DATA_DIR}'...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print(f"No PDF documents found in '{DATA_DIR}'. Add PDFs and run again.")
        sys.exit(0)

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks.")

    print(f"Initializing embeddings model '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Creating Chroma database in '{CHROMA_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("Persisting Chroma database to disk...")
    vectorstore.persist()

    print(f"Database successfully created and stored in '{CHROMA_DIR}'.")


if __name__ == "__main__":
    main()
