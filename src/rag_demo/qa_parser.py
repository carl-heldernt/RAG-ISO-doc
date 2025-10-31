import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if not azure_openai_api_key or not azure_openai_embedding_deployment:
    raise ValueError("❌ Missing Azure OpenAI environment variables in .env")

def build_qa_vectorstore(qa_file: str, persist_dir: str = "chroma_iso10816_qa"):
    loader = TextLoader(qa_file, encoding="utf-8")
    docs = loader.load()

    # We assume each Q&A block is short enough, so chunk size is generous
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_openai_api_key,
        azure_deployment=azure_openai_embedding_deployment,
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Stored Q&A chunks in {persist_dir}, total {len(chunks)} chunks")

if __name__ == "__main__":
    qa_file = "res/iso10816-3_qa_bilingual.md"
    build_qa_vectorstore(qa_file)
