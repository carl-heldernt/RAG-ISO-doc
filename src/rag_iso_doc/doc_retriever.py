import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if not azure_openai_api_key or not azure_openai_embedding_deployment:
    raise ValueError("❌ Missing Azure OpenAI env vars in .env")


def format_with_citation(results):
    """Format retrieved chunks with ISO-style citation (source + page)."""
    formatted = []
    for r in results:
        citation = f"{r.metadata.get('source', 'ISO 10816-3:2009')}"
        if r.metadata.get("page"):
            citation += f", Page {r.metadata['page']}"
        formatted.append(f"[{citation}]\n{r.page_content[:400]}...")
    return "\n\n".join(formatted)


if __name__ == "__main__":
    persist_dir = "chroma_iso10816"

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_openai_api_key,
        azure_deployment=azure_openai_embedding_deployment,
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    query = "Base on ISO 10816-3:2009, what machines are excluded?"
    results = retriever.invoke(query)

    print(f"\n🔎 Query: {query}\n")
    print(format_with_citation(results))
