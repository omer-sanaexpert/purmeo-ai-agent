# --- Purmeo(DE) tools: thin wrappers so they’re separate from SanaExpert tools ---
import os
from dotenv import load_dotenv
from fastapi import logger
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from tools import escalate_to_human_germany, get_order_information_by_email, get_order_information_by_orderid, get_product_information

PURMEO_BRAND = "purmeo"
PURMEO_COUNTRY = "DE"

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-pinecone-purmeo"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)

# Load the multilingual-e5-small model
embedding_model = None


embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")


@tool
def query_knowledgebase_sanaexpert(q: str) -> str:
    """Query the SanaExpert knowledge base for product information, return policies, shipment policies, and general information.

    Args:
        q (str): The query string to search in the knowledge base.

    Returns:
        str: A concatenated string of the top 5 matching results from the knowledge base.
    """
    print("query_knowledgebase_sanaexpert")
    query_embedding = embedding_model.encode([q])[0].tolist()
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    return "\n\n".join([match.metadata["text"] for match in results.matches])




@tool
def purmeo_get_order_by_email(email: str) -> dict:
    """Purmeo-DE: Lookup order(s) by customer email"""
    # Reuse your internal function but keep brand separation if needed:
    data = get_order_information_by_email(email)
    return {"brand": PURMEO_BRAND, "country": PURMEO_COUNTRY, "data": data}

@tool
def purmeo_get_order_by_id(order_id: str) -> dict:
    """Purmeo-DE: Lookup order by order id"""
    data = get_order_information_by_orderid(order_id)
    return {"brand": PURMEO_BRAND, "country": PURMEO_COUNTRY, "data": data}

@tool
def purmeo_get_products(q: str = "") -> dict:
    """Purmeo-DE: Get product info / list (optionally filtered by q)"""
    data = get_product_information(q)
    return {"brand": PURMEO_BRAND, "country": PURMEO_COUNTRY, "data": data}

@tool
def purmeo_query_kb(q: str) -> dict:
    """Purmeo-DE: KB / FAQs (separate logical tool even if powered by same store)"""
    txt = query_knowledgebase_sanaexpert(q)  # swap to purmeo KB when available
    return {"brand": PURMEO_BRAND, "country": PURMEO_COUNTRY, "text": txt}

