# tools.py
import os
from langchain_anthropic import ChatAnthropic
import requests
import json
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

import hashlib
import hmac
from anthropic import Anthropic
import logging
import uuid
from zendesklib import ZendeskTicketManager
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

import re
from typing import Dict, Any, List, Optional

# Initialize global variables if needed
manager = ZendeskTicketManager()
user_conversations = {}
# In-memory storage for request and ticket IDs
requests_and_tickets = {}
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTA_NEW")
INSTAGRAM_ACCESS_TOKEN_IT= os.getenv("INSTA_IT")
INSTAGRAM_ACCESS_TOKEN_DE= os.getenv("INSTA_DE")
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,  # This is the default and can be omitted
)

def get_thread_id_from_ticket_id(requests_and_tickets, target_ticket_id):
    for thread_id, info in requests_and_tickets.items():
        print(thread_id, info)
        if info["ticket_id"] == int(target_ticket_id):
            return thread_id
        else:
            print(type(target_ticket_id))
    return None  # If not found

# Endpoint URL
url = os.environ.get("SCL_URL")
username = os.environ.get("SCL_USERNAME")
password = os.environ.get("SCL_PASSWORD")
shipping_url = os.environ.get("SHIPMENT_TRACKING_URL")
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")

api_cache = {
    "get_order_information_by_email": {},
    "get_order_information": {},
    "get_voucher_information": {"data": None, "timestamp": None},
    "get_product_information": {"data": None, "timestamp": None},
    "purmeo_get_product_information": {"data": None, "timestamp": None},
    "purmeo_get_order_information": {},
}

CACHE_EXPIRY_HOURS = 24  # Cache expiration

def is_cache_valid(timestamp):
    from datetime import datetime, timedelta
    return timestamp and datetime.now() - timestamp < timedelta(hours=CACHE_EXPIRY_HOURS)
# Question rewriter
websystem = """You are a question re-writer that converts an input question to a better version optimized for web search."""
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", websystem),
    ("human", "Here is the initial question:\n\n{question}\nFormulate an improved question."),
])
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1)

#llm = ChatGroq(model="llama3-70b-8192", temperature=1)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# Web search tool
web_search_tool = TavilySearchResults(k=1, search_engine="google")

SHOPIFY_DOMAIN = os.getenv("SHOPIFY_DOMAIN", "purmeo.de")  # e.g. "sanaexpert.es"
STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_TOKEN", "")
API_VERSION = os.getenv("SHOPIFY_STOREFRONT_API_VERSION", "2024-07")  # adjust if needed

GRAPHQL_ENDPOINT = f"https://{SHOPIFY_DOMAIN}/api/{API_VERSION}/graphql.json"


SHOPIFY_ADMIN_TOKEN = os.getenv("SHOPIFY_ADMIN_TOKEN", "")
SHOPIFY_ADMIN_API_VERSION = os.getenv("SHOPIFY_ADMIN_API_VERSION", "2024-07")
ADMIN_BASE = f"https://{SHOPIFY_DOMAIN}/admin/api/{SHOPIFY_ADMIN_API_VERSION}"

# Expect these to exist in your module:
# api_cache = {"purmeo_get_product_information": {"data": None, "timestamp": datetime.min}}
# def is_cache_valid(ts: datetime) -> bool: ...

# Add/replace the GraphQL with images on product + variant
PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 100, after: $cursor, sortKey: TITLE) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        title
        handle
        onlineStoreUrl
        images(first: 1) {
          edges {
            node {
              url
              altText
            }
          }
        }
        variants(first: 100) {
          edges {
            node {
              id
              title
              sku
              availableForSale
              price { amount currencyCode }
              compareAtPrice { amount currencyCode }
              image {
                url
                altText
              }
            }
          }
        }
      }
    }
  }
}
"""

def _get_product_featured_image(prod: dict) -> dict:
    img_edges = ((prod.get("images") or {}).get("edges")) or []
    if img_edges:
        node = img_edges[0]["node"] or {}
        return {"url": node.get("url") or "", "alt": (node.get("altText") or "").strip()}
    return {"url": "", "alt": ""}

def _image_from_node(img_node: dict) -> dict:
    if not img_node:
        return {"url": "", "alt": ""}
    return {"url": img_node.get("url") or "", "alt": (img_node.get("altText") or "").strip()}


def _post_graphql(query: str, variables: Optional[dict] = None) -> dict:
  headers = {
      "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
      "Content-Type": "application/json",
      "Accept": "application/json",
  }
  resp = requests.post(GRAPHQL_ENDPOINT, json={"query": query, "variables": variables or {}}, headers=headers, timeout=30)
  resp.raise_for_status()
  data = resp.json()
  if "errors" in data:
      raise RuntimeError(f"Shopify GraphQL errors: {data['errors']}")
  return data["data"]

def _fetch_all_products() -> List[dict]:
  products: List[dict] = []
  cursor = None
  while True:
    data = _post_graphql(PRODUCTS_QUERY, {"cursor": cursor})
    block = data["products"]
    for edge in block["edges"]:
      products.append(edge["node"])
    if not block["pageInfo"]["hasNextPage"]:
      break
    cursor = block["pageInfo"]["endCursor"]
  return products

def _variant_numeric_id(variant_gid: str) -> str:
  # gid format: gid://shopify/ProductVariant/39828834484401
  m = re.search(r"/(\d+)$", variant_gid)
  return m.group(1) if m else variant_gid

def _product_url(domain: str, handle: str, variant_gid: str, product_online_url: Optional[str]) -> str:
  variant_id_num = _variant_numeric_id(variant_gid)
  if product_online_url:
    # Ensure we append variant param correctly
    sep = "&" if "?" in product_online_url else "?"
    return f"{product_online_url}{sep}variant={variant_id_num}"
  # Fallback: construct from handle
  return f"https://{domain}/products/{handle}?variant={variant_id_num}"

def _money_amount(m: Optional[dict]) -> str:
  if not m or "amount" not in m or m["amount"] is None:
    return ""
  # Normalize to string with 2 decimals (Shopify already supplies strings)
  try:
    return f"{float(m['amount']):.2f}"
  except Exception:
    return str(m["amount"])

@tool
def purmeo_get_product_information() -> Dict[str, Any]:
    """Retrieve current pricing, urls, and images for products."""
    print("purmeo_get_product_pricing")

    if is_cache_valid(api_cache["purmeo_get_product_information"]["timestamp"]):
        print("Returning cached product info")
        return api_cache["purmeo_get_product_information"]["data"]

    if not STOREFRONT_TOKEN:
        raise RuntimeError("Missing SHOPIFY_STOREFRONT_TOKEN environment variable for Storefront API access.")

    products = _fetch_all_products()
    result: Dict[str, Any] = {}

    for p in products:
        title = (p.get("title") or "").strip()
        handle = p.get("handle")
        online_url = p.get("onlineStoreUrl")
        variants = (p.get("variants") or {}).get("edges", [])
        multi_variant = len(variants) > 1

        product_img = _get_product_featured_image(p)

        for v_edge in variants:
            v = v_edge["node"]
            v_title = v.get("title") or ""
            key = title if not multi_variant or v_title.lower() in ("default title", "default") else f"{title} - {v_title}"

            variant_img = _image_from_node(v.get("image"))

            # Prefer variant image, fallback to product image
            primary_img = variant_img if variant_img["url"] else product_img

            entry = {
                "price": _money_amount(v.get("price")),
                "compare_at_price": _money_amount(v.get("compareAtPrice")),
                "sku": (v.get("sku") or "").strip(),
                "url": _product_url(SHOPIFY_DOMAIN, handle, v["id"], online_url),
                "status": "available" if v.get("availableForSale") else "not available",
                "image_url": primary_img["url"],
                "image_alt": primary_img["alt"],
                "product_image_url": product_img["url"],
                "product_image_alt": product_img["alt"],
            }

            result[key] = entry

    api_cache["purmeo_get_product_information"] = {"data": result, "timestamp": datetime.now()}
    return result

def _admin_get(path: str, params: Optional[dict] = None) -> dict:
    if not SHOPIFY_ADMIN_TOKEN:
        raise RuntimeError(
            "Missing SHOPIFY_ADMIN_TOKEN for Shopify Admin API access. "
            "Set it to enable purmeo_get_order_information."
        )
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN,
        "Accept": "application/json",
    }
    resp = requests.get(f"{ADMIN_BASE}{path}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _normalize_order(o: dict) -> Dict[str, Any]:
    """Flatten key fields + fulfillment tracking."""
    fulfillments = o.get("fulfillments", []) or []
    tracking = []
    for f in fulfillments:
        track_numbers = f.get("tracking_numbers") or []
        track_urls = f.get("tracking_urls") or []
        # Some carriers put a single url/number; Shopify may also have tracking_info array
        items = max(len(track_numbers), len(track_urls))
        for i in range(items or 1):
            tracking.append({
                "tracking_company": f.get("tracking_company") or "",
                "tracking_number": (track_numbers[i] if i < len(track_numbers) else "") or (f.get("tracking_number") or ""),
                "tracking_url": (track_urls[i] if i < len(track_urls) else "") or (f.get("tracking_url") or ""),
                "status": f.get("shipment_status") or "",
                "created_at": f.get("created_at"),
                "updated_at": f.get("updated_at"),
            })

    return {
        "id": o.get("id"),
        "name": o.get("name"),                      # e.g. "#1001"
        "order_number": o.get("order_number"),
        "created_at": o.get("created_at"),
        "currency": o.get("currency"),
        "financial_status": o.get("financial_status"),
        "fulfillment_status": o.get("fulfillment_status"),
        "current_total_price": o.get("current_total_price"),
        "subtotal_price": o.get("subtotal_price"),
        "total_discounts": o.get("total_discounts"),
        "customer_email": (o.get("email") or (o.get("customer") or {}).get("email") or ""),
        "shipping_address": o.get("shipping_address") or {},
        "billing_address": o.get("billing_address") or {},
        "line_items": [
            {
                "title": li.get("title"),
                "variant_title": li.get("variant_title"),
                "sku": li.get("sku"),
                "quantity": li.get("quantity"),
                "price": li.get("price"),
                "fulfillment_status": li.get("fulfillment_status"),
            }
            for li in (o.get("line_items") or [])
        ],
        "fulfillments": tracking,                   # flattened tracking entries
        "raw": o,                                   # keep full payload for debugging if needed
    }

@tool
def purmeo_get_order_information(order_id: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve Purmeo order details (Shopify Admin API) by order_id (Shopify order name, e.g. 1001 or #1001)
    OR by customer email (returns the most recent order for that email).

    Args:
        order_id (Optional[str]): Shopify order "name" (with or without leading '#').
        email (Optional[str]): Customer email address.

    Returns:
        Dict[str, Any]: Normalized order details incl. line items and tracking info.

    Notes:
        - Requires SHOPIFY_ADMIN_TOKEN env var.
        - Uses cache with 24h expiry.
    """
    print("purmeo_get_order_information")
    if not (order_id or email):
        raise ValueError("Provide either order_id or email.")

    # Build cache key
    cache_key = f"id::{order_id.strip()}" if order_id else f"email::{email.strip().lower()}"
    bucket = api_cache.setdefault("purmeo_get_order_information", {})

    # Serve from cache if valid
    cached = bucket.get(cache_key)
    if cached and is_cache_valid(cached.get("timestamp")):
        print("Returning cached purmeo order info")
        return cached["data"]

    params = {"status": "any", "limit": 1, "order": "created_at desc"}

    try:
        if order_id:
            # Shopify stores "name" like "#1001"; accept "1001" or "#1001"
            order_name = order_id.strip()
            if not order_name.startswith("#"):
                order_name = f"#{order_name}"
            # Search by order name
            res = _admin_get("/orders.json", {**params, "name": order_name})
            orders = res.get("orders") or []

        else:
            # Lookup by email, newest order first
            res = _admin_get("/orders.json", {**params, "email": email.strip().lower()})
            orders = res.get("orders") or []

        if not orders:
            result = {"ok": False, "message": "No matching order found.", "query": {"order_id": order_id, "email": email}}
            bucket[cache_key] = {"data": result, "timestamp": datetime.now()}
            return result

        order = orders[0]
        normalized = _normalize_order(order)
        result = {"ok": True, "order": normalized}

        # Cache and return
        bucket[cache_key] = {"data": result, "timestamp": datetime.now()}
        return result

    except requests.HTTPError as e:
        try:
            payload = e.response.json()
        except Exception:
            payload = {"error": str(e)}
        result = {"ok": False, "message": "Shopify Admin API error", "details": payload}
        bucket[cache_key] = {"data": result, "timestamp": datetime.now()}
        return result
    except Exception as e:
        result = {"ok": False, "message": f"Unexpected error: {str(e)}"}
        bucket[cache_key] = {"data": result, "timestamp": datetime.now()}
        return result


@tool
def web_search(query: str) -> str:
    """
    Perform a web search based on the given query.

    Args:
        query (str): The query for the web search.

    Returns:
        str: A string containing the search results.
    """
    print("web search")
    rewritten_query = question_rewriter.invoke({"question": query})
    print(rewritten_query)
    
    # Perform web search
    docs = web_search_tool.invoke({"query": rewritten_query}) or []

    print(docs[0]['content'] if docs else "No results found.")

    return docs[0]['content'] if docs else "No results found."

@tool
def get_order_information_by_orderid(order_id: str) -> Dict[str, Any]:
    """Retrieve order and shipping details by order ID.

    Args:
        order_id (str): The unique identifier for the order.

    Returns:
        Dict[str, Any]: A dictionary containing order details, including shipping information.
    """
    print("get_order_information_by_orderid")
    print("order id : ",order_id)

    # Check if the order data is cached and still valid
    if order_id in api_cache["get_order_information"]:
        cached_entry = api_cache["get_order_information"][order_id]
        if is_cache_valid(cached_entry["timestamp"]):
            print("Returning cached order info")
            return cached_entry["data"]

    # If not cached or expired, call API
    payload = {
        "action": "getOrderInformation",
        "order_id": order_id
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_order_information"][order_id] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

@tool
def get_order_information_by_email(email: str) -> Dict[str, Any]:
    """Retrieve order and shipping details of the last order by email.

    Args:
        email (str): The email of the customer.

    Returns:
        Dict[str, Any]: A dictionary containing order details, including shipping information.
    """
    print("get_order_information_by_email")
    print("email: ",email)

    # Check if the order data is cached and still valid
    if email in api_cache["get_order_information_by_email"]:
        cached_entry = api_cache["get_order_information_by_email"][email]
        if is_cache_valid(cached_entry["timestamp"]):
            print("Returning cached order info")
            return cached_entry["data"]

    # If not cached or expired, call API
    payload = {
        "action": "getOrderInformation",
        "mail_address": email
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_order_information_by_email"][email] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

@tool
def get_voucher_information() -> Dict[str, Any]:
    """Retrieve current voucher codes and related information.

    Returns:
        Dict[str, Any]: A dictionary containing voucher information.
    """
    print("get_voucher_information")

    if is_cache_valid(api_cache["get_voucher_information"]["timestamp"]):
        print("Returning cached voucher info")
        return api_cache["get_voucher_information"]["data"]

    # Fetch from API if cache is expired
    payload = {"action": "getCurrentShopifyVoucherCodes"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_voucher_information"] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

@tool
def get_product_information() -> Dict[str, Any]:
    """Retrieve current pricing and url for products.

    Returns:
        Dict[str, Any]: A dictionary containing product pricing , name and url information.
    """
    print("get_product_pricing")

    if is_cache_valid(api_cache["get_product_information"]["timestamp"]):
        print("Returning cached product info")
        return api_cache["get_product_information"]["data"]

    # Fetch from API if cache is expired
    payload = {"action": "getCurrentShopifyPrices"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_product_information"] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()


@tool
def escalate_to_human(name: str, email: str, thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        name (str): The name of the person requesting escalation.
        email (str): The email address of the person requesting escalation.
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("escalate_to_human", name, email)
    if not name or not email:
        return "Please provide both your name and email to escalate the ticket."
    print("thread id "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk 25793382446353
        if manager.update_user_details(requester_id, ticket_id, email, name , summary,"25793382446353",additional_tag="ai_spain_shopify"):
            # Add the LLM-generated summary as a public comment
            return f"Escalated ticket created for {name} ({email})"
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details(requester_id, ticket_id, email, name,"","25793382446353",additional_tag="ai_spain_shopify"):
            fallback_message = f"Ticket escalated for {name} ({email}). Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket created for {name} ({email})"
    
    return "Something went wrong. Please contact support@sanaexpert.com"    

@tool
def escalate_to_human_italy(name: str, email: str, thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        name (str): The name of the person requesting escalation.
        email (str): The email address of the person requesting escalation.
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("escalate_to_human", name, email)
    if not name or not email:
        return "Please provide both your name and email to escalate the ticket."
    print("thread id "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details(requester_id, ticket_id, email, name , summary, "25793178334481", additional_tag="ai_italy_shopify"):
            # Add the LLM-generated summary as a public comment
            return f"Escalated ticket created for {name} ({email})"
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details(requester_id, ticket_id, email, name,"25793178334481",additional_tag="ai_italy_shopify"):
            fallback_message = f"Ticket escalated for {name} ({email}). Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket created for {name} ({email})"
    
    return "Something went wrong. Please contact support@sanaexpert.com"    


# need to change the assignee id for germany 
@tool
def purmeo_escalate_human(name: str, email: str, thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        name (str): The name of the person requesting escalation.
        email (str): The email address of the person requesting escalation.
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("escalate_to_human", name, email)
    if not name or not email:
        return "Please provide both your name and email to escalate the ticket."
    print("thread id "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details(requester_id, ticket_id, email, name , summary, "25793382446353", additional_tag="purmeo_germany_shopify"):
            # Add the LLM-generated summary as a public comment
            return f"Escalated ticket created for {name} ({email})"
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details(requester_id, ticket_id, email, name,"25793382446353",additional_tag="purmeo_germany_shopify"):
            fallback_message = f"Ticket escalated for {name} ({email}). Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket created for {name} ({email})"
    
    return "Something went wrong. Please contact support@purmeo.de"  

@tool
def escalate_to_human_ig_spain(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details_ig(requester_id, ticket_id, summary,"25793382446353",additional_tag="ai_spain_instagram"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_ig(requester_id, ticket_id, "25793382446353"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id,additional_tag="ai_spain_instagram"):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 


@tool
def escalate_to_human_whatsapp_spain(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details_ig(requester_id, ticket_id, summary,"25793382446353",additional_tag="ai_spain_whatsapp"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_whatsapp(requester_id, ticket_id, "25793382446353"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id,additional_tag="ai_spain_instagram"):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 


@tool
def escalate_to_human_whatsapp_italy(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # 25793382446353 is assigne id for spain
        # 25793178334481 is assigne id for italy
        # Update user details in Zendesk
        if manager.update_user_details_whatsapp(requester_id, ticket_id, summary,"25793178334481",additional_tag="ai_italy_whatsapp"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_ig(requester_id, ticket_id, "25793178334481",additional_tag="ai_italy_whatsapp"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 

@tool
def escalate_to_human_ig_germany(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details_ig(requester_id, ticket_id, summary,"25793382446353",additional_tag="ai_germany_instagram"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_ig(requester_id, ticket_id, "25793382446353",additional_tag="ai_germany_instagram"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 



@tool
def escalate_to_human_email_spain(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        #print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details_email(requester_id, ticket_id, summary,"25793382446353",additional_tag="ai_spain_email"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_email(requester_id, ticket_id, "25793382446353",additional_tag="ai_spain_email"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 


@tool
def escalate_to_human_email_italy(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        #print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk to alberto italy
        if manager.update_user_details_email(requester_id, ticket_id, summary,"25793178334481",additional_tag="ai_italy_email"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_email(requester_id, ticket_id, "25793178334481",additional_tag="ai_italy_email"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 

@tool
def escalate_to_human_ig_italy(thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("thread id from insta spain "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    user_conversations[thread_id]["status"] = "escalated"
    print(user_conversations[thread_id]["status"])
    
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount_or_vouchers, sensitive_medical_question or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # 25793382446353 is assigne id for spain
        # 25793178334481 is assigne id for italy
        # Update user details in Zendesk
        if manager.update_user_details_ig(requester_id, ticket_id, summary,"25793178334481",additional_tag="ai_italy_instagram"):
            # Add the LLM-generated summary as a public comment
            user_conversations[thread_id]["status"] = "escalated"
            return f"Escalated ticket Successfully created."
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details_ig(requester_id, ticket_id, "25793178334481",additional_tag="ai_italy_instagram"):
            fallback_message = f"Ticket escalated for {ticket_id} . Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket Successfully created.)"
    
    return "Something went wrong. Please contact support@sanaexpert.com" 


def create_payload(message_text, recipient_id):
    # Escape the message_text and recipient_id using json.dumps
    payload = {
        'message': json.dumps({'text': message_text}),
        'recipient': json.dumps({'id': recipient_id})
    }
    return payload

def send_insta(recipient_id, message_text):
    url = 'https://graph.instagram.com/v21.0/me/messages'

    headers = {
        'Authorization': f'Bearer {INSTAGRAM_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }

    payload = create_payload(message_text, recipient_id)

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("✅ Message sent successfully!")
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def send_insta_de(recipient_id, message_text):
    url = 'https://graph.instagram.com/v21.0/me/messages'

    headers = {
        'Authorization': f'Bearer {INSTAGRAM_ACCESS_TOKEN_DE}',
        'Content-Type': 'application/json'
    }

    payload = create_payload(message_text, recipient_id)

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("✅ Message sent successfully!")
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def send_insta_it(recipient_id, message_text):
    url = 'https://graph.instagram.com/v21.0/me/messages'

    headers = {
        'Authorization': f'Bearer {INSTAGRAM_ACCESS_TOKEN_IT}',
        'Content-Type': 'application/json'
    }

    payload = create_payload(message_text, recipient_id)

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("✅ Message sent successfully!")
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None
import requests
import os

def get_instagram_user_from_psid(psid, access_token=None):
    """
    Get user's Instagram information from PSID with specific permission handling.
    This function is optimized for cases where the token works for sending messages
    but not for retrieving user details.
    
    Args:
        psid (str): The Page-Scoped ID of the user
        access_token (str, optional): Instagram access token. If None, will use environment variable
        
    Returns:
        dict: User information or None if the request fails
    """
    # Get access token from parameter or environment variable
    token = access_token or os.environ.get('INSTA_NEW')
    print("Token:",token)
    print(type(token))
    
    if not token:
        print("❌ No access token provided")
        return None
    
    url = f"https://graph.facebook.com/v21.0/{psid}?fields=username,name&access_token={token}"
    response = requests.get(url)
    print(response.json())
    json_response = response.json()
    
    return {
        'psid': json_response['id'],
        'instagram_id': json_response['username'],
        'name': json_response['name'],
    }
        
def get_instagram_user_from_psid_it(psid, access_token=None):
    """
    Get user's Instagram information from PSID with specific permission handling.
    This function is optimized for cases where the token works for sending messages
    but not for retrieving user details.
    
    Args:
        psid (str): The Page-Scoped ID of the user
        access_token (str, optional): Instagram access token. If None, will use environment variable
        
    Returns:
        dict: User information or None if the request fails
    """
    # Get access token from parameter or environment variable
    token = access_token or os.environ.get('INSTA_IT')
    print("Token:",token)
    print(type(token))
    
    if not token:
        print("❌ No access token provided")
        return None
    
    url = f"https://graph.facebook.com/v21.0/{psid}?fields=username,name&access_token={token}"
    response = requests.get(url)
    print(response.json())
    json_response = response.json()
    
    return {
        'psid': json_response['id'],
        'instagram_id': json_response['username'],
        'name': json_response['name'],
    }
        
