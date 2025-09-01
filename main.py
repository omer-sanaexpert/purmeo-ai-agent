import re
from fastapi import FastAPI, Depends, HTTPException, Body, Header, Query, Request
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated, Optional, TypedDict, List  # Import for State definition
from langgraph.graph.message import AnyMessage, add_messages  # Import for State definition
import uuid
import os
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool  # Import the @tool decorator
import requests
import json
from langchain_core.output_parsers import StrOutputParser
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends
from datetime import datetime, timedelta
from anthropic import Anthropic, Client
from dotenv import load_dotenv
from purmeo_tools import purmeo_query_kb
from tools import purmeo_get_product_information, purmeo_escalate_human, purmeo_get_order_information,escalate_to_human_email_italy, escalate_to_human_ig_germany, escalate_to_human_email_spain, escalate_to_human_ig_italy,escalate_to_human_whatsapp_spain,escalate_to_human_whatsapp_italy, get_instagram_user_from_psid, get_thread_id_from_ticket_id,send_insta,escalate_to_human,escalate_to_human_italy,escalate_to_human_ig_spain, get_order_information_by_email, get_order_information_by_orderid, get_product_information, send_insta_it, send_insta_de, user_conversations, requests_and_tickets,llm
from zendesklib import ZendeskTicketManager
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import logging
import hmac
import hashlib
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta


from utils import strip_html, count_tokens
from prompts import primary_assistant_prompt_purmeo_de,primary_assistant_prompt, primary_assistant_prompt_italy, primary_assistant_prompt_germany, primary_assistant_prompt_ig_spain,primary_assistant_prompt_ig_germany, primary_assistant_prompt_ig_italy, primary_assistant_prompt_email_spain,primary_assistant_prompt_email_italy
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_llm_cache(InMemoryCache())

load_dotenv() 

NUDGE_DELAY_SECONDS = 90

nudge_tasks: dict[str, asyncio.Task] = {}
pending_nudges: defaultdict[str, list] = defaultdict(list)
last_activity: dict[str, datetime] = {}

#env variables:
AUTH_USERNAME = os.getenv("API_USERNAME", "sanaexpert")  # Set these in your .env file
AUTH_PASSWORD = os.getenv("API_PASSWORD", "San@Xpert997755")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Endpoint URL
url = os.environ.get("SCL_URL")
username = os.environ.get("SCL_USERNAME")
password = os.environ.get("SCL_PASSWORD")
shipping_url = os.environ.get("SHIPMENT_TRACKING_URL")
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Configuration for Instagram Graph API
class InstagramConfig:
    ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    ACCESS_TOKEN_IT = os.environ.get("INSTA_IT")
    ACCESS_TOKEN_DE = os.environ.get("INSTA_DE")
    PAGE_ID = os.environ.get("INSTAGRAM_PAGE_ID")
    PAGE_ID_IT = os.environ.get("INSTAGRAM_PAGE_ID_IT")
    PAGE_ID_DE = os.environ.get("INSTAGRAM_PAGE_ID_DE")
    WEBHOOK_VERIFY_TOKEN = os.environ.get("CUSTOM_VERIFY_TOKEN")
    APP_SECRET = os.environ.get("FACEBOOK_APP_SECRET")
    APP_SECRET_IT = os.environ.get("FACEBOOK_APP_SECRET_IT")
    APP_SECRET_DE = os.environ.get("FACEBOOK_APP_SECRET_DE")
    API_BASE_URL = "https://graph.facebook.com/v19.0"


#modify this
class WhatsappConfig:
    ACCESS_TOKEN = os.environ.get("WHATSAPP_ES_TOKEN")
    ACCESS_TOKEN_IT = os.environ.get("WHATSAPP_IT_TOKEN")
    PAGE_ID = os.environ.get("WHATSAPP_ES_PAGE_ID")
    PAGE_ID_IT = os.environ.get("WHATSAPP_ES_PAGE_ID")
    WEBHOOK_VERIFY_TOKEN = os.environ.get("CUSTOM_VERIFY_TOKEN")
    APP_SECRET = os.environ.get("WHATSAPP_ES_APP_SECRET")
    APP_SECRET_IT = os.environ.get("WHATSAPP_IT_APP_SECRET")
    API_BASE_URL = "https://graph.facebook.com/v22.0"


manager = ZendeskTicketManager()
# Cache dictionary to store API responses
api_cache = {
    "get_order_information_by_email": {},
    "get_order_information": {},
    "get_voucher_information": {"data": None, "timestamp": None},
    "get_product_information": {"data": None, "timestamp": None}
}

CACHE_EXPIRY_HOURS = 24  # Set cache expiration time to 24 hours

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTA_NEW")
INSTAGRAM_ACCESS_TOKEN_IT = os.getenv("INSTA_IT")

def is_cache_valid(timestamp):
    """Check if the cached data is still valid."""
    return timestamp and datetime.now() - timestamp < timedelta(hours=CACHE_EXPIRY_HOURS)


# Security setup
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != AUTH_USERNAME or credentials.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials

client = Anthropic(
    api_key=ANTHROPIC_API_KEY,  # This is the default and can be omitted
)


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="your-secret-keykshdfbdsjkfh")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
print("welcome")

# In-memory storage for user conversations




# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-pinecone-purmeo"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)



# Load the multilingual-e5-small model
embedding_model = None


embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")



LOG_FILE = "chat_logs.json"

def save_log(user_id, user_message, assistant_response):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "user_message": user_message,
        "assistant_response": assistant_response
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)



def _last_user_text(msgs) -> str:
    for m in reversed(msgs):
        # HumanMessage or tuple ("user", text)
        t = getattr(m, "type", None)
        if t == "human" or ("HumanMessage" in str(type(m))):
            c = m.content
            if isinstance(c, str): return c
            if isinstance(c, list): return " ".join(str(p) for p in c)
        if isinstance(m, tuple) and len(m) == 2 and m[0] == "user":
            return str(m[1])
    return ""

def _extract_text(m) -> str:
    c = getattr(m, "content", None)
    if isinstance(c, str): return c
    if isinstance(c, list): return " ".join(str(p) for p in c)
    if isinstance(m, tuple) and len(m) == 2 and m[0] == "user":
        return str(m[1])
    return ""

def _last_human_idx(msgs) -> int:
    for i in range(len(msgs)-1, -1, -1):
        m = msgs[i]
        if isinstance(m, HumanMessage): return i
        if isinstance(m, tuple) and m[0] == "user": return i
        if getattr(m, "type", None) == "human": return i
    return -1

def _tool_ran_since(msgs, since_idx: int, tool_name: str) -> bool:
    # LangChain ToolNode returns ToolMessage(name=<tool_name>)
    for m in msgs[since_idx+1:]:
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == tool_name:
            return True
    return False

def is_product_intent(text: str) -> bool:
    if not text: return False
    t = text.lower()
    kws = ["produkt","preis","kosten","zutaten","inhaltsstoffe","dosierung",
           "verträglichkeit","nebenwirkung","lieferbar","verfügbarkeit",
           "vorrätig","bild","link","kapseln","vitamin","abo","probierpaket","purmeo"]
    return any(k in t for k in kws)


#instagram start


class MessageEntry(BaseModel):
    """
    Model for Instagram message webhook payload
    """
    id: str
    time: int
    messaging: Optional[List[Dict]] = None
    changes: Optional[List[Dict]] = None

class WebhookPayload(BaseModel):
    """
    Webhook payload structure
    """
    object: str
    entry: List[MessageEntry]

class MessageService:
    @staticmethod
    def send_reply_message(recipient_id: str, message: str):
        """
        Send a reply message to a specific Instagram user
        """
        url = f"{InstagramConfig.API_BASE_URL}/{InstagramConfig.PAGE_ID}/messages"
        
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message},
            "messaging_type": "RESPONSE",
            "access_token": InstagramConfig.ACCESS_TOKEN
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to send Instagram message: {str(e)}"
            )

class WhatsappMessageService:
    @staticmethod
    def send_reply_message(recipient_id: str, message: str):
        """
        Send a reply message to a specific WhatsApp user
        """
        url = f"{WhatsappConfig.API_BASE_URL}/{WhatsappConfig.PAGE_ID}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": message}
        }
        headers = {
            "Authorization": f"Bearer {WhatsappConfig.ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url, 
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to send WhatsApp message: {str(e)}"
            )
    @staticmethod
    def send_reply_message_it(recipient_id: str, message: str):
        """
        Send a reply message to a specific WhatsApp user
        """
        logger.info(f"Reply message to Whatsapp IT:")
        url = f"{WhatsappConfig.API_BASE_URL}/{WhatsappConfig.PAGE_ID_IT}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": message}
        }
        headers = {
            "Authorization": f"Bearer {WhatsappConfig.ACCESS_TOKEN_IT}",
            "Content-Type": "application/json"
        }
        logger.info(f"Reply message to Whatsapp IT is: {message}")
        try:
            response = requests.post(
                url, 
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"After Reply message to Whatsapp IT: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to send IT message: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to send WhatsApp IT message: {str(e)}"
            )


# Define the State class
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    thread_id: str | None
    shipping_url: str | None
    name : str | None
    email : str | None
    order_id: str | None
    postal_code: str | None





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


def handle_tool_error(state) -> dict:
    print("handle_tool_error" , state.get("error"))
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], 
        exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

# Define the Assistant class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            order_id = configuration.get("order_id", None)
            name = configuration.get("name", None)
            email = configuration.get("email", None)
            thread_id = configuration.get("thread_id", None)  # Get thread_id from config
            shipping_url = configuration.get("shipping_url", None)
            page_url = configuration.get("page_url", None)
            
            state = {
                **state, 
                "order_id": order_id,
                "thread_id": thread_id,  # Add thread_id to state
                "shipping_url": shipping_url,
                "name": name,
                "email": email,
                "page_url": page_url
            }
            #print("Thread ID from assistant: ", thread_id)
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages, "thread_id": thread_id, "shipping_url": shipping_url , "name": name, "email": email, "page_url": page_url}
            else:
                break
        return {"messages": result}


class AssistantPurmeoDE(Assistant):
    def __call__(self, state: State, config: RunnableConfig):
        # keep your state enrichment
        configuration = config.get("configurable", {}) or {}
        state = {
            **state,
            "order_id": configuration.get("order_id"),
            "thread_id": configuration.get("thread_id"),
            "shipping_url": configuration.get("shipping_url"),
            "name": configuration.get("name"),
            "email": configuration.get("email"),
            "page_url": configuration.get("page_url"),
        }

        msgs = state["messages"]
        if not msgs:
            return super().__call__(state, config)

        # 🚫 Never force after a ToolMessage; let the model compose
        if isinstance(msgs[-1], ToolMessage):
            return super().__call__(state, config)

        # Only consider forcing on the latest Human turn
        hi = _last_human_idx(msgs)
        if hi == -1:  # no human found → normal flow
            return super().__call__(state, config)

        user_text = _extract_text(msgs[hi])

        # If product intent AND tool has NOT already returned since that human, force once
        if is_product_intent(user_text) and not _tool_ran_since(msgs, hi, "purmeo_get_product_information"):
            tool_call_id = str(uuid.uuid4())
            forced = AIMessage(
                content="",
                tool_calls=[{
                    "name": "purmeo_get_product_information",
                    "args": {"query": user_text, "lang": "de"},
                    "id": tool_call_id,
                }],
            )
            return {"messages": forced}

        # Otherwise, normal behavior
        return super().__call__(state, config)



# --- Bind Purmeo-DE toolset ---
part_1_tools_purmeo_de = [
    purmeo_get_order_information,
    purmeo_get_product_information,
    purmeo_query_kb,
    purmeo_escalate_human,
]



# Tools list
part_1_tools = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human]
# Tools list
part_1_tools_italy = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_italy]
part_1_tools_germany = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert]

part_1_tools_ig_spain = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_ig_spain]

part_1_tools_ig_germany = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_ig_germany]
part_1_tools_ig_italy = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_ig_italy]

part_1_tools_whatsapp_spain = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_whatsapp_spain]
part_1_tools_whatsapp_italy = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_whatsapp_italy]


part_1_tools_email_spain = [get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_email_spain]

part_1_tools_email_italy = [get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human_email_italy]


part_1_assistant_runnable_purmeo_de = primary_assistant_prompt_purmeo_de | llm.bind_tools(part_1_tools_purmeo_de)



# Build assistant runnable
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
part_1_assistant_runnable_italy = primary_assistant_prompt_italy | llm.bind_tools(part_1_tools_italy)
part_1_assistant_runnable_germany = primary_assistant_prompt_germany | llm.bind_tools(part_1_tools_germany)
part_1_assistant_runnable_ig_spain = primary_assistant_prompt_ig_spain | llm.bind_tools(part_1_tools_ig_spain)
part_1_assistant_runnable_ig_germany = primary_assistant_prompt_ig_germany | llm.bind_tools(part_1_tools_ig_germany)
part_1_assistant_runnable_ig_italy = primary_assistant_prompt_ig_italy | llm.bind_tools(part_1_tools_ig_italy)

part_1_assistant_runnable_whatsapp_spain = primary_assistant_prompt_ig_spain | llm.bind_tools(part_1_tools_whatsapp_spain)
part_1_assistant_runnable_whatsapp_italy = primary_assistant_prompt_ig_italy | llm.bind_tools(part_1_tools_whatsapp_italy)

part_1_assistant_runnable_email_spain = primary_assistant_prompt_email_spain | llm.bind_tools(part_1_tools_email_spain)
part_1_assistant_runnable_email_italy = primary_assistant_prompt_email_italy | llm.bind_tools(part_1_tools_email_italy)



builder = StateGraph(State)
builder.add_node("assistant", AssistantPurmeoDE(part_1_assistant_runnable_purmeo_de))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_purmeo_de))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_purmeo_de = builder.compile(checkpointer=memory)



# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)



# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_italy))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_italy))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_italy = builder.compile(checkpointer=memory)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_germany))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_germany))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_germany = builder.compile(checkpointer=memory)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_ig_spain))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_ig_spain))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_ig_spain = builder.compile(checkpointer=memory)


# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_ig_germany))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_ig_germany))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_ig_germany = builder.compile(checkpointer=memory)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_ig_italy))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_ig_italy))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_ig_italy = builder.compile(checkpointer=memory)


# Build graph email spain
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_email_spain))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_email_spain))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_email_spain = builder.compile(checkpointer=memory)


# Build graph email italy
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable_email_italy))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools_email_italy))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph_email_italy = builder.compile(checkpointer=memory)

def extract_message_field(resp: Any) -> str:
    """
    Return the 'message' string from the assistant response.
    Accepts a dict or a JSON string. Falls back to a regex or the raw string.
    """
    data = resp

    # If it's a string, try JSON first
    if isinstance(resp, str):
        s = resp.strip()
        try:
            data = json.loads(s)
        except Exception:
            # Fallback: quick regex to grab "message": "..."
            m = re.search(r'"message"\s*:\s*"([^"]*)"', s)
            if m:
                return m.group(1)
            # Last resort: return the raw string (caller can strip HTML)
            return s

    # If it's already a dict
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, str):
            return msg
        # Optional: common alternates if your LLM sometimes returns other shapes
        if isinstance(data.get("content"), str):
            return data["content"]

    # If it's something else (list, None, etc.), return a safe string
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)

def call_llm_to_fix_json(raw_text: str) -> str:
    """
    Use an LLM to fix malformed JSON and return valid JSON only.
    If input cannot be parsed, the LLM will wrap it as {"text": "<original>"}.
    """
    fix_json_system = """You are a JSON fixer.
    Take possibly malformed JSON-like text and return ONLY valid JSON.
    Do not include explanations, comments, or code fences.
    If the text cannot be turned into JSON, wrap it as {"text": "<original>"}.
    """

    fix_json_prompt = ChatPromptTemplate.from_messages([
        ("system", fix_json_system),
        ("human", "Here is the input text:\n\n{input_text}\n\nReturn valid JSON only."),
    ])

    # Pick your LLM
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    fix_json_chain = fix_json_prompt | llm | StrOutputParser()

    return fix_json_chain.invoke({"input_text": raw_text})



class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for each user")
    message: str = Field(..., description="User message")
    cemail: str = Field(None, description="Email of logged in user")
    cname: str = Field(None, description="Name of logged in user")
    page_url: str = Field(None, description="URL of the page where the chat was initiated")


@app.post("/chat")
async def chat(request_data: ChatRequest, request: Request):
    user_id = request_data.user_id
    user_message = strip_html(request_data.message)
    page_url = request_data.page_url
    print("page url: "+page_url)
    
    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")
    
    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": [],
            "status": "new",
        }
    
    # the problem could be that a user couldnt be able to create multiple tickets.
    thread_id = user_conversations[user_id]["thread_id"]
    print("Thread ID from chat: ", thread_id)

    email = ""
    name = ""
    #if email is in the user message then assign user email to the email variable
    
    
    # Check if this thread already has a ticket
    if thread_id not in requests_and_tickets:
        requester_id, ticket_id = manager.create_anonymous_ticket(user_message, country="ES")
        requests_and_tickets[thread_id] = {
            "requester_id": requester_id,
            "ticket_id": ticket_id
        }
        if request_data.cemail:
            email = request_data.cemail
            name = request_data.cname
            print("Email: ", email)
            manager.update_user_information(requester_id, ticket_id, email, name)
    else:
        requester_id = requests_and_tickets[thread_id]["requester_id"]
        ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    

    # First, add the user message to the ticket
    if not manager.add_public_comment(ticket_id, user_message, requester_id):
        print("Failed to add public comment to ticket")
    else:
        print("Added public comment to ticket")
    
    
    config = {
        "configurable": {
            "order_id": "",
            "postal_code": "",
            "thread_id": thread_id,
            "email": email,
            "name": name,
            "shipping_url": shipping_url,
            "page_url": page_url
        }
    }
    
    user_conversations[user_id]["history"].append(f"You: {user_message}")
    
    # Initialize a set to track printed events
    printed_events = set()
    
    try:
        events = part_1_graph.stream(
            {"messages": [("user", (user_message))]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
    
    last_assistant_response = ""
    raw_events = list(events)
    
    for event in raw_events:
        # Print each event
        _print_event(event, printed_events)
        if "messages" in event:
            for message in event["messages"]:
                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                    content = message.content
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                    elif isinstance(content, list):
                        content = " ".join(str(part) for part in content)
                    elif isinstance(content, str):
                        last_assistant_response = content
    
    # Then add the assistant response after getting it
    if last_assistant_response:
        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
            print("Failed to add public comment by agent to ticket")
        else:
            print("Added public comment by agent to ticket")
    
    return {"response": last_assistant_response}


@app.post("/chatitaly")
async def chat(request_data: ChatRequest, request: Request):
    user_id = request_data.user_id
    user_message = strip_html(request_data.message)
    page_url = request_data.page_url
    print("page url: "+page_url)
    
    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")
    
    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": [],
            "status": "new",
        }
    
    # the problem could be that a user couldnt be able to create multiple tickets.
    thread_id = user_conversations[user_id]["thread_id"]
    print("Thread ID from chat: ", thread_id)
    email = ""
    name = ""
    
    # Check if this thread already has a ticket
    if thread_id not in requests_and_tickets:
        requester_id, ticket_id = manager.create_anonymous_ticket(user_message, country="IT")
        requests_and_tickets[thread_id] = {
            "requester_id": requester_id,
            "ticket_id": ticket_id
        }
        if request_data.cemail:
            email = request_data.cemail
            name = request_data.cname
            print("Email: ", email)
            manager.update_user_information(requester_id, ticket_id, email, name)
    else:
        requester_id = requests_and_tickets[thread_id]["requester_id"]
        ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # First, add the user message to the ticket
    if not manager.add_public_comment(ticket_id, user_message, requester_id):
        print("Failed to add public comment to ticket")
    else:
        print("Added public comment to ticket")

    
    config = {
        "configurable": {
            "order_id": "",
            "postal_code": "",
            "thread_id": thread_id,
            "email": email,
            "name": "",
            "shipping_url": shipping_url,
            "page_url": page_url
        }
    }
    
    user_conversations[user_id]["history"].append(f"You: {user_message}")
    
    # Initialize a set to track printed events
    printed_events = set()
    
    try:
        events = part_1_graph_italy.stream(
            {"messages": [("user", (user_message))]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
    
    last_assistant_response = ""
    raw_events = list(events)
    
    for event in raw_events:
        # Print each event
        _print_event(event, printed_events)
        if "messages" in event:
            for message in event["messages"]:
                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                    content = message.content
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                    elif isinstance(content, list):
                        content = " ".join(str(part) for part in content)
                    elif isinstance(content, str):
                        last_assistant_response = content
    
    # Then add the assistant response after getting it
    if last_assistant_response:
        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
            print("Failed to add public comment by agent to ticket")
        else:
            print("Added public comment by agent to ticket")
    
    return {"response": last_assistant_response}

def ensure_valid_json(text: str) -> str:
    """
    Check if `text` is valid JSON. If not, try to fix it using LLM or return the raw text.
    """
    try:
        # Try to parse JSON directly
        parsed = json.loads(text)
        # Re-dump to normalized JSON string
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        # If invalid, call your LLM repair logic
        try:
            # Example pseudo-code: use your existing LLM to repair JSON
            fixed_json = call_llm_to_fix_json(text)  
            parsed = json.loads(fixed_json)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # If still invalid, fallback to original text
            return text

@app.post("/purmeochatgermany")
async def chat(request_data: ChatRequest, request: Request):
    user_id = request_data.user_id
    user_message = strip_html(request_data.message)
    page_url = request_data.page_url
    print("page url: "+page_url)
    print("user_id: "+user_id)
    print("message: "+user_message)
    
    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")
    
    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": [],
            "status": "new",
        }
    
    # the problem could be that a user couldnt be able to create multiple tickets.
    thread_id = user_conversations[user_id]["thread_id"]
    print("Thread ID from chat: ", thread_id)
    email = ""
    name = ""
    
    # Check if this thread already has a ticket
    if thread_id not in requests_and_tickets:
        requester_id, ticket_id = manager.create_anonymous_ticket_purmeo(user_message, country="DE")
        requests_and_tickets[thread_id] = {
            "requester_id": requester_id,
            "ticket_id": ticket_id
        }
        if request_data.cemail:
            email = request_data.cemail
            name = request_data.cname
            print("Email: ", email)
            manager.update_user_information(requester_id, ticket_id, email, name)
    else:
        requester_id = requests_and_tickets[thread_id]["requester_id"]
        ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # check if the user_message json is correct. If not then use LLM to get correct JSON.


    # First, add the user message to the ticket
    if not manager.add_public_comment(ticket_id, user_message, requester_id):
        print("Failed to add public comment to ticket")
    else:
        print("Added public comment to ticket")

    
    config = {
        "configurable": {
            "order_id": "",
            "postal_code": "",
            "thread_id": thread_id,
            "email": email,
            "name": "",
            "shipping_url": shipping_url,
            "page_url": page_url
        }
    }
    
    user_conversations[user_id]["history"].append(f"You: {user_message}")
    
    # Initialize a set to track printed events
    printed_events = set()
    
    try:
        events = part_1_graph_purmeo_de.stream(
            {"messages": [("user", (user_message))]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
    
    last_assistant_response = ""
    raw_events = list(events)
    
    for event in raw_events:
        # Print each event
        _print_event(event, printed_events)
        if "messages" in event:
            for message in event["messages"]:
                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                    content = message.content
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                    elif isinstance(content, list):
                        content = " ".join(str(part) for part in content)
                    elif isinstance(content, str):
                        last_assistant_response = content
    
    # Then add the assistant response after getting it
    if last_assistant_response:
        # check if the last_assistant_response json is correct. If not then use LLM to get correct JSON.
        last_assistant_response = last_assistant_response
        # Pull just the "message" field
        message_text = extract_message_field(last_assistant_response)
        if not manager.add_public_comment(ticket_id, strip_html(message_text), "32601040249617"):
            print("Failed to add public comment by agent to ticket")
        else:
            print("Added public comment by agent to ticket")
    print(last_assistant_response)
    return {"response": last_assistant_response}


@app.get("/")
def index(credentials: HTTPBasicCredentials = Depends(authenticate)):
    return FileResponse("index.html", media_type="text/html")




@app.get("/webhook/chat/instagram")
async def verify_webhook(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for Instagram
    """
    logger.info(f"Verification attempt: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    logger.info(f"Webhook verify token: {InstagramConfig.WEBHOOK_VERIFY_TOKEN}")
    if hub_mode == "subscribe" and hub_verify_token == InstagramConfig.WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        # Return the challenge as a plain text response
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook/chat/instagram")
async def handle_webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Webhook endpoint to receive Instagram messages and events
    """
    # Get raw body for signature verification
    body = await request.body()
    body_text = body.decode('utf-8')
    
    # Verify webhook signature if provided
    if x_hub_signature_256 and InstagramConfig.APP_SECRET:
        # Verify the signature
        expected_signature = hmac.new(
            InstagramConfig.APP_SECRET.encode('utf-8'), 
            body, 
            hashlib.sha256
        ).hexdigest()
        
        received_signature = x_hub_signature_256.replace("sha256=", "")
        
        if not hmac.compare_digest(expected_signature, received_signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Parse the webhook payload
    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received webhook payload: {payload_dict}")
    
        # Process different types of webhook events
        if payload_dict.get("object") == "instagram":
            for entry in payload_dict.get("entry", []):

                # Handle direct messages
                if "messaging" in entry:
                    for messaging in entry["messaging"]:
                        if "message" in messaging:
                            user_id = messaging["sender"]["id"]
                            message_text = messaging["message"].get("text", "")
                            #check if the message is not from our instagram page
                            if (user_id != "17841451105460824"):
                            
                                logger.info(f"Received message from {user_id}: {message_text}")
                                print("Received message from {user_id}: {message_text}")
                                
                                user_message = message_text
                                
                                if not user_id or not user_message:
                                    raise HTTPException(status_code=400, detail="Both user_id and message are required")
                                
                                if user_id not in user_conversations:
                                    user_conversations[user_id] = {
                                        "thread_id": user_id,
                                        "history": [],
                                        "status": "new",
                                    }
                                
                                # the problem could be that a user couldnt be able to create multiple tickets.
                                thread_id = user_conversations[user_id]["thread_id"]
                                print("Thread ID from chat: ", thread_id)

                                name = ""
                                #if email is in the user message then assign user email to the email variable
                                
                                
                                # Check if this thread already has a ticket
                                if thread_id not in requests_and_tickets:
                                    #TODO : Get the username and name from instagram api for that id
                                    #user_info = get_instagram_user_from_psid(user_id)
    
                                    #if user_info:
                                     #   print(f"User name: {user_info['name']}")
                                      #  print(f"Instagram ID: {user_info['instagram_id']}")
                                   # temp_name = user_info['name']
                                    temp_name = "ES Instagram User"
                                    requester_id, ticket_id = manager.create_instagram_ticket(user_message, temp_name,user_id, country="ES")
                                    requests_and_tickets[thread_id] = {
                                        "requester_id": requester_id,
                                        "ticket_id": ticket_id
                                    }
                                   # manager.update_instagram_fields(requester_id, user_info['psid'], user_info['instagram_id'])
                                else:
                                    requester_id = requests_and_tickets[thread_id]["requester_id"]
                                    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
                                

                                # First, add the user message to the ticket
                                if not manager.add_public_comment(ticket_id, user_message, requester_id):
                                    print("Failed to add public comment to ticket")
                                else:
                                    print("Added public comment to ticket")
                                
                                if user_conversations[user_id]["status"] != "escalated":
                                    config = {
                                        "configurable": {
                                            "order_id": "",
                                            "postal_code": "",
                                            "thread_id": thread_id,
                                            "email": "",
                                            "name": name,
                                            "shipping_url": shipping_url,
                                            "page_url": ""
                                        }
                                    }
                                    
                                    user_conversations[user_id]["history"].append(f"You: {user_message}")
                                    
                                    # Initialize a set to track printed events
                                    printed_events = set()
                                    
                                    try:
                                        events = part_1_graph_ig_spain.stream(
                                            {"messages": [("user", (user_message))]}, config, stream_mode="values"
                                        )
                                    except Exception as e:
                                        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
                                    
                                    last_assistant_response = ""
                                    raw_events = list(events)
                                    
                                    for event in raw_events:
                                        # Print each event
                                        _print_event(event, printed_events)
                                        if "messages" in event:
                                            for message in event["messages"]:
                                                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                                                    content = message.content
                                                    if isinstance(content, dict) and "text" in content:
                                                        content = content["text"]
                                                    elif isinstance(content, list):
                                                        content = " ".join(str(part) for part in content)
                                                    elif isinstance(content, str):
                                                        last_assistant_response = content
                                    
                                    # Then add the assistant response after getting it
                                    if last_assistant_response:
                                        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
                                            print("Failed to add public comment by agent to ticket")
                                        else:
                                            print("Added public comment by agent to ticket")

                                    print(type(last_assistant_response))
                                    print(last_assistant_response)
                                    logger.log(
                                        logging.INFO, f"Sending message to user {user_id}: {last_assistant_response}"
                                    )
                                    print(user_id)
                                    #testing
                                    #Remove this user_id in case of real deployment
                                    #user_id = "3891652204430815"
                                    # Send the response back to the user
                                    send_insta(user_id, last_assistant_response)
                                else:
                                    continue
                
                # Handle comments, mentions, etc.
                elif "changes" in entry:
                    for change in entry["changes"]:
                        logger.info(f"Received change event: {change}")
                        # Process different types of changes
                        # ...
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/webhook/es/whatsapp")
async def verify_webhook(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for Instagram
    """
    logger.info(f"Verification attempt: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    logger.info(f"Webhook verify token: {WhatsappConfig.WEBHOOK_VERIFY_TOKEN}")
    if hub_mode == "subscribe" and hub_verify_token == WhatsappConfig.WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        # Return the challenge as a plain text response
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.get("/webhook/it/whatsapp")
async def verify_webhook_it(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for Whatsapp
    """
    logger.info(f"Verification attempt: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    logger.info(f"Webhook verify token: {WhatsappConfig.WEBHOOK_VERIFY_TOKEN}")
    if hub_mode == "subscribe" and hub_verify_token == WhatsappConfig.WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        # Return the challenge as a plain text response
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook/es/whatsapp")
async def webhook_es_whatsapp(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Webhook endpoint to receive WhatsApp messages via Facebook Graph API.
    - Verifies incoming requests using app secret if provided.
    - Processes user messages only.
    - Responds with AI-generated content.
    - Sends response back via WhatsApp.
    """
    body = await request.body()
    body_text = body.decode('utf-8')
 

    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received WhatsApp webhook payload: {payload_dict}")

        if payload_dict.get("object") == "whatsapp_business_account":
            for entry in payload_dict.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if "messages" in value:
                        for message in value["messages"]:
                            from_wa_id = message.get("from")
                            wa_profile_name = value.get("contacts", [{}])[0].get("profile", {}).get("name", "")
                            message_type = message.get("type")
                            
                            if message_type == "text":
                                message_text = message.get("text", {}).get("body", "").strip()

                                if not message_text:
                                    continue  # Skip non-text or empty messages

                                logger.info(f"Received WhatsApp message from {from_wa_id} ({wa_profile_name}): {message_text}")
                                print(f"Received WhatsApp message from {from_wa_id}: {message_text}")

                                # Initialize user context
                                if from_wa_id not in user_conversations:
                                    user_conversations[from_wa_id] = {
                                        "thread_id": from_wa_id,
                                        "history": [],
                                        "status": "new",
                                        "name": wa_profile_name,
                                        "country": "ES"
                                    }

                                thread_id = user_conversations[from_wa_id]["thread_id"]
                                name = user_conversations[from_wa_id]["name"]

                                # Create or get Zendesk ticket
                                if thread_id not in requests_and_tickets:
                                    requester_id, ticket_id = manager.create_whatsapp_ticket(
                                        message_text, name, from_wa_id, country="ES"
                                    )
                                    requests_and_tickets[thread_id] = {
                                        "requester_id": requester_id,
                                        "ticket_id": ticket_id
                                    }
                                else:
                                    requester_id = requests_and_tickets[thread_id]["requester_id"]
                                    ticket_id = requests_and_tickets[thread_id]["ticket_id"]

                                # Add user message to ticket
                                if not manager.add_public_comment(ticket_id, message_text, requester_id):
                                    logger.error("Failed to add public comment to ticket")
                                else:
                                    logger.info("Added public comment to ticket")

                                # Skip if escalated
                                if user_conversations[from_wa_id]["status"] == "escalated":
                                    continue

                                # Prepare config for AI assistant
                                config = {
                                    "configurable": {
                                        "order_id": "",
                                        "postal_code": "",
                                        "thread_id": thread_id,
                                        "email": "",
                                        "name": name,
                                        "shipping_url": shipping_url,
                                        "page_url": ""
                                    }
                                }

                                # Stream AI response
                                printed_events = set()
                                try:
                                    events = part_1_graph_ig_spain.stream(
                                        {"messages": [("user", message_text)]},
                                        config,
                                        stream_mode="values"
                                    )
                                except Exception as e:
                                    logger.error(f"AI response error: {str(e)}")
                                    continue

                                last_assistant_response = ""
                                raw_events = list(events)
                                for event in raw_events:
                                    _print_event(event, printed_events)
                                    if "messages" in event:
                                        for msg in event["messages"]:
                                            if hasattr(msg, "content") and "AIMessage" in str(type(msg)):
                                                content = msg.content
                                                if isinstance(content, dict) and "text" in content:
                                                    content = content["text"]
                                                elif isinstance(content, list):
                                                    content = " ".join(str(part) for part in content)
                                                elif isinstance(content, str):
                                                    last_assistant_response = content

                                # Log and send reply
                                if last_assistant_response:
                                    if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
                                        logger.error("Failed to add AI response to ticket")
                                    else:
                                        logger.info("Added AI response to ticket")

                                    logger.info(f"Sending WhatsApp reply to {from_wa_id}: {last_assistant_response}")
                                    WhatsappMessageService.send_reply_message(from_wa_id, last_assistant_response)

                        return JSONResponse(status_code=200, content={"status": "success"})
        return JSONResponse(status_code=200, content={"status": "ignored"})
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/it/whatsapp")
async def webhook_it_whatsapp(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Webhook endpoint to receive WhatsApp messages via Facebook Graph API.
    - Verifies incoming requests using app secret if provided.
    - Processes user messages only.
    - Responds with AI-generated content.
    - Sends response back via WhatsApp.
    """
    body = await request.body()
    body_text = body.decode('utf-8')
 

    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received WhatsApp webhook payload IT: {payload_dict}")

        if payload_dict.get("object") == "whatsapp_business_account":
            for entry in payload_dict.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if "messages" in value:
                        for message in value["messages"]:
                            from_wa_id = message.get("from")
                            wa_profile_name = value.get("contacts", [{}])[0].get("profile", {}).get("name", "")
                            message_type = message.get("type")
                            
                            if message_type == "text":
                                message_text = message.get("text", {}).get("body", "").strip()

                                if not message_text:
                                    continue  # Skip non-text or empty messages

                                logger.info(f"Received WhatsApp IT message from {from_wa_id} ({wa_profile_name}): {message_text}")
                                print(f"Received WhatsApp message from {from_wa_id}: {message_text}")

                                # Initialize user context
                                if from_wa_id not in user_conversations:
                                    user_conversations[from_wa_id] = {
                                        "thread_id": from_wa_id,
                                        "history": [],
                                        "status": "new",
                                        "name": wa_profile_name,
                                        "country": "ES"
                                    }

                                thread_id = user_conversations[from_wa_id]["thread_id"]
                                name = user_conversations[from_wa_id]["name"]

                                # Create or get Zendesk ticket
                                if thread_id not in requests_and_tickets:
                                    requester_id, ticket_id = manager.create_whatsapp_ticket(
                                        message_text, name, from_wa_id, country="IT"
                                    )
                                    requests_and_tickets[thread_id] = {
                                        "requester_id": requester_id,
                                        "ticket_id": ticket_id
                                    }
                                else:
                                    requester_id = requests_and_tickets[thread_id]["requester_id"]
                                    ticket_id = requests_and_tickets[thread_id]["ticket_id"]

                                # Add user message to ticket
                                if not manager.add_public_comment(ticket_id, message_text, requester_id):
                                    logger.error("Failed to add public Whatsapp IT  comment to ticket")
                                else:
                                    logger.info("Added public comment Whatsapp IT to ticket")

                                # Skip if escalated
                                if user_conversations[from_wa_id]["status"] == "escalated":
                                    continue

                                # Prepare config for AI assistant
                                config = {
                                    "configurable": {
                                        "order_id": "",
                                        "postal_code": "",
                                        "thread_id": thread_id,
                                        "email": "",
                                        "name": name,
                                        "shipping_url": shipping_url,
                                        "page_url": ""
                                    }
                                }

                                # Stream AI response
                                printed_events = set()
                                try:
                                    events = part_1_graph_ig_italy.stream(
                                        {"messages": [("user", message_text)]},
                                        config,
                                        stream_mode="values"
                                    )
                                except Exception as e:
                                    logger.error(f"AI response error: {str(e)}")
                                    continue

                                last_assistant_response = ""
                                raw_events = list(events)
                                for event in raw_events:
                                    _print_event(event, printed_events)
                                    if "messages" in event:
                                        for msg in event["messages"]:
                                            if hasattr(msg, "content") and "AIMessage" in str(type(msg)):
                                                content = msg.content
                                                if isinstance(content, dict) and "text" in content:
                                                    content = content["text"]
                                                elif isinstance(content, list):
                                                    content = " ".join(str(part) for part in content)
                                                elif isinstance(content, str):
                                                    last_assistant_response = content

                                # Log and send reply
                                if last_assistant_response:
                                    if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
                                        logger.error("Failed to add AI response to ticket")
                                    else:
                                        logger.info("Added AI response to ticket")

                                    logger.info(f"Sending WhatsApp reply to {from_wa_id}: {last_assistant_response}")
                                    WhatsappMessageService.send_reply_message_it(from_wa_id, last_assistant_response)

                        return JSONResponse(status_code=200, content={"status": "success"})
        return JSONResponse(status_code=200, content={"status": "ignored"})
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/email/spain")
async def webhook_email_spain(request: Request):
    """
    Webhook endpoint to receive and process email tickets from Zendesk for Spanish customers.
    This endpoint:
    1. Receives email ticket updates from Zendesk
    2. Acknowledges receipt immediately (preventing timeouts)
    3. Processes the ticket information asynchronously
    4. Uses AI to generate appropriate responses
    5. Posts responses back to the ticket
    """
    body = await request.body()
    body_text = body.decode('utf-8')
    
    # Parse the webhook payload
    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received email webhook payload: {payload_dict}")
        
        # Extract just the essential information needed to queue the task
        ticket_id = payload_dict.get("ticket", {}).get("id")
        
        if not ticket_id:
            logger.error("Missing required ticket ID in webhook payload")
            return {"status": "error", "message": "Missing ticket ID"}
            
        # Queue the processing task to run in the background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            process_email_ticket, 
            payload_dict=payload_dict
        )
        
        # Respond immediately to the webhook
        return JSONResponse(
            status_code=200,
            content={"status": "accepted", "message": f"Ticket {ticket_id} queued for processing"},
            background=background_tasks
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse webhook JSON: {str(e)}")
        return {"status": "error", "message": "Invalid JSON payload"}
    except Exception as e:
        logger.error(f"Error in webhook receiver: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_email_ticket(payload_dict: dict):
    """
    Process the email ticket asynchronously after webhook has already responded.
    This allows for longer processing time without causing gateway timeouts.
    """
    try:
        # Extract ticket information
        ticket_id = payload_dict.get("ticket", {}).get("id")
        requester_id = payload_dict.get("ticket", {}).get("requester_id")
        requester_email = payload_dict.get("ticket", {}).get("requester", {}).get("email", "")
        requester_name = payload_dict.get("ticket", {}).get("requester", {}).get("name", "")
        latest_comment = payload_dict.get("ticket", {}).get("latest_comment") or payload_dict.get("ticket", {}).get("latest_public_comment")
        
        # Check if this is a new message we need to respond to
        if not latest_comment or not requester_id:
            logger.error(f"Missing required information for ticket {ticket_id}")
            return
        
        # Generate a thread ID based on the ticket ID to maintain conversation context
        thread_id = f"email_{ticket_id}"
        
        # Add ticket to our tracking system if it's new
        if thread_id not in requests_and_tickets:
            requests_and_tickets[thread_id] = {
                "requester_id": requester_id,
                "ticket_id": ticket_id
            }
        
        # Check if this is our own comment (avoid loops)
        # if payload_dict.get("ticket", {}).get("latest_comment_added_by_id") == "32601040249617":  # ID of the bot agent
        #     logger.info(f"Skipping our own comment on ticket {ticket_id}")
        #     return
        
        # Configure AI processing
        config = {
            "configurable": {
                "order_id": "",
                "postal_code": "",
                "thread_id": thread_id,
                "email": requester_email,
                "name": requester_name,
                "shipping_url": shipping_url,
                "page_url": ""
            }
        }
        
        # Track conversation (optional)
        if thread_id not in user_conversations:
            user_conversations[thread_id] = {
                "thread_id": thread_id,
                "history": [],
                "status": "new",
            }
        user_conversations[thread_id]["history"].append(f"Customer: {latest_comment}")
        
        # Initialize event tracking for AI processing
        printed_events = set()
        
        # Process with AI using the email-specific graph
        try:
            events = part_1_graph_email_spain.stream(
                {"messages": [("user", latest_comment)]}, config, stream_mode="values"
            )
        except Exception as e:
            logger.error(f"Failed to process email with AI for ticket {ticket_id}: {str(e)}")
            return
        
        # Extract AI response
        last_assistant_response = ""
        raw_events = list(events)
        
        for event in raw_events:
            _print_event(event, printed_events)
            if "messages" in event:
                for message in event["messages"]:
                    if hasattr(message, "content") and "AIMessage" in str(type(message)):
                        content = message.content
                        if isinstance(content, dict) and "text" in content:
                            content = content["text"]
                        elif isinstance(content, list):
                            content = " ".join(str(part) for part in content)
                        elif isinstance(content, str):
                            last_assistant_response = content
        
        # Post AI response back to the ticket
        if last_assistant_response:
            if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "31549253490321"):
                logger.error(f"Failed to add AI response to ticket {ticket_id}")
            else:
                logger.info(f"Successfully added AI response to ticket {ticket_id}")
    
    except Exception as e:
        logger.error(f"Error in background processing for ticket: {str(e)}")



# email italy

@app.post("/webhook/email/italy")
async def webhook_email_italy(request: Request):
    """
    Webhook endpoint to receive and process email tickets from Zendesk for Spanish customers.
    This endpoint:
    1. Receives email ticket updates from Zendesk
    2. Acknowledges receipt immediately (preventing timeouts)
    3. Processes the ticket information asynchronously
    4. Uses AI to generate appropriate responses
    5. Posts responses back to the ticket
    """
    body = await request.body()
    body_text = body.decode('utf-8')
    
    # Parse the webhook payload
    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received email webhook payload: {payload_dict}")
        
        # Extract just the essential information needed to queue the task
        ticket_id = payload_dict.get("ticket", {}).get("id")
        
        if not ticket_id:
            logger.error("Missing required ticket ID in webhook payload")
            return {"status": "error", "message": "Missing ticket ID"}
            
        # Queue the processing task to run in the background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            process_email_ticket_italy, 
            payload_dict=payload_dict
        )
        
        # Respond immediately to the webhook
        return JSONResponse(
            status_code=200,
            content={"status": "accepted", "message": f"Ticket {ticket_id} queued for processing"},
            background=background_tasks
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse webhook JSON: {str(e)}")
        return {"status": "error", "message": "Invalid JSON payload"}
    except Exception as e:
        logger.error(f"Error in webhook receiver: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_email_ticket_italy(payload_dict: dict):
    """
    Process the email ticket asynchronously after webhook has already responded.
    This allows for longer processing time without causing gateway timeouts.
    """
    try:
        # Extract ticket information
        ticket_id = payload_dict.get("ticket", {}).get("id")
        requester_id = payload_dict.get("ticket", {}).get("requester_id")
        requester_email = payload_dict.get("ticket", {}).get("requester", {}).get("email", "")
        requester_name = payload_dict.get("ticket", {}).get("requester", {}).get("name", "")
        latest_comment = payload_dict.get("ticket", {}).get("latest_comment") or payload_dict.get("ticket", {}).get("latest_public_comment")
        
        # Check if this is a new message we need to respond to
        if not latest_comment or not requester_id:
            logger.error(f"Missing required information for ticket {ticket_id}")
            return
        
        # Generate a thread ID based on the ticket ID to maintain conversation context
        thread_id = f"email_{ticket_id}"
        
        # Add ticket to our tracking system if it's new
        if thread_id not in requests_and_tickets:
            requests_and_tickets[thread_id] = {
                "requester_id": requester_id,
                "ticket_id": ticket_id
            }
        
        # Check if this is our own comment (avoid loops)
        # if payload_dict.get("ticket", {}).get("latest_comment_added_by_id") == "32601040249617":  # ID of the bot agent
        #     logger.info(f"Skipping our own comment on ticket {ticket_id}")
        #     return
        
        # Configure AI processing
        config = {
            "configurable": {
                "order_id": "",
                "postal_code": "",
                "thread_id": thread_id,
                "email": requester_email,
                "name": requester_name,
                "shipping_url": shipping_url,
                "page_url": ""
            }
        }
        
        # Track conversation (optional)
        if thread_id not in user_conversations:
            user_conversations[thread_id] = {
                "thread_id": thread_id,
                "history": [],
                "status": "new",
            }
        user_conversations[thread_id]["history"].append(f"Customer: {latest_comment}")
        
        # Initialize event tracking for AI processing
        printed_events = set()
        
        # Process with AI using the email-specific graph
        try:
            events = part_1_graph_email_italy.stream(
                {"messages": [("user", latest_comment)]}, config, stream_mode="values"
            )
        except Exception as e:
            logger.error(f"Failed to process email with AI for ticket {ticket_id}: {str(e)}")
            return
        
        # Extract AI response
        last_assistant_response = ""
        raw_events = list(events)
        
        for event in raw_events:
            _print_event(event, printed_events)
            if "messages" in event:
                for message in event["messages"]:
                    if hasattr(message, "content") and "AIMessage" in str(type(message)):
                        content = message.content
                        if isinstance(content, dict) and "text" in content:
                            content = content["text"]
                        elif isinstance(content, list):
                            content = " ".join(str(part) for part in content)
                        elif isinstance(content, str):
                            last_assistant_response = content
        
        # Post AI response back to the ticket
        if last_assistant_response:
            if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "31549253490321"):
                logger.error(f"Failed to add AI response to ticket {ticket_id}")
            else:
                logger.info(f"Successfully added AI response to ticket {ticket_id}")
    
    except Exception as e:
        logger.error(f"Error in background processing for ticket: {str(e)}")


#email italy end



@app.get("/webhook/chat/it/instagram")
async def verify_webhook_it(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for Instagram
    """
    logger.info(f"Verification attempt: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    logger.info(f"Webhook verify token: {InstagramConfig.WEBHOOK_VERIFY_TOKEN}")
    if hub_mode == "subscribe" and hub_verify_token == InstagramConfig.WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        # Return the challenge as a plain text response
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook/chat/it/instagram")
async def handle_webhook_it(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Webhook endpoint to receive Instagram messages and events
    """
    # Get raw body for signature verification
    body = await request.body()
    body_text = body.decode('utf-8')
    
    # Verify webhook signature if provided
    if x_hub_signature_256 and InstagramConfig.APP_SECRET_IT:
        # Verify the signature
        expected_signature = hmac.new(
            InstagramConfig.APP_SECRET_IT.encode('utf-8'), 
            body, 
            hashlib.sha256
        ).hexdigest()
        
        received_signature = x_hub_signature_256.replace("sha256=", "")
        
        if not hmac.compare_digest(expected_signature, received_signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Parse the webhook payload
    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received webhook payload: {payload_dict}")
    
        # Process different types of webhook events
        if payload_dict.get("object") == "instagram":
            for entry in payload_dict.get("entry", []):

                # Handle direct messages
                if "messaging" in entry:
                    for messaging in entry["messaging"]:
                        if "message" in messaging:
                            user_id = messaging["sender"]["id"]
                            message_text = messaging["message"].get("text", "")
                            #check if the message is not from our instagram page
                            if (user_id != "17841406043537503"):
                            
                                logger.info(f"Received message from {user_id}: {message_text}")
                                print("Received message from {user_id}: {message_text}")
                                
                                user_message = message_text
                                
                                if not user_id or not user_message:
                                    raise HTTPException(status_code=400, detail="Both user_id and message are required")
                                
                                if user_id not in user_conversations:
                                    user_conversations[user_id] = {
                                        "thread_id": user_id,
                                        "history": [],
                                        "status": "new",
                                    }
                                
                                # the problem could be that a user couldnt be able to create multiple tickets.
                                thread_id = user_conversations[user_id]["thread_id"]
                                print("Thread ID from chat: ", thread_id)

                                name = ""
                                #if email is in the user message then assign user email to the email variable
                                
                                
                                # Check if this thread already has a ticket
                                if thread_id not in requests_and_tickets:
                                    #TODO : Get the username and name from instagram api for that id
                                    #user_info = get_instagram_user_from_psid(user_id)
    
                                    #if user_info:
                                     #   print(f"User name: {user_info['name']}")
                                      #  print(f"Instagram ID: {user_info['instagram_id']}")
                                   # temp_name = user_info['name']
                                    temp_name = "IT Instagram User"
                                    requester_id, ticket_id = manager.create_instagram_ticket(user_message, temp_name,user_id, country="IT")
                                    requests_and_tickets[thread_id] = {
                                        "requester_id": requester_id,
                                        "ticket_id": ticket_id
                                    }
                                   # manager.update_instagram_fields(requester_id, user_info['psid'], user_info['instagram_id'])
                                else:
                                    requester_id = requests_and_tickets[thread_id]["requester_id"]
                                    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
                                

                                # First, add the user message to the ticket
                                if not manager.add_public_comment(ticket_id, user_message, requester_id):
                                    print("Failed to add public comment to ticket")
                                else:
                                    print("Added public comment to ticket")
                                
                                if user_conversations[user_id]["status"] != "escalated":
                                    config = {
                                        "configurable": {
                                            "order_id": "",
                                            "postal_code": "",
                                            "thread_id": thread_id,
                                            "email": "",
                                            "name": name,
                                            "shipping_url": shipping_url,
                                            "page_url": ""
                                        }
                                    }
                                    
                                    user_conversations[user_id]["history"].append(f"You: {user_message}")
                                    
                                    # Initialize a set to track printed events
                                    printed_events = set()
                                    
                                    try:
                                        events = part_1_graph_ig_italy.stream(
                                            {"messages": [("user", (user_message))]}, config, stream_mode="values"
                                        )
                                    except Exception as e:
                                        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
                                    
                                    last_assistant_response = ""
                                    raw_events = list(events)
                                    
                                    for event in raw_events:
                                        # Print each event
                                        _print_event(event, printed_events)
                                        if "messages" in event:
                                            for message in event["messages"]:
                                                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                                                    content = message.content
                                                    if isinstance(content, dict) and "text" in content:
                                                        content = content["text"]
                                                    elif isinstance(content, list):
                                                        content = " ".join(str(part) for part in content)
                                                    elif isinstance(content, str):
                                                        last_assistant_response = content
                                    
                                    # Then add the assistant response after getting it
                                    if last_assistant_response:
                                        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
                                            print("Failed to add public comment by agent to ticket")
                                        else:
                                            print("Added public comment by agent to ticket")

                                    print(type(last_assistant_response))
                                    print(last_assistant_response)
                                    logger.log(
                                        logging.INFO, f"Sending message to user {user_id}: {last_assistant_response}"
                                    )
                                    print(user_id)
                                    #testing
                                    #Remove this user_id in case of real deployment
                                    #user_id = "3891652204430815"
                                    # Send the response back to the user
                                    send_insta_it(user_id, last_assistant_response)
                                else:
                                    continue
                
                # Handle comments, mentions, etc.
                elif "changes" in entry:
                    for change in entry["changes"]:
                        logger.info(f"Received change event: {change}")
                        # Process different types of changes
                        # ...
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

#instagram germany:
@app.get("/webhook/chat/de/instagram")
async def verify_webhook_de(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for Instagram
    """
    logger.info(f"Verification attempt: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    logger.info(f"Webhook verify token: {InstagramConfig.WEBHOOK_VERIFY_TOKEN}")
    if hub_mode == "subscribe" and hub_verify_token == InstagramConfig.WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        # Return the challenge as a plain text response
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook/chat/de/instagram")
async def handle_webhook_de(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Webhook endpoint to receive Instagram messages and events
    """
    # Get raw body for signature verification
    body = await request.body()
    body_text = body.decode('utf-8')
    
    # Verify webhook signature if provided
    if x_hub_signature_256 and InstagramConfig.APP_SECRET_DE:
        # Verify the signature
        expected_signature = hmac.new(
            InstagramConfig.APP_SECRET_DE.encode('utf-8'), 
            body, 
            hashlib.sha256
        ).hexdigest()
        
        received_signature = x_hub_signature_256.replace("sha256=", "")
        
        if not hmac.compare_digest(expected_signature, received_signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Parse the webhook payload
    try:
        payload_dict = json.loads(body_text)
        logger.info(f"Received webhook payload: {payload_dict}")
    
        # Process different types of webhook events
        if payload_dict.get("object") == "instagram":
            for entry in payload_dict.get("entry", []):

                # Handle direct messages
                if "messaging" in entry:
                    for messaging in entry["messaging"]:
                        if "message" in messaging:
                            user_id = messaging["sender"]["id"]
                            message_text = messaging["message"].get("text", "")
                            #check if the message is not from our instagram page
                            if (user_id != "100737502614412"):
                            
                                logger.info(f"Received message from {user_id}: {message_text}")
                                print("Received message from {user_id}: {message_text}")
                                
                                user_message = message_text
                                
                                if not user_id or not user_message:
                                    raise HTTPException(status_code=400, detail="Both user_id and message are required")
                                
                                if user_id not in user_conversations:
                                    user_conversations[user_id] = {
                                        "thread_id": user_id,
                                        "history": [],
                                        "status": "new",
                                    }
                                
                                # the problem could be that a user couldnt be able to create multiple tickets.
                                thread_id = user_conversations[user_id]["thread_id"]
                                print("Thread ID from chat: ", thread_id)

                                name = ""
                                #if email is in the user message then assign user email to the email variable
                                
                                
                                # Check if this thread already has a ticket
                                if thread_id not in requests_and_tickets:
                                    #TODO : Get the username and name from instagram api for that id
                                    #user_info = get_instagram_user_from_psid(user_id)
    
                                    #if user_info:
                                     #   print(f"User name: {user_info['name']}")
                                      #  print(f"Instagram ID: {user_info['instagram_id']}")
                                   # temp_name = user_info['name']
                                    temp_name = "DE Instagram User"
                                    requester_id, ticket_id = manager.create_instagram_ticket(user_message, temp_name,user_id, country="DE")
                                    requests_and_tickets[thread_id] = {
                                        "requester_id": requester_id,
                                        "ticket_id": ticket_id
                                    }
                                   # manager.update_instagram_fields(requester_id, user_info['psid'], user_info['instagram_id'])
                                else:
                                    requester_id = requests_and_tickets[thread_id]["requester_id"]
                                    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
                                

                                # First, add the user message to the ticket
                                if not manager.add_public_comment(ticket_id, user_message, requester_id):
                                    print("Failed to add public comment to ticket")
                                else:
                                    print("Added public comment to ticket")
                                
                                if user_conversations[user_id]["status"] != "escalated":
                                    config = {
                                        "configurable": {
                                            "order_id": "",
                                            "postal_code": "",
                                            "thread_id": thread_id,
                                            "email": "",
                                            "name": name,
                                            "shipping_url": shipping_url,
                                            "page_url": ""
                                        }
                                    }
                                    
                                    user_conversations[user_id]["history"].append(f"You: {user_message}")
                                    
                                    # Initialize a set to track printed events
                                    printed_events = set()
                                    
                                    try:
                                        events = part_1_graph_ig_germany.stream(
                                            {"messages": [("user", (user_message))]}, config, stream_mode="values"
                                        )
                                    except Exception as e:
                                        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")
                                    
                                    last_assistant_response = ""
                                    raw_events = list(events)
                                    
                                    for event in raw_events:
                                        # Print each event
                                        _print_event(event, printed_events)
                                        if "messages" in event:
                                            for message in event["messages"]:
                                                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                                                    content = message.content
                                                    if isinstance(content, dict) and "text" in content:
                                                        content = content["text"]
                                                    elif isinstance(content, list):
                                                        content = " ".join(str(part) for part in content)
                                                    elif isinstance(content, str):
                                                        last_assistant_response = content
                                    
                                    # Then add the assistant response after getting it
                                    if last_assistant_response:
                                        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
                                            print("Failed to add public comment by agent to ticket")
                                        else:
                                            print("Added public comment by agent to ticket")

                                    print(type(last_assistant_response))
                                    print(last_assistant_response)
                                    logger.log(
                                        logging.INFO, f"Sending message to user {user_id}: {last_assistant_response}"
                                    )
                                    print(user_id)
                                    #testing
                                    #Remove this user_id in case of real deployment
                                    #user_id = "3891652204430815"
                                    # Send the response back to the user
                                    send_insta_de(user_id, last_assistant_response)
                                else:
                                    continue
                
                # Handle comments, mentions, etc.
                elif "changes" in entry:
                    for change in entry["changes"]:
                        logger.info(f"Received change event: {change}")
                        # Process different types of changes
                        # ...
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/webhook/zendesk/sendinsta")
async def webhook_listener(request: Request):
    payload = await request.json()
    print("Validated Payload from Zendesk:", payload)
    ticket_id = payload["ticket"]["id"]
    print("Ticket ID: ", ticket_id)
    user_id = get_thread_id_from_ticket_id(requests_and_tickets, ticket_id)
    print("User ID: ", user_id)
    last_assistant_response = payload["ticket"]["latest_public_comment"]
    #forward the message to instagram
    send_insta(user_id, last_assistant_response)
    return {"status": "Message sent to instagram successfully"}

@app.post("/webhook/zendesk/sendinsta/de")
async def webhook_listenerde(request: Request):
    payload = await request.json()
    print("Validated Payload from Zendesk:", payload)
    ticket_id = payload["ticket"]["id"]
    print("Ticket ID: ", ticket_id)
    user_id = get_thread_id_from_ticket_id(requests_and_tickets, ticket_id)
    print("User ID: ", user_id)
    last_assistant_response = payload["ticket"]["latest_public_comment"]
    #forward the message to instagram
    send_insta_de(user_id, last_assistant_response)
    return {"status": "Message sent to instagram successfully"}

@app.post("/webhook/zendesk/sendinsta/it")
async def webhook_listenerde(request: Request):
    payload = await request.json()
    print("Validated Payload from Zendesk:", payload)
    ticket_id = payload["ticket"]["id"]
    print("Ticket ID: ", ticket_id)
    user_id = get_thread_id_from_ticket_id(requests_and_tickets, ticket_id)
    print("User ID: ", user_id)
    last_assistant_response = payload["ticket"]["latest_public_comment"]
    #forward the message to instagram
    send_insta_it(user_id, last_assistant_response)
    return {"status": "Message sent to instagram successfully"}

@app.post("/webhook/zendesk/sendwhatsapp")
async def webhook_whatsapp_listener(request: Request):
    """
    Webhook endpoint to forward Zendesk agent replies to WhatsApp users.
    - Receives Zendesk payload with ticket ID and latest comment.
    - Retrieves the user's WhatsApp ID from thread/ticket mapping.
    - Sends the message via WhatsApp using WhatsappMessageService.
    """
    try:
        payload = await request.json()
        logger.info(f"Validated Payload from Zendesk: {payload}")
        print("Validated Payload from Zendesk:", payload)

        ticket_id = payload["ticket"]["id"]
        logger.info(f"Ticket ID: {ticket_id}")
        print("Ticket ID: ", ticket_id)

        # Get the user's WhatsApp ID (thread_id should map to wa_id)
        user_id = get_thread_id_from_ticket_id(requests_and_tickets, ticket_id)
        if not user_id:
            raise HTTPException(status_code=404, detail="User ID not found for ticket")

        logger.info(f"User ID (WhatsApp ID): {user_id}")
        print("User ID (WhatsApp ID): ", user_id)

        # Get the latest public comment from the ticket
        last_assistant_response = payload["ticket"]["latest_public_comment"]
        if not last_assistant_response:
            raise HTTPException(status_code=400, detail="No public comment found in ticket")

        # Send the message via WhatsApp
        logger.info(f"Sending message to WhatsApp user {user_id}: {last_assistant_response}")
        WhatsappMessageService.send_reply_message(user_id, last_assistant_response)

        return {"status": "success", "message": "Message sent to WhatsApp successfully"}

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except KeyError as e:
        logger.error(f"Missing expected field in payload: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to send message to WhatsApp: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint to verify the service is running"""
    return {"status": "Webhook server is running"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response


@app.post("/webhook/zendesk/classify")
async def classify_ticket_relevance(request: Request):
    """
    Webhook endpoint to classify ticket comments as relevant or not using Claude AI.
    If not relevant, mark the ticket as 'Not Relevant' in Zendesk.
    """
    try:
        payload = await request.json()
        logger.info(f"Received payload for relevance classification: {payload}")
        print("Received payload for relevance classification: ", payload)
        
        # Extract required fields
        ticket_id = payload.get("ticket", {}).get("id")
        latest_comment = payload.get("ticket", {}).get("latest_comment") or payload.get("ticket", {}).get("latest_public_comment")
        
        print("Ticket ID: ", ticket_id)
        print("Latest Comment: ", latest_comment)
        if not ticket_id or not latest_comment:
            logger.error("Missing ticket_id or latest_comment in webhook payload.")
            raise HTTPException(status_code=400, detail="Invalid payload: missing ticket_id or comment.")
        
        logger.info(f"Ticket ID: {ticket_id} | Comment: {latest_comment}")
        
        # Ask Claude whether the comment is relevant
        system_prompt = """
You are an AI Assistant. Classify the following ticket comment as either RELEVANT or NOT_RELEVANT for customer support.

Rules:
- RELEVANT: If the customer is asking for help, product questions, delivery issues, returns, order status, etc.
- NOT: If the comment is spam, emojis only, promotional, sales or not related to support.

Example:
Comment: "I need help with my order"
Response: RELEVANT

Comment: "Hey there! We’re the team behind ADDICT+, and we’ve been keeping an eye out for someone just like you!
We’re thrilled to invite you to our 😍EXCLUSIVE ACTIVEWEAR DROP!

We want to gift you our special ADDICTPACK 📦 — it’s packed with 5 FREE premium items just for you! 🎁✨Yes, it’s 100% real, and we can’t wait to see you rock our gear!"
Response: NOT

Comment: "I love your product! ❤️"
Response: RELEVANT

Comment: "Check out this amazing deal! 💰"
Response: NOT

Comment: "Can you please assist me with my order?"
Response: RELEVANT


Respond with only one word: RELEVANT or NOT.
"""
        user_prompt = f"Comment: {latest_comment}\n\nIs this comment relevant?"

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        classification = response.content[0].text.strip().upper()
        logger.info(f"Claude classified the comment as: {classification}")
        
        if classification == "NOT":
            # Call your existing method to mark the ticket
            success = manager.mark_ticket_as_not_relevant(ticket_id)
            if success:
                return {"status": "Ticket marked as Not Relevant successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to mark ticket as not relevant.")
        
        return {"status": f"Comment classified as {classification}, no action needed."}
    
    except Exception as e:
        logger.error(f"Error in classify_ticket_relevance webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


