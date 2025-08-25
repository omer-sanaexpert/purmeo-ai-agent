from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import httpx
import os
import logging
from typing import Optional
from dotenv import load_dotenv
from typing import List
from fastapi import Query

from tools import send_insta

class ReceivedMessage(BaseModel):
    sender_username: str
    message_text: str
    timestamp: str


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Instagram Hello Sender")

# Instagram API credentials
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTA_WEBHOOK")
INSTAGRAM_BUSINESS_ID = os.getenv("INSTAGRAM_BUSINESS_ID")  # Your Instagram Business Account ID
API_VERSION = "v18.0"  # Using the latest API version as of 2025

# Set to True to bypass actual API calls for testing
USE_MOCK = os.getenv("USE_MOCK", "True").lower() == "true"

class MessageResponse(BaseModel):
    success: bool
    message: str
    debug_info: Optional[dict] = None

async def _send_hello_implementation(username: str, debug: bool = False):
    """
    Implementation logic for sending a 'hello' message to an Instagram user.
    """
    debug_info = {}
    
    try:
        # If mock mode is enabled, just return a mock success response
        if USE_MOCK:
            logger.info(f"MOCK MODE: Simulating sending message to {username}")
            return {
                "success": True, 
                "message": f"MOCK: Hello message sent to {username}",
                "debug_info": {"mock_mode": True, "username": username} if debug else None
            }
        
        logger.info(f"Attempting to send message to Instagram user: {username}")
        
        if not INSTAGRAM_ACCESS_TOKEN:
            raise ValueError("INSTAGRAM_ACCESS_TOKEN not set in environment variables")
            
        if not INSTAGRAM_BUSINESS_ID:
            raise ValueError("INSTAGRAM_BUSINESS_ID not set in environment variables")
        
        async with httpx.AsyncClient() as client:
            # Current approach to message a user:
            # 1. We don't search for usernames directly - we need the PSID (Page-Scoped ID)
            # 2. The user must have messaged your business account first
            
            # In a real implementation, you would have a database mapping usernames to PSIDs
            # For demonstration, we'll use a mock PSID lookup
            
            # Log that we're using a simulated PSID
            logger.info(f"Note: Using a simulated PSID for {username} since direct username lookup is not available")
            
            # This is where you would look up the PSID for the username in your database
            # For this example, we'll create a simulated PSID based on the username
            simulated_psid = f"3891652204430815"
            
            debug_info["username"] = username
            debug_info["simulated_psid"] = simulated_psid
            debug_info["note"] = "Using simulated PSID - in production, get this from your database"
            
            # Send message using the Instagram Messaging API
            message_url = f"https://graph.facebook.com/{API_VERSION}/{INSTAGRAM_BUSINESS_ID}/messages"
            
            message_data = {
                "recipient": {"id": simulated_psid},
                "message": {"text": "Hello from SanaExpert!"},
                "access_token": INSTAGRAM_ACCESS_TOKEN
            }
            
            logger.info(f"Sending message to simulated PSID: {simulated_psid}")
            message_response = await client.post(message_url, json=message_data)
            
            debug_info["message_url"] = message_url
            debug_info["message_status"] = message_response.status_code
            
            if message_response.status_code != 200:
                debug_info["message_response"] = message_response.text
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to send message. API returned: {message_response.text}"
                )
            
            debug_info["message_response"] = message_response.json()
            
            return {
                "success": True, 
                "message": f"Hello message sent to {username}",
                "debug_info": debug_info if debug else None
            }
            
    except HTTPException as he:
        # Re-raise HTTP exceptions with debug info
        if debug:
            he.detail = {"error": he.detail, "debug_info": debug_info}
        raise he
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}", exc_info=True)
        # Include debugging information in the error response
        if debug:
            error_detail = {"error": str(e), "debug_info": debug_info}
        else:
            error_detail = f"Error sending message: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/send-hello/{username}", response_model=MessageResponse)
async def send_hello_get(
    username: str, 
    debug: bool = Query(False, description="Enable debug mode to get additional information")
):
    """
    Send a 'hello' message to a specific Instagram user using GET request.
    """
    return await _send_hello_implementation(username, debug)

@app.post("/send-hello/{username}", response_model=MessageResponse)
async def send_hello_post(
    username: str, 
    debug: bool = Query(False, description="Enable debug mode to get additional information")
):
    """
    Send a 'hello' message to a specific Instagram user using POST request.
    """
    return await _send_hello_implementation(username, debug)

@app.get("/")
async def root():
    mode = "MOCK" if USE_MOCK else "LIVE"
    return {
        "message": f"Instagram Hello Sender API ({mode} MODE)", 
        "endpoints": {
            "send_hello": "/send-hello/{username}",
        },
        "api_version": API_VERSION,
        "note": "Instagram requires users to message your business first before you can message them"
    }

@app.get("/received-messages", response_model=List[ReceivedMessage])
async def get_received_messages(debug: bool = Query(False, description="Enable debug mode")):
    """
    Endpoint to display your own received messages.
    """
    if USE_MOCK:
        logger.info("MOCK MODE: Returning simulated received messages")
        return [
            ReceivedMessage(
                sender_username="user123",
                message_text="Hello!",
                timestamp="2025-03-10T15:00:00Z"
            ),
            ReceivedMessage(
                sender_username="user456",
                message_text="Hi there!",
                timestamp="2025-03-10T16:30:00Z"
            )
        ]

    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_BUSINESS_ID:
        raise HTTPException(status_code=500, detail="Instagram credentials not configured correctly.")
    try:
        async with httpx.AsyncClient() as client:
            messages_url = f"https://graph.facebook.com/{API_VERSION}/me/conversations?fields=messages{{from,message,created_time}}&access_token={INSTAGRAM_ACCESS_TOKEN}"

            response = await client.get(messages_url)
            
            if response.status_code != 200:
                detail = response.json()
                print(detail)
                logger.error(f"Failed to fetch received messages: {detail}")
                raise HTTPException(status_code=500, detail="Failed to fetch received messages.")

            data = response.json()
            received_messages = []

            for conversation in data.get('data', []):
                for message in conversation.get('messages', {}).get('data', []):
                    received_messages.append(
                        ReceivedMessage(
                            sender_username=message['from']['username'] if 'username' in message['from'] else "unknown",
                            message_text=message.get('message', ''),
                            timestamp=message.get('created_time')
                        )
                    )
                    

            if debug:
                logger.info(f"Fetched {len(received_messages)} messages")

            return received_messages
    except Exception as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    send_insta("3891652204430815", "Hello from SanaExpert!")