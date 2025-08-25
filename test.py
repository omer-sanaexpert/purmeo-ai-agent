import os
import requests
from dotenv import load_dotenv
load_dotenv() 

def get_instagram_user_from_psid(psid=None, access_token=None):
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
    token = access_token or os.environ.get('INSTAGRAM_ACCESS_TOKEN')
    print("Token:",token)
    print(type(token))
    
    if not token:
        print("❌ No access token provided")
        return None
    psid = "1796744007770755"
    #psid="3891652204430815"
    url = f"https://graph.facebook.com/v21.0/{psid}?fields=username,name&access_token={token}"
    response = requests.get(url)
    print(response.json())
    json_response = response.json()
    
    return {
        'psid': json_response['id'],
        'instagram_id': json_response['username'],
        'name': json_response['name'],
    }

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTA_NEW")

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

#get_instagram_user_from_psid()
send_insta("1796744007770755", "Hello from SanaExpert!")

