from pydantic import BaseModel
import requests
import json
import uuid
import os
from dotenv import load_dotenv
from typing import Dict, Tuple, Optional

# Load environment variables
load_dotenv()

class BrowserInfo(BaseModel):
    browser_family: str
    browser_version: Optional[str]
    os_family: str
    os_version: Optional[str]
    device_family: str
    device_brand: Optional[str]
    device_model: Optional[str]
    is_mobile: bool
    is_tablet: bool
    is_desktop: bool
    is_bot: bool
    raw_user_agent: str

class LocationInfo(BaseModel):
    country_code: Optional[str]
    country_name: Optional[str]
    city: Optional[str]
    postal_code: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    continent: Optional[str]
    subdivision: Optional[str]
    accuracy_radius: Optional[int]

class RequestInfo(BaseModel):
    # Previous request fields...
    method: str
    url: str
    base_url: str
    path: str
    headers: Dict[str, str]
    client_host: Optional[str]
    
    # New fields for browser and location
    browser_info: Optional[BrowserInfo]
    location_info: Optional[LocationInfo]

class ZendeskTicketManager:
    def __init__(self):
        """Initialize Zendesk configuration from environment variables."""
        self.subdomain = os.environ.get("ZENDESK_SUBDOMAIN")
        self.admin_email = os.environ.get("ZENDESK_EMAIL")
        self.api_token = os.environ.get("ZENDESK_API_TOKEN")
        
        if not all([self.subdomain, self.admin_email, self.api_token]):
            raise ValueError("Missing required environment variables")
        
        self.base_url = f"https://{self.subdomain}.zendesk.com"
        self.auth = (f"{self.admin_email}/token", self.api_token)
        self.headers = {"Content-Type": "application/json"}
        self.current_ticket_id = None
    
    def add_tags(self, ticket_id: int, tag: str ="", additional_tag: str ="") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "tags": ["ticket_by_ai",tag,additional_tag]
            }
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
    
    def add_tags_purmeo(self, ticket_id: int, tag: str ="", additional_tag: str ="" , additional_2 : str = "") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "tags": ["ticket_by_ai",tag,additional_tag, additional_2]
            }
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
        
    def add_multiple_comments(self, ticket_id: int, user_message: str, assistant_message: str, requester_id: str, assistant_id: str = "32601040249617") -> bool:
        """
        Add both user and assistant messages to a Zendesk ticket in a single API call.

        Args:
            ticket_id (int): The ID of the Zendesk ticket.
            user_message (str): The message from the user.
            assistant_message (str): The response from the assistant.
            requester_id (str): The user’s Zendesk requester ID.
            assistant_id (str): The Zendesk ID for the assistant (default: "32601040249617").

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        comment_data = {
            "ticket": {
                "comment": [
                    {
                        "body": user_message,
                        "public": True,
                        "author_id": requester_id
                    },
                    {
                        "body": assistant_message,
                        "public": True,
                        "author_id": assistant_id
                    }
                ]
            }
        }

        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=comment_data
            )
            response.raise_for_status()
            print(f"Successfully added user and assistant messages to ticket {ticket_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error adding multiple comments: {str(e)}")
            return False

    def create_anonymous_ticket(self, message: str, country:str) -> Tuple[Optional[int], Optional[int]]:
        """
        Create a temporary anonymous ticket in Zendesk with request metadata.
        
        Args:
            message (str): Initial ticket message
            request_info (RequestInfo): Request metadata including browser and location info
            
        Returns:
            Tuple[Optional[int], Optional[int]]: Tuple of (requester_id, ticket_id)
        """
        temp_email = f"anonymous_{uuid.uuid4().hex}@temporary.com"
        temp_name = "Anonymous User"
        ticket_data = {}
        if country == "ES":
            ticket_data = {
                "request": {
                    "subject": "Support Request from Shopify ES",
                    "comment": {
                        "body": "Hola, soy María de SanaExpert. 🌿 ¿Cómo puedo ayudarte hoy?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        elif country == "IT":
            ticket_data = {
                "request": {
                    "subject": "Support Request from Shopify IT",
                    "comment": {
                        "body": "Ciao, sono Maria di SanaExpert. 🌿 Come posso aiutarti oggi?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        elif country == "DE":
            ticket_data = {
                "request": {
                    "subject": "Support Request from Shopify DE",
                    "comment": {
                        "body": "Hallo, ich bin Maria von SanaExpert. 🌿 Kommt das auch?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        else:
            ticket_data = {
                "request": {
                    "subject": "Support Request from Shopify",
                    "comment": {
                        "body": "Hello, I'm Maria from SanaExpert. 🌿 How can I help you today?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        try:
            response = requests.post(
                f"{self.base_url}/api/v2/requests",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            
            ticket = response.json()["request"]

            self.add_tags(ticket['id'], "not_escalated", "shopify_"+country.lower())
            #update the ticket status to pending
            # ticket_data = {
            #         "ticket": {
            #             "status": "pending",
            #             "tags": ["ticket_by_ai"]
            #         }
            #     }
            # response = requests.put(
            #     f"{self.base_url}/api/v2/tickets/{ticket['id']}",
            #     auth=self.auth,
            #     headers=self.headers,
            #     json=ticket_data
            # )
            # response.raise_for_status()
            
            print(ticket)
            self.current_ticket_id = ticket['id']
            
            return ticket["requester_id"], ticket["id"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating ticket: {str(e)}")
            return None, None
        

    def create_anonymous_ticket_purmeo(self, message: str, country:str) -> Tuple[Optional[int], Optional[int]]:
            """
            Create a temporary anonymous ticket in Zendesk with request metadata.
            
            Args:
                message (str): Initial ticket message
                request_info (RequestInfo): Request metadata including browser and location info
                
            Returns:
                Tuple[Optional[int], Optional[int]]: Tuple of (requester_id, ticket_id)
            """
            temp_email = f"anonymous_{uuid.uuid4().hex}@temporary.com"
            temp_name = "Anonymous User"
            ticket_data = {}
            if country == "ES":
                ticket_data = {
                    "request": {
                        "subject": "Support Request from Shopify ES",
                        "comment": {
                            "body": "Hola, soy María de SanaExpert. 🌿 ¿Cómo puedo ayudarte hoy?",
                            "author_id": "32601040249617",
                            "public": False,
                        },
                        "requester": {
                            "name": temp_name
                        },
                    }
                }
            elif country == "IT":
                ticket_data = {
                    "request": {
                        "subject": "Support Request from Shopify IT",
                        "comment": {
                            "body": "Ciao, sono Maria di SanaExpert. 🌿 Come posso aiutarti oggi?",
                            "author_id": "32601040249617",
                            "public": False,
                        },
                        "requester": {
                            "name": temp_name
                        },
                    }
                }
            elif country == "DE":
                ticket_data = {
                    "request": {
                        "subject": "Support Request from Shopify Purmeo DE",
                        "comment": {
                            "body": "Hallo, ich bin Maria von Purmeo. 🌿 Kommt das auch?",
                            "author_id": "32601040249617",
                            "public": False,
                        },
                        "requester": {
                            "name": temp_name
                        },
                    }
                }
            else:
                ticket_data = {
                    "request": {
                        "subject": "Support Request from Shopify Purmeo",
                        "comment": {
                            "body": "Hello, I'm Maria from SanaExpert. 🌿 How can I help you today?",
                            "author_id": "32601040249617",
                            "public": False,
                        },
                        "requester": {
                            "name": temp_name
                        },
                    }
                }
            try:
                response = requests.post(
                    f"{self.base_url}/api/v2/requests",
                    auth=self.auth,
                    headers=self.headers,
                    json=ticket_data
                )
                response.raise_for_status()
                
                ticket = response.json()["request"]

                self.add_tags_purmeo(ticket['id'], "not_escalated","purmeo", "shopify_"+country.lower())
                #update the ticket status to pending
                # ticket_data = {
                #         "ticket": {
                #             "status": "pending",
                #             "tags": ["ticket_by_ai"]
                #         }
                #     }
                # response = requests.put(
                #     f"{self.base_url}/api/v2/tickets/{ticket['id']}",
                #     auth=self.auth,
                #     headers=self.headers,
                #     json=ticket_data
                # )
                # response.raise_for_status()
                
                print(ticket)
                self.current_ticket_id = ticket['id']
                
                return ticket["requester_id"], ticket["id"]
                
            except requests.exceptions.RequestException as e:
                print(f"Error creating ticket: {str(e)}")
                return None, None
        
    def create_instagram_ticket(self, message: str, name: str,id:int, country:str) -> Tuple[Optional[int], Optional[int]]:
        """
        Create a instagram ticket in Zendesk with request metadata.
        
        Args:
            message (str): Initial ticket message
            request_info (RequestInfo): Request metadata including browser and location info
            
        Returns:
            Tuple[Optional[int], Optional[int]]: Tuple of (requester_id, ticket_id)
        """
        temp_name = name
        ticket_data = {}
        if country == "ES":
            ticket_data = {
                "request": {
                    "subject": "Instagram Support Request from ES",
                    "comment": {
                        "body": "Hola, soy María de SanaExpert. 🌿 ¿Cómo puedo ayudarte hoy?",
                        "author_id": 32601040249617,
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name,
                        "id": id
                    },
                    "group_id":25871199760273
                }
            }
        elif country == "IT":
            ticket_data = {
                "request": {
                    "subject": "Instagram Support Request from IT",
                    "comment": {
                        "body": "Ciao, sono Maria di SanaExpert. 🌿 Come posso aiutarti oggi?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                    "group_id":25871203593233
                }
            }
        elif country == "DE":
            ticket_data = {
                "request": {
                    "subject": "Instagram Support Request from DE",
                    "comment": {
                        "body": "Hallo, ich bin Maria von SanaExpert. 🌿 Wie kann ich Ihnen heute helfen?",
                        "author_id": 32601040249617,
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name,
                        "id": id
                    },
                    "group_id":25871199760273
                }
            }

        else:
            ticket_data = {
                "request": {
                    "subject": "Instagram Support Request from Shopify",
                    "comment": {
                        "body": "Hello, I'm Maria from SanaExpert. 🌿 How can I help you today?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        try:
            response = requests.post(
                f"{self.base_url}/api/v2/requests",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            
            ticket = response.json()["request"]

            print(ticket)
            self.current_ticket_id = ticket['id']
            self.add_tags(ticket['id'], "not_escalated", "instagram_"+country.lower())
            
            return ticket["requester_id"], ticket["id"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating ticket: {str(e)}")
            return None, None
    def update_user_phone_number(self, user_id: int, phone_number: str) -> bool:
        """
        Update the phone number of a Zendesk user.

        Args:
            user_id (int): The ID of the Zendesk user.
            phone_number (str): The new phone number to set.

        Returns:
            bool: True if the update is successful, False otherwise.
        """
        user_data = {
            "user": {
                "phone": phone_number
            }
        }

        try:
            response = requests.put(
                f"{self.base_url}/api/v2/users/{user_id}.json",
                auth=self.auth,
                headers=self.headers,
                json=user_data
            )
            response.raise_for_status()
            print(f"Successfully updated phone number for user {user_id}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error updating phone number: {str(e)}")
            return False


    def create_whatsapp_ticket(self, message: str, name: str,id:int, country:str) -> Tuple[Optional[int], Optional[int]]:
        """
        Create a instagram ticket in Zendesk with request metadata.
        
        Args:
            message (str): Initial ticket message
            request_info (RequestInfo): Request metadata including browser and location info
            
        Returns:
            Tuple[Optional[int], Optional[int]]: Tuple of (requester_id, ticket_id)
        """
        temp_name = name
        ticket_data = {}
        if country == "ES":
            ticket_data = {
                "request": {
                    "subject": "Whatsapp Support Request from ES",
                    "comment": {
                        "body": "Hola, soy María de SanaExpert. 🌿 ¿Cómo puedo ayudarte hoy?",
                        "author_id": 32601040249617,
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name,
                        "id": id
                    },
                    "group_id":25871199760273
                }
            }
        elif country == "IT":
            ticket_data = {
                "request": {
                    "subject": "Whatsapp Support Request from IT",
                    "comment": {
                        "body": "Ciao, sono Maria di SanaExpert. 🌿 Come posso aiutarti oggi?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                    "group_id":25871203593233
                }
            }
        else:
            ticket_data = {
                "request": {
                    "subject": "Whatsapp Support Request from Shopify",
                    "comment": {
                        "body": "Hello, I'm Maria from SanaExpert. 🌿 How can I help you today?",
                        "author_id": "32601040249617",
                        "public": False,
                    },
                    "requester": {
                        "name": temp_name
                    },
                }
            }
        try:
            response = requests.post(
                f"{self.base_url}/api/v2/requests",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            
            ticket = response.json()["request"]

            #updated the customer phone number
            #from_wa_id
            self.update_user_phone_number(ticket["requester_id"],id)
            print(ticket)
            self.current_ticket_id = ticket['id']
            self.add_tags(ticket['id'], "not_escalated", "whatsapp"+country.lower())
            
            return ticket["requester_id"], ticket["id"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating ticket: {str(e)}")
            return None, None

    def add_tag_to_ticket(self, ticket_id, tag):
        """
        Add a tag to a ticket in Zendesk.
        
        Args:
            ticket_id: The ID of the ticket to update
            tag: The tag to add to the ticket
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, get the current ticket data
            response = requests.get(
                f"{self.base_url}/api/v2/tickets/{ticket_id}.json",
                headers=self.headers,
                auth=self.auth,
            )
            if response.status_code != 200:
                print(f"Failed to get ticket data: {response.status_code} {response.text}")
                return False
                
            ticket_data = response.json()['ticket']
            
            # Get current tags or initialize empty list
            current_tags = ticket_data.get('tags', [])
            
            # Add the new tag if it's not already present
            if tag not in current_tags:
                current_tags.append(tag)
            
            # Update the ticket with the new tags
            update_data = {
                "ticket": {
                    "tags": current_tags
                }
            }
            
            update_response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}.json",
                json=update_data,
                headers=self.headers,
                auth=self.auth,
            )
            
            if update_response.status_code == 200:
                print(f"Tag '{tag}' added to ticket {ticket_id}")
                return True
            else:
                print(f"Failed to add tag: {update_response.status_code} {update_response.text}")
                return False
                
        except Exception as e:
            print(f"Error adding tag to ticket: {str(e)}")
            return False

    def update_ticket_status(self, ticket_id: int, new_status: str = "open", tag: str ="", assignee_id: str="",additional_tag: str ="") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "status": new_status,
                "tags": ["ticket_by_ai","escalated_by_ai","ai_shopify",tag,additional_tag]
            }
        }
        #assign to , clara
        ticket_data["ticket"]["assignee_id"] = assignee_id
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
        
    def update_ticket_status_ig(self, ticket_id: int, new_status: str = "open", tag: str ="", assignee_id: str="",additional_tag: str ="") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "status": new_status,
                "tags": ["ticket_by_ai","escalated_by_ai","ai_instagram",tag,additional_tag]
            }
        }
        #assign to , clara
        ticket_data["ticket"]["assignee_id"] = assignee_id
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
    
    def update_ticket_status_whatsapp(self, ticket_id: int, new_status: str = "open", tag: str ="", assignee_id: str="",additional_tag: str ="") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "status": new_status,
                "tags": ["ticket_by_ai","escalated_by_ai","ai_whatsapp",tag,additional_tag]
            }
        }
        #assign to , clara
        ticket_data["ticket"]["assignee_id"] = assignee_id
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
    
    

    def update_ticket_status_email(self, ticket_id: int, new_status: str = "open", tag: str ="", assignee_id: str="",additional_tag: str ="") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "status": new_status,
                "tags": ["ticket_by_ai","escalated_by_ai","ai_email",tag,additional_tag]
            }
        }
        #assign to , clara
        ticket_data["ticket"]["assignee_id"] = assignee_id
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
        
    def assign_ticket(self, ticket_id: int, requester_id: int) -> bool:
        """
        Assign a ticket to a specific requester.
        
        Args:
            ticket_id (int): ID of the ticket to update
            requester_id (int): ID of the requester to assign the ticket to
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "requester_id": requester_id
            }
        }

        #TODO modify the ticket to assign the comments to the requester as well.
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            print(f"Successfully assigned ticket {ticket_id} to requester {requester_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error assigning ticket: {str(e)}")
            return False

    def update_user_details(self, requester_id: int,ticket_id:int, new_email: str, new_name: str, summary:str, assignee_id:str,additional_tag:str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            # Check if user already exists
            search_response = requests.get(
                f"{self.base_url}/api/v2/users/search?query={new_email}",
                auth=self.auth,
                headers=self.headers
            )
            search_response.raise_for_status()
            
            existing_users = search_response.json().get("users", [])
            print("tags",summary)
            if existing_users:
                existing_user = existing_users[0]
                print(f"User found: {existing_user['name']} ({existing_user['email']})")
                
                # If we have a current ticket, update its status 

                ### update the requester_id to existing ticket
                
                self.assign_ticket(ticket_id, existing_user['id'])
                #self.add_tag_to_ticket(ticket_id, summary)
                
                self.update_ticket_status(ticket_id, "open",summary,assignee_id,additional_tag)
                return True
            
            # Update user details if no existing user found
            user_data = {
                "user": {
                    "email": new_email,
                    "name": new_name
                }
            }
            
            update_response = requests.put(
                f"{self.base_url}/api/v2/users/{requester_id}",
                auth=self.auth,
                headers=self.headers,
                json=user_data
            )
            print("update response",update_response.json())
            update_response.raise_for_status()
            self.update_ticket_status(ticket_id, "open",summary,assignee_id,additional_tag)
            #self.add_tag_to_ticket(ticket_id, summary)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False

    def update_user_details_ig(self, requester_id: int,ticket_id:int, summary:str, assignee_id:str,additional_tag:str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            self.update_ticket_status_ig(ticket_id, "open",summary,assignee_id,additional_tag)
            #self.add_tag_to_ticket(ticket_id, summary)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False
    
    def update_user_details_whatsapp(self, requester_id: int,ticket_id:int, summary:str, assignee_id:str,additional_tag:str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            self.update_ticket_status_whatsapp(ticket_id, "open",summary,assignee_id,additional_tag)
            #self.add_tag_to_ticket(ticket_id, summary)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False
        
    def update_user_details_email(self, requester_id: int,ticket_id:int, summary:str, assignee_id:str,additional_tag:str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            self.update_ticket_status_email(ticket_id, "open",summary,assignee_id,additional_tag)
            #self.add_tag_to_ticket(ticket_id, summary)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False


    def update_user_information(self, requester_id: int,ticket_id:int, new_email: str, name:str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            # Check if user already exists
            search_response = requests.get(
                f"{self.base_url}/api/v2/users/search?query={new_email}",
                auth=self.auth,
                headers=self.headers
            )
            search_response.raise_for_status()
            
            existing_users = search_response.json().get("users", [])
            
            if existing_users:
                existing_user = existing_users[0]
                print(f"User found: {existing_user['name']} ({existing_user['email']})")
                
                # If we have a current ticket, update its status 

                ### update the requester_id to existing ticket
                
                self.assign_ticket(ticket_id, existing_user['id'])
                #self.add_tag_to_ticket(ticket_id, summary)
                return True
            else:
                # Update user details if no existing user found
                user_data = {
                    "user": {
                        "email": new_email,
                        "name": name
                    }
                }
                
                update_response = requests.put(
                    f"{self.base_url}/api/v2/users/{requester_id}",
                    auth=self.auth,
                    headers=self.headers,
                    json=user_data
                )
                print("update response",update_response.json())
                update_response.raise_for_status()
                #self.add_tag_to_ticket(ticket_id, summary)
                return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False
        
    def add_public_comment(self, ticket_id: int, comment: str, requester_id: str) -> bool:
        """
        Add a public comment to a Zendesk ticket.
        
        Args:
            ticket_id (int): The ID of the ticket to update.
            comment (str): The comment text.
            requester_id (str): ID of the author making the comment.
        
        Returns:
            bool: Success status of the comment addition.
        """
        comment_data = {
            "ticket": {
                "comment": {
                    "body": comment,
                    "public": True,  # Changed to True for proper sequencing
                    "author_id": requester_id
                },
            }
        }
        #31549253490321
        # Only override author_id if it's the bot ID
        # if requester_id == "32601040249617":
        #     # This ensures bot comments are properly identified
        #     pass  # Author ID is already set above
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=comment_data
            )
            response.raise_for_status()
            print(f"Successfully added public comment to ticket {ticket_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error adding public comment: {str(e)}")
            return False
    
    def add_public_comment_purmeo(self, ticket_id: int, comment: str, requester_id: str) -> bool:
        """
        Add a public comment to a Zendesk ticket.
        
        Args:
            ticket_id (int): The ID of the ticket to update.
            comment (str): The comment text.
            requester_id (str): ID of the author making the comment.
        
        Returns:
            bool: Success status of the comment addition.
        """
        comment_data = {
            "ticket": {
                "comment": {
                    "body": comment,
                    "public": True,  # Changed to True for proper sequencing
                    "author_id": requester_id
                },
            }
        }
        #31549253490321
        # Only override author_id if it's the bot ID
        # if requester_id == "32601040249617":
        #     # This ensures bot comments are properly identified
        #     pass  # Author ID is already set above
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=comment_data
            )
            response.raise_for_status()
            print(f"Successfully added public comment to ticket {ticket_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error adding public comment: {str(e)}")
            return False

    def update_instagram_fields(self, requester_id: int, instagram_id: str, instagram_username: str) -> bool:
        """
        Update a Zendesk user's custom fields with Instagram ID and username.

        Args:
            requester_id (int): Zendesk user ID
            instagram_id (str): Instagram numeric ID
            instagram_username (str): Instagram username (handle)

        Returns:
            bool: Success status of the update
        """
        user_data = {
            "user": {
                "user_fields": {
                    "instagram_id": instagram_id,
                    "instagram_user_name": instagram_username
                }
            }
        }

        try:
            response = requests.put(
                f"{self.base_url}/api/v2/users/{requester_id}.json",
                auth=self.auth,
                headers=self.headers,
                json=user_data
            )
            response.raise_for_status()
            print(f"Successfully updated Instagram fields for user {requester_id}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error updating Instagram fields: {str(e)}")
            return False
        

    def mark_ticket_as_not_relevant(self, ticket_id: int) -> bool:
        """
        Mark the ticket as Not Relevant by updating its status and adding a 'not_relevant' tag.
        
        Args:
            ticket_id (int): ID of the ticket to update.
        
        Returns:
            bool: Success status of the update.
        """
        ticket_data = {
            "ticket": {
                "status": "new",  # Zendesk statuses are usually: new, open, pending, hold, solved, closed
                "tags": ["ticket_by_email", "not_relevant"]
            }
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}.json",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            print(f"Ticket {ticket_id} successfully marked as Not Relevant")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error marking ticket as Not Relevant: {str(e)}")
            return False
