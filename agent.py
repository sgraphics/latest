import asyncio
import base64
import requests
import json

# Remove unused imports
# import os
# import sys
# from io import BytesIO

import nacl.secret
import nacl.public
import nacl.encoding
import nacl.utils

import base58
import ed25519

from py_near.account import Account


from nearai.agents.environment import Environment

# Get the environment from globals
env = globals().get("env")

# Get environment variables
signer_private_key = env.env_vars.get("signer_private_key", None)
# Get encryption key from environment (if needed)
ENCRYPTION_KEY = env.env_vars.get("encryption_key", None)

def describe_image(url: str):
    """
    Analyzes an image from a URL and returns a text description of what is in the picture.

    Args:
        url: A string URL pointing to an image

    Returns:
        A text description of the image content
    """
    try:
        # Download the image from the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the image data
        image_data = response.content
        
        # Encode the image as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Determine image format from URL
        if url.lower().endswith(".png"):
            image_format = "png"
        elif url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
            image_format = "jpeg"
        elif url.lower().endswith(".gif"):
            image_format = "gif"
        elif url.lower().endswith(".webp"):
            image_format = "webp"
        else:
            # Default to png if format can't be determined
            image_format = "png"
        
        # Create the message with the image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What do you see? "
                        "Include all relevant details about objects, people, "
                        "settings, colors, and any text visible in the image.",
                    },
                ],
            }
        ]
        
        # Use the vision model to analyze the image
        model = "phi-3-vision-128k-instruct"
        response = env.completion(messages, model=model)
        return response
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def describe_image(image_data: bytes):
    """
    Analyzes an image from a URL and returns a text description of what is in the picture.

    Args:
        image_data: The raw image data as bytes

    Returns:
        A text description of the image content
    """
    try:
        # Encode the image as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        image_format = "jpg"
        
        # Create the message with the image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What do you see? "
                        "Include all relevant details about objects, people, "
                        "settings, colors, and any text visible in the image.",
                    },
                ],
            }
        ]
        
        # Use the vision model to analyze the image
        model = "phi-3-vision-128k-instruct"
        response = env.completion(messages, model=model)
        return response
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def retrieve_from_ipfs(cid: str):
    """
    Retrieve a file from IPFS by CID

    Args:
        cid: The CID of the file to retrieve

    Returns:
        The file data as bytes
    """
    try:
        # Construct the IPFS gateway URL
        gateway_url = f"https://{cid}.ipfs.w3s.link"
        
        # Fetch the file
        response = requests.get(gateway_url, timeout=10)

        if not response.ok:
            error_msg = (
                f"Failed to retrieve file: {response.status_code} {response.reason}"
            )
            raise Exception(error_msg)

        # Get the content
        content = response.content
        return content
    except Exception as e:
        raise


def decrypt_with_nacl(encrypted_data_json):
    """
    Decrypt data using NaCl's Box (asymmetric encryption)

    Args:
        encrypted_data_json: JSON string containing encrypted data

    Returns:
        Decrypted data as bytes
    """
    try:
        # Parse the JSON payload
        payload = json.loads(encrypted_data_json)

        # Check if we have all required fields
        if (
            not payload.get("nonce")
            or not payload.get("encryptedData")
            or not payload.get("senderPublicKey")
        ):
            raise Exception(
                "Invalid payload format - missing required fields for asymmetric decryption"
            )

        # Get the private key from environment
        if not ENCRYPTION_KEY:
            raise Exception("Private key not found in environment variables")

        # Convert the private key from base64, removing the 'b64:' prefix if present
        private_key_base64 = ENCRYPTION_KEY.replace('b64:', '') if ENCRYPTION_KEY.startswith('b64:') else ENCRYPTION_KEY
        try:
            private_key_bytes = base64.b64decode(private_key_base64)
        except Exception as e:
            raise Exception(f"Failed to decode private key: {str(e)}")

        # Create the private key object
        try:
            private_key = nacl.public.PrivateKey(private_key_bytes)
        except Exception as e:
            raise Exception(f"Failed to create private key object: {str(e)}")

        # Get the sender's public key
        try:
            sender_public_key_bytes = base64.b64decode(payload["senderPublicKey"])
            sender_public_key = nacl.public.PublicKey(sender_public_key_bytes)
        except Exception as e:
            raise Exception(f"Failed to decode sender's public key: {str(e)}")

        # Get decryption components
        try:
            nonce = base64.b64decode(payload["nonce"])
            encrypted_data = base64.b64decode(payload["encryptedData"])
        except Exception as e:
            raise Exception(f"Failed to decode nonce or encrypted data: {str(e)}")

        # Create a box for decryption
        box = nacl.public.Box(private_key, sender_public_key)

        # Decrypt the data
        try:
            decrypted_data = box.decrypt(encrypted_data, nonce=nonce)
        except Exception as e:
            raise Exception(f"Decryption failed: {str(e)}")

        return decrypted_data

    except Exception as e:
        env.add_system_log(f"Error in decrypt_with_nacl: {str(e)}")
        raise

async def verify_task(task_id: str):
    """
    Verifies a task by its ID, retrieves and attempts to decrypt the evidence

    Args:
        task_id: The ID of the task to verify

    Returns:
        Information about the task and the evidence
    """
    # Get the task from the contract
    contract_id = "commchain.testnet"
    
    # Create and initialize the account
    acc = Account(account_id=signer_account_id, private_key=signer_private_key, rpc_addr="https://rpc.testnet.pagoda.co")
    await acc.startup()
    
    # Get the task
    view_result = await acc.view_function(
        contract_id, "get_task", {"id": int(task_id)}
    )
    # Extract the actual result from the ViewFunctionResult object
    # According to the models.py file, ViewFunctionResult has a 'result' attribute
    if not view_result:
        return f"Task with ID {task_id} not found - view_result is empty"
    
    # Access the result attribute directly since it's an object, not a dictionary
    result = view_result.result
    
    # If the task doesn't exist
    if not result:
        return f"Task with ID {task_id} not found"
    
    # If the task is already verified
    if result["status"] == 1:  # 1 = verified
        return f"Task {task_id} is already verified with result: {result['result']}"
    
    evidence = result["evidence"]
    
    # Extract the actual CID from our custom format
    actual_cid = evidence.replace("storj-", "")
    
    # Create the IPFS URL
    ipfs_url = f"https://{actual_cid}.ipfs.w3s.link"

    
    # Fetch the file
    response = requests.get(ipfs_url, timeout=10)

    if not response.ok:
        error_msg = (
            f"Failed to retrieve file: {response.status_code} {response.reason}"
        )
        raise Exception(error_msg)

    image_data = decrypt_with_nacl(response.content)
    
    description = describe_image(image_data)
    # Return the URL
    return (
        f"Task {task_id} found. Data: {description}"
        f"To analyze this evidence, you can use the describe_image tool with the URL."
    )

def get_account_id(public_key):
    url = f"https://test.api.fastnear.com/v0/public_key/{public_key}"
    response = requests.get(url)
    response.raise_for_status()
    content = response.json()
    account_ids = content.get("account_ids", [])

    if len(account_ids):
        return account_ids[0]
    else:
        return None

def get_public_key(extended_private_key):
    private_key_base58 = extended_private_key.replace("ed25519:", "")

    decoded = base58.b58decode(private_key_base58)
    secret_key = decoded[:32]

    signing_key = ed25519.SigningKey(secret_key)
    verifying_key = signing_key.get_verifying_key()

    base58_public_key = base58.b58encode(verifying_key.to_bytes()).decode()

    return base58_public_key


# Add this wrapper function for verify_task
def verify_task_sync(task_id: str):
    """
    Synchronous wrapper for the async verify_task function
    
    Args:
        task_id: The ID of the task to verify
        
    Returns:
        Information about the task and the evidence
    """
    # Use asyncio.run to run the async function in a synchronous context
    return asyncio.run(verify_task(task_id))

def agent(env: Environment):
    """
    Main agent function that processes user messages and generates responses
    
    Args:
        env_param: The environment object (optional)
    """
    try:
        
        # Register tools
        tool_registry = env.get_tool_registry()
        tool_registry.register_tool(verify_task_sync)
        
        # Get system prompt
        system_prompt = """
        You are an AI assistant that can help with two main tasks:
        
        1. Task Verification:
            - Extract evidence URLs from tasks stored on the NEAR blockchain
            - Verify if tasks have been completed
            - Analyze the evidence using the describe_image tool
        
        When a user sends you an image URL or asks about analyzing an image, 
        use the describe_image tool.
        
        When a user asks you to verify a task, use the verify_task tool which 
        will retrieve the task and provide the evidence URL.
        
        After getting the evidence URL from verify_task, you should use the 
        describe_image tool to analyze the evidence.
        
        Examples of how you can help:
        - Describe what's in an image
        - Verify if an image contains specific content
        - Verify tasks and analyze their evidence
        - Provide detailed descriptions of task evidence
        
        Always be helpful, accurate, and respectful in your responses.
        """
        
        # Get all messages
        messages = env.list_messages()
        
        # Add system prompt to the beginning
        messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Get tool definitions
        all_tools = tool_registry.get_all_tool_definitions()
        
        # Get response with tools
        env.completion_and_run_tools(messages, tools=all_tools)
        
        # Request user input for the next interaction
        env.request_user_input()
        
    except Exception as e:
        env.add_reply(f"I encountered an error. Please try again or rephrase your question {str(e)}.")



signer_public_key = get_public_key(signer_private_key)
signer_account_id = get_account_id(signer_public_key)
agent(env)
