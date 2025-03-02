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


from py_near.account import Account


from nearai.agents.environment import Environment
from utils import AiUtils

# Get the environment from globals
env = globals().get("env")

# Get environment variables
signer_private_key = env.env_vars.get("signer_private_key", None)
# Get encryption key from environment (if needed)
ENCRYPTION_KEY = env.env_vars.get("encryption_key", None)

# We'll initialize these later in the agent function
signer_public_key = None
signer_account_id = None

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

        # Convert the private key from base64 if needed
        private_key_bytes = (
            base64.b64decode(ENCRYPTION_KEY)
            if ENCRYPTION_KEY.startswith("b64:")
            else ENCRYPTION_KEY.encode("utf-8")
        )

        # Ensure the key is the right length for NaCl (32 bytes)
        if len(private_key_bytes) != nacl.public.PrivateKey.SIZE:
            # If not the right length, hash it to get a key of the right length
            import hashlib

            private_key_bytes = hashlib.sha256(private_key_bytes).digest()

        # Create the private key object
        private_key = nacl.public.PrivateKey(private_key_bytes)

        # Get the sender's public key
        sender_public_key_bytes = base64.b64decode(payload["senderPublicKey"])
        sender_public_key = nacl.public.PublicKey(sender_public_key_bytes)

        # Get decryption components
        nonce = base64.b64decode(payload["nonce"])
        encrypted_data = base64.b64decode(payload["encryptedData"])

        # Create a box for decryption
        box = nacl.public.Box(private_key, sender_public_key)

        # Decrypt the data
        decrypted_data = box.decrypt(encrypted_data, nonce=nonce)

        return decrypted_data

    except Exception as e:
        print(f"Error decrypting content: {str(e)}")
        raise


async def verify_task(task_id: str):
    """
    Verifies a task by its ID, retrieves and attempts to decrypt the evidence

    Args:
        task_id: The ID of the task to verify

    Returns:
        Information about the task and the evidence
    """
    try:
        # Get the task from the contract
        contract_id = "commchain.testnet"
        
        # Initialize utils and account info
        utils = AiUtils(env, agent)
        global signer_public_key, signer_account_id
        if not signer_account_id:
            signer_public_key = utils.get_public_key(signer_private_key)
            signer_account_id = utils.get_account_id(signer_public_key)
        
        # Create and initialize the account
        acc = Account(signer_account_id, signer_private_key)
        await acc.startup()
        
        # Get the task
        result = await acc.view_function(
            contract_id, "get_task", {"id": int(task_id)}
        )
        
        # If the task doesn't exist
        if not result:
            return f"Task with ID {task_id} not found"
        
        # If the task is already verified
        if result["status"] == 1:  # 1 = verified
            return f"Task {task_id} is already verified with result: {result['result']}"
        
        # Get the evidence CID
        try:
            evidence = result["evidence"]
            
            # Extract the actual CID from our custom format
            actual_cid = evidence.replace("storj-", "")
            
            # Create the IPFS URL
            ipfs_url = f"https://{actual_cid}.ipfs.w3s.link"
            
            # Return the URL
            return (
                f"Task {task_id} found. Evidence URL: {ipfs_url}\n\n"
                f"To analyze this evidence, you can use the describe_image tool with the URL."
            )
        except Exception as evidence_error:
            return f"Error processing task evidence: {str(evidence_error)}"
    except Exception as e:
        return f"Error verifying task: {str(e)}"


def agent():
    """
    Main agent function that processes user messages and generates responses
    
    Args:
        env_param: The environment object (optional)
    """
    try:
        # Initialize utils and account info
        utils = AiUtils(env, agent)
        global signer_public_key, signer_account_id
        signer_public_key = utils.get_public_key(signer_private_key)
        signer_account_id = utils.get_account_id(signer_public_key)
        
        # Register tools
        tool_registry = env.get_tool_registry()
        tool_registry.register_tool(describe_image)
        tool_registry.register_tool(verify_task)
        
        # Get system prompt
        system_prompt = """
        You are an AI assistant that can help with two main tasks:
        
        1. Image Analysis:
            - Analyze images from URLs
            - Provide detailed descriptions of image content
            - Identify objects, people, text, and other elements in images
        
        2. Task Verification:
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
        response = env.completions_and_run_tools(messages, tools=all_tools)
        
        # Add the response to the chat
        env.add_reply(response)
        
    except Exception as e:
        env.add_reply(f"I encountered an error. Please try again or rephrase your question.")

class CommchainAgent:
    def __init__(self, env):
        self.env = env
        
        # Get environment variables
        self.signer_private_key = self.env.env_vars.get("signer_private_key", None)
        self.encryption_key = self.env.env_vars.get("encryption_key", None)
        
        # Initialize account info
        self.utils = AiUtils(self.env, self.run)
        self.signer_public_key = self.utils.get_public_key(self.signer_private_key)
        self.signer_account_id = self.utils.get_account_id(self.signer_public_key)
        
    def describe_image(self, url):
        """Analyzes an image from a URL and returns a description"""
        try:
            # Download the image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Get the image data and encode as base64
            image_data = response.content
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # Determine image format
            if url.lower().endswith(".png"):
                image_format = "png"
            elif url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
                image_format = "jpeg"
            elif url.lower().endswith(".gif"):
                image_format = "gif"
            elif url.lower().endswith(".webp"):
                image_format = "webp"
            else:
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
            response = self.env.completion(messages, model=model)
            return response
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    async def verify_task(self, task_id):
        """Verifies a task by its ID"""
        try:
            # Get the task from the contract
            contract_id = "commchain.testnet"
            
            # Create and initialize the account
            acc = Account(self.signer_account_id, self.signer_private_key)
            await acc.startup()
            
            # Get the task
            result = await acc.view_function(
                contract_id, "get_task", {"id": int(task_id)}
            )
            
            # If the task doesn't exist
            if not result:
                return f"Task with ID {task_id} not found"
            
            # If the task is already verified
            if result["status"] == 1:  # 1 = verified
                return f"Task {task_id} is already verified with result: {result['result']}"
            
            # Get the evidence CID
            try:
                evidence = result["evidence"]
                
                # Extract the actual CID from our custom format
                actual_cid = evidence.replace("storj-", "")
                
                # Create the IPFS URL
                ipfs_url = f"https://{actual_cid}.ipfs.w3s.link"
                
                # Return the URL
                return (
                    f"Task {task_id} found. Evidence URL: {ipfs_url}\n\n"
                    f"To analyze this evidence, you can use the describe_image tool with the URL."
                )
            except Exception as evidence_error:
                return f"Error processing task evidence: {str(evidence_error)}"
        except Exception as e:
            return f"Error verifying task: {str(e)}"
    
    def run(self):
        try:
            # Register tools
            tool_registry = self.env.get_tool_registry(new=True)
            
            # Register the describe_image tool
            tool_registry.register_tool(self.describe_image)
            
            # Register the verify_task tool
            tool_registry.register_tool(self.verify_task)
            
            # Get system prompt
            system_prompt = """
            You are an AI assistant that can help with two main tasks:
            
            1. Image Analysis:
                - Analyze images from URLs
                - Provide detailed descriptions of image content
                - Identify objects, people, text, and other elements in images
            
            2. Task Verification:
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
            messages = self.env.list_messages()
            
            # Add system prompt to the beginning
            messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Get tool definitions
            all_tools = tool_registry.get_all_tool_definitions()
            
            # Get response with tools
            response = self.env.completions_and_run_tools(messages, tools=all_tools)
            
            # Add the response to the chat
            self.env.add_reply(response)
            
        except Exception as e:
            self.env.add_reply(f"I encountered an error: {str(e)}. Please try again or rephrase your question.")


# Initialize and run the agent
if globals().get('env', None):
    agent = CommchainAgent(globals().get('env'))
    agent.run()