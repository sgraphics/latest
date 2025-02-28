import asyncio
import enum
import base64
import os
import re
import requests
from typing import Optional, List, Tuple
from urllib.parse import urlparse, unquote

from nearai.agents.environment import Environment
from py_near.account import Account
from py_near.dapps.core import NEAR
from nearai.shared.inference_client import InferenceClient

from utils import AiUtils, State

# load user's private key
signer_private_key = globals()['env'].env_vars.get("signer_private_key", None)

utils = AiUtils(env, agent)


def is_valid_image_url(url: str) -> bool:
    """Check if a URL is likely to be a valid image URL."""
    try:
        parsed = urlparse(url)
        # Check if URL has a scheme and netloc
        if not (parsed.scheme and parsed.netloc):
            return False
        
        # Check if URL ends with common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in image_extensions):
            return True
        
        # If no extension, make a HEAD request to check content type
        try:
            response = requests.head(url, timeout=5)
            content_type = response.headers.get('Content-Type', '')
            return content_type.startswith('image/')
        except:
            # If HEAD request fails, try a small GET request
            response = requests.get(url, timeout=5, stream=True)
            content_type = response.headers.get('Content-Type', '')
            response.close()  # Close the connection without downloading the full content
            return content_type.startswith('image/')
    except:
        return False


def get_actual_wikipedia_image_url(wiki_url: str) -> Optional[str]:
    """Extract the actual image URL from a Wikipedia image link."""
    try:
        # Handle Wikipedia image URLs
        if 'wikipedia.org' in wiki_url and '/media/File:' in wiki_url:
            # Get the filename from the URL
            filename = wiki_url.split('File:')[-1]
            filename = unquote(filename)  # Handle URL encoding
            
            # Use the Wikipedia API instead of scraping the page
            # This avoids loading the entire HTML which can be very large
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles=File:{filename}&prop=imageinfo&iiprop=url&format=json"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                # Get the first page (there should only be one)
                for page_id in pages:
                    image_info = pages[page_id].get('imageinfo', [])
                    if image_info:
                        return image_info[0].get('url')
        
        return wiki_url
    except Exception as e:
        print(f"Error processing Wikipedia URL: {e}")
        return wiki_url


def download_image_from_url(url: str, save_path: str) -> bool:
    """Download an image from a URL and save it to the specified path."""
    try:
        # Log the URL being processed
        env.add_agent_log(f"Attempting to download image from: {url}")
        
        # Make a direct request to the URL
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            
            # Verify this is actually an image
            if content_type.startswith('image/'):
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                env.add_agent_log(f"Successfully downloaded image ({len(response.content)} bytes)")
                return True
            else:
                env.add_agent_log(f"URL returned non-image content type: {content_type}")
                return False
        else:
            env.add_agent_log(f"Failed to download image. Status code: {response.status_code}")
            return False
    except Exception as e:
        env.add_agent_log(f"Error downloading image: {e}")
        return False


def extract_image_urls(text: str) -> List[str]:
    """Extract image URLs from text."""
    # Simple URL regex pattern for direct image links
    url_pattern = r'https?://\S+\.(jpg|jpeg|png|gif|webp|bmp)'
    direct_urls = re.findall(url_pattern, text, re.IGNORECASE)
    
    # More comprehensive URL pattern that might catch other image URLs
    general_url_pattern = r'https?://\S+'
    potential_urls = re.findall(general_url_pattern, text)
    
    # Combine and filter to only include valid image URLs
    image_urls = []
    
    # First add URLs that end with image extensions
    for url in potential_urls:
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
            image_urls.append(url)
    
    # Then check other URLs if they might be images
    for url in potential_urls:
        if url not in image_urls and is_valid_image_url(url):
            image_urls.append(url)
    
    return image_urls


async def get_image_description(env: Environment, image_path: str) -> str:
    """Get a description of an image using a vision-capable model."""
    try:
        env.add_agent_log("Uploading image as file attachment")
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
        
        # Get the filename from the path
        filename = os.path.basename(image_path)
        
        # Upload the file to the thread
        file_obj = env.write_file(
            filename=filename,
            content=image_content,
            filetype="image/png",
            write_to_disk=True
        )
        
        # First, add the image to the thread with our question
        user_message = env.add_message(
            "user",
            "Please describe this image in detail. What do you see?",
            attachments=[{"file_id": file_obj.id, "tools": []}]
        )
        
        # Now get all messages including our image message
        thread_messages = env.list_messages(limit=10, order="desc")
        
        # Format them for the completion call
        formatted_messages = []
        formatted_messages.append({"role": "system", "content": "You are a helpful assistant that can analyze images in detail."})
        
        # Add the most recent messages in chronological order
        for msg in reversed(thread_messages[:5]):
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Make the completion call with these messages
        try:
            env.add_agent_log(f"Calling vision model with {len(formatted_messages)} messages")
            response = env.completion(
                formatted_messages,
                model="llama-3p2-11b-vision-instruct"
            )
            return response
        except Exception as e:
            env.add_agent_log(f"Error with first vision model: {e}")
            try:
                env.add_agent_log("Attempting to use llama-3p2-90b-vision-instruct model")
                response = env.completion(
                    formatted_messages,
                    model="llama-3p2-90b-vision-instruct"
                )
                return response
            except Exception as e2:
                env.add_agent_log(f"Error with fallback vision model: {e2}")
                return f"I couldn't analyze this image due to technical limitations. The image might be too large or in an unsupported format."
    except Exception as e:
        env.add_agent_log(f"Error in image processing: {e}")
        return f"I couldn't analyze this image due to technical limitations."


async def process_image_from_url_or_attachment(env: Environment) -> Tuple[Optional[str], Optional[str]]:
    """Process an image from either a URL in the message or an attachment."""
    image_description = None
    image_source = None
    latest_messages = env.list_messages(limit=5, order="desc")
    
    for message in latest_messages:
        if message.get("role") == "user":
            # First check for image URLs in the message content
            message_content = message.get("content", "")
            image_urls = extract_image_urls(message_content)
            
            if image_urls:
                # Use the first valid image URL
                for url in image_urls:
                    temp_image_path = os.path.join(env.get_agent_temp_path(), f"url_image_{hash(url)}.jpg")
                    
                    # Log the URL being processed
                    env.add_agent_log(f"Processing image URL: {url}")
                    
                    if download_image_from_url(url, temp_image_path):
                        # Check file size - if too large, resize it
                        file_size = os.path.getsize(temp_image_path)
                        env.add_agent_log(f"Downloaded image size: {file_size} bytes")
                        
                        # If file is too large (>4MB), we might need to resize it
                        # This would require additional image processing libraries
                        
                        image_description = await get_image_description(env, temp_image_path)
                        image_source = f"URL: {url}"
                        break
                    else:
                        env.add_agent_log(f"Failed to download image from URL: {url}")
            
            # If no valid image URL, check for attachments
            if not image_description:
                files = env.list_files_from_thread(order="desc")
                image_files = [f for f in files if f.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                
                if image_files:
                    # Get the most recent image
                    latest_image = image_files[0]
                    # Download the image to a temporary location
                    image_content = env.read_file_by_id(latest_image.id)
                    temp_image_path = os.path.join(env.get_agent_temp_path(), latest_image.filename)
                    
                    with open(temp_image_path, "wb") as f:
                        if isinstance(image_content, str):
                            f.write(image_content.encode('utf-8'))
                        else:
                            f.write(image_content)
                    
                    # Get description of the image
                    image_description = await get_image_description(env, temp_image_path)
                    image_source = f"Attachment: {latest_image.filename}"
            
            # Break after processing the first user message with an image
            if image_description:
                break
    
    return image_description, image_source


async def agent(env: Environment, state: State):
    # Check for images in the latest user message (either URL or attachment)
    image_description, image_source = await process_image_from_url_or_attachment(env)
    
    if not signer_private_key:
        env.add_reply("Add a secret `signer_private_key` with the private key from your NEAR mainnet account to start")
    else:
        reset_state = False
        
        # If we have an image description, add it to the conversation
        if image_description:
            env.add_reply(f"I see an image in your message ({image_source}). Here's what I can tell about it:\n\n{image_description}\n\nHow can I help you with your NEAR account?")
            reset_state = True

        elif state.action == Actions.GET_USER_DATA:
            # collect user's data
            messages = utils.get_messages(state)
            reply = env.completion(messages)

            data = utils.parse_response(reply)

            if data.get("action") is not None:
                state.action = Actions(data["action"])

            if data.get("amount") is not None:
                state.amount = data["amount"]

            if data.get("receiver_id") is not None:
                state.receiver_id = data["receiver_id"]

            if state.action != Actions.NEAR_SHOW_ACCOUNT:
                env.add_reply(data.get("message"))

        if state.action == Actions.NEAR_SHOW_ACCOUNT:
            # if user asked to show account details
            signer_public_key = utils.get_public_key(signer_private_key)
            signer_account_id = utils.get_account_id(signer_public_key)

            print(f"Reading {signer_account_id}")

            if signer_account_id:
                balance = await utils.get_account_balance(signer_account_id, signer_private_key)

                print(f"Balance {balance}")

                fts = utils.get_account_fts(state, signer_account_id)
                if len(fts) > 0:
                    ft_balances_markdown_str = utils.format_tokens_as_markdown(state, fts)
                else:
                    ft_balances_markdown_str = ""

                nfts = utils.get_account_nfts(state, signer_account_id)
                if len(nfts) > 0:
                    nfts_markdown_str = utils.format_nfts_as_markdown(state, nfts)
                else:
                    nfts_markdown_str = ""

                pools = utils.get_account_staking_pools(state, signer_account_id)
                if len(pools) > 0:
                    pools_markdown_str = utils.format_pools_as_markdown(state, pools)
                else:
                    pools_markdown_str = ""

                env.add_reply(
                    f"Your account is [{signer_account_id}](https://nearblocks.io/address/{signer_account_id}).\nYour account balance is {balance} NEAR.{pools_markdown_str}{ft_balances_markdown_str}{nfts_markdown_str}")
            else:
                env.add_reply(f"Mainnet account with public key {signer_public_key} not found")

            reset_state = True

        if state.action == Actions.NEAR_TRANSFER and state.receiver_id and state.amount > 0:
            # if user asked to make a transfer, and we have all the necessary data

            signer_public_key = utils.get_public_key(signer_private_key)
            signer_account_id = utils.get_account_id(signer_public_key)
            acc = Account(signer_account_id, signer_private_key)

            await acc.startup()

            transaction_hash = await acc.send_money(state.receiver_id, int(NEAR * state.amount), nowait=True)

            env.add_reply(f"Transaction created: [{transaction_hash}](https://nearblocks.io/txns/{transaction_hash})")

            reset_state = True

        if state.action == Actions.NEAR_STAKE and state.receiver_id and state.amount > 0:
            # if user asked to make a transfer, and we have all the necessary data

            signer_public_key = utils.get_public_key(signer_private_key)
            signer_account_id = utils.get_account_id(signer_public_key)
            acc = Account(signer_account_id, signer_private_key)

            await acc.startup()

            transaction_hash = await acc.function_call(state.receiver_id, 'deposit_and_stake', {}, 100000000000000,
                                                       int(NEAR * state.amount), nowait=True)

            env.add_reply(f"Transaction created: [{transaction_hash}](https://nearblocks.io/txns/{transaction_hash})")

            reset_state = True

        if reset_state:
            # just reset state after the successful action
            state.action = "GET_USER_DATA"
            state.amount = None
            state.receiver_id = None

    utils.save_state(state)


class Actions(enum.Enum):
    GET_USER_DATA = "GET_USER_DATA"
    NEAR_SHOW_ACCOUNT = "NEAR_SHOW_ACCOUNT"
    NEAR_TRANSFER = "NEAR_TRANSFER"
    NEAR_STAKE = "NEAR_STAKE"


state = State(**utils.get_state())
if state.action:
    state.action = Actions(state.action)
if state.amount:
    state.amount = float(state.amount)

asyncio.run(agent(env, state))