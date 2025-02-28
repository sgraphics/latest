import asyncio
import base64
import requests
from io import BytesIO

from nearai.agents.environment import Environment

# Define a simple state class
class State:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the image data
        image_data = response.content
        
        # Encode the image as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Determine image format from content or URL
        if url.lower().endswith('.png'):
            image_format = "png"
        elif url.lower().endswith('.jpg') or url.lower().endswith('.jpeg'):
            image_format = "jpeg"
        elif url.lower().endswith('.gif'):
            image_format = "gif"
        elif url.lower().endswith('.webp'):
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
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail. What do you see? Include all relevant details about objects, people, settings, colors, and any text visible in the image."
                    }
                ]
            }
        ]
        
        # Use the vision model to analyze the image
        model = "phi-3-vision-128k-instruct"
        response = env.completion(messages, model=model)
        
        return response
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


async def agent(env: Environment, state: State):
    # Register the image description tool
    tool_registry = env.get_tool_registry(new=True)  # Create a new registry with only our tool
    tool_registry.register_tool(describe_image)
    
    # Get system prompt
    system_prompt = """
    You are an AI assistant specialized in image analysis. You can help users understand what's in their images by providing detailed descriptions.
    
    You have access to a powerful image analysis tool that can:
    - Analyze images from URLs
    - Provide detailed descriptions of image content
    - Identify objects, people, text, and other elements in images
    
    When a user sends you an image URL or asks about analyzing an image, use the describe_image tool to process it.
    
    Examples of how you can help:
    - Describe what's in an image
    - Verify if an image contains specific content
    - Read text from images
    - Analyze scenes, objects, and people in photos
    
    Always be helpful, accurate, and respectful in your analysis.
    """
    
    # Get all messages
    messages = env.list_messages()
    
    # Add system prompt to the beginning
    messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Use tools in completion
    all_tools = tool_registry.get_all_tool_definitions()
    response = env.completions_and_run_tools(messages, tools=all_tools)
    
    # Add the response to the chat
    if response:
        env.add_reply(response)
    
    # Request user input for the next interaction
    env.request_user_input()


# Initialize state and run the agent
state = State()
asyncio.run(agent(env, state))