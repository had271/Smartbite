import chainlit as cl
from dotenv import load_dotenv
import os
from ultralytics import YOLO
from PIL import Image
import cohere



# Load environment variables
load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Initialize YOLO model
yolo_model = YOLO("yolo11n")

# Shopping cart (stored in user session)
shopping_cart = []


def get_recipe_image(query):
    """Generate recipe image URL from Unsplash"""
    return f"https://source.unsplash.com/600x400/?{query},food"


async def smartbite_llm(user_text, detected=None):

    system_prompt = """You are SmartBite..."""

    full_prompt = system_prompt + "\n\n"
    full_prompt += f"User request: {user_text}\n"

    if detected:
        full_prompt += f"Available ingredients: {', '.join(detected)}\n"

    full_prompt += "\nPlease suggest a delicious recipe!"

    try:
        response = co.chat(
            model="command-r-plus",
            message=full_prompt
        )

        reply = response.text
        return reply

    except Exception as e:
        print("COHERE ERROR:", e)
        return f" Error generating recipe: {str(e)}"




@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    shopping_cart.clear()
    
    await cl.Message(
        content="Hello 👋 I'm your **SmartBite AI Chef**!\n\n"
                "I can help you:\n"
                " Suggest recipes based on your ingredients\n"
                " Detect ingredients from photos\n"
                " Create shopping lists for missing items\n\n"
                "💡 **Try:** Upload a photo of your fridge or tell me what ingredients you have!",
        author="smartbite"
    ).send()

    actions = [
    cl.Action(
        name="view_cart",
        value="view_cart",
        label="🛒 View Shopping Cart",
        input_type="button",
        payload={"value": "view_cart"}
        ),
    cl.Action(
        name="clear_cart",
        value="clear_cart",
        label=" Clear Cart",
        input_type="button",
        payload={"value": "clear_cart"}  
        ),
    ]


    await cl.Message(content="Quick Actions:", actions=actions).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""

    images = [file for file in message.elements if "image" in file.mime]

    if images:
        await process_image(images[0])
    else:
        await process_text(message.content)


async def process_text(user_text):
    msg = cl.Message(content="", author="smartbite")
    await msg.send()

    reply = await smartbite_llm(user_text)

    msg.content = reply
    await msg.update()

    if shopping_cart:
        cart_msg = " **Shopping Cart Updated:**\n" + "\n".join([f"• {item}" for item in shopping_cart])
        await cl.Message(content=cart_msg).send()


async def process_image(image_file):
    """Process uploaded image with YOLO detection"""

    processing_msg = cl.Message(content=" Detecting ingredients in your image...")
    await processing_msg.send()

    try:
        # Correct Chainlit way: load from path
        img = Image.open(image_file.path)

        results = yolo_model(img)

        if len(results[0].boxes) > 0:
            labels = list(set([yolo_model.names[int(c)] for c in results[0].boxes.cls]))

            detection_msg = f" **Detected {len(labels)} ingredients:**\n" + ", ".join(labels)
            await cl.Message(content=detection_msg).send()

            recipe_msg = cl.Message(content=" Generating recipe...")
            await recipe_msg.send()

            reply = await smartbite_llm(
                "Suggest a creative and delicious recipe using these ingredients",
                labels
            )

            recipe_msg.content = f"📸 **Recipe based on detected ingredients:**\n\n{reply}"
            await recipe_msg.update()

            recipe_image_url = get_recipe_image(labels[0] if labels else "recipe")
            elements = [
                cl.Image(name="recipe_image", url=recipe_image_url, display="inline")
            ]
            await cl.Message(content="✨ **Suggested Dish:**", elements=elements).send()

            if shopping_cart:
                cart_msg = " **Shopping Cart Updated:**\n" + "\n".join([f"• {item}" for item in shopping_cart])
                await cl.Message(content=cart_msg).send()

        else:
            await cl.Message(
                content=" No ingredients detected. Try a clearer photo!"
            ).send()

    except Exception as e:
        await cl.Message(content=f"Error processing image: {str(e)}").send()


@cl.action_callback("view_cart")
async def on_view_cart(action):
    if shopping_cart:
        cart_content = " **Your Shopping Cart:**\n\n" + "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(shopping_cart)]
        )
        await cl.Message(content=cart_content).send()
    else:
        await cl.Message(content="Your shopping cart is empty!").send()

    await action.remove()


@cl.action_callback("clear_cart")
async def on_clear_cart(action):
    shopping_cart.clear()
    await cl.Message(content="Shopping cart cleared!").send()
    await action.remove()


@cl.on_chat_end
async def end():
    if shopping_cart:
        cart_summary = " **Final Shopping List:**\n" + "\n".join([f"• {item}" for item in shopping_cart])
        await cl.Message(content=cart_summary).send()
