import gradio as gr
from together import Together
import base64
from PIL import Image
import io
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import psycopg2
from psycopg2 import Error
from urllib.parse import urlparse
from functools import lru_cache
import time
from typing import Tuple, Any, Optional
import genarationData

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Together client
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

# Configuration constants
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

CACHE_TIMEOUT = 3600
MAX_RETRIES = 3
RETRY_DELAY = 1

@lru_cache(maxsize=1)
def get_db_config() -> dict:
    """Parse PostgreSQL URL and return connection configuration."""
    parsed = urlparse(POSTGRES_URL)
    return {
        'host': parsed.hostname,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.strip('/'),
        'port': parsed.port,
    }

def get_db_connection() -> psycopg2.extensions.connection:
    """Establish a connection to the PostgreSQL database with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            config = get_db_config()
            connection = psycopg2.connect(**config)
            if connection:
                logger.info('Successfully connected to PostgreSQL database')
                return connection
        except Error as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Final attempt failed to connect to PostgreSQL: {e}")
                raise
            time.sleep(RETRY_DELAY)
    raise Exception("Failed to connect to database after retries")

def init_db():
    """Initialize the database schema if it doesn't exist."""
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS generated_images (
                    id SERIAL PRIMARY KEY,
                    generation_prompt TEXT NOT NULL,
                    generation_timestamp TIMESTAMP NOT NULL,
                    generation_width INT NOT NULL,
                    generation_height INT NOT NULL,
                    generation_steps INT NOT NULL,
                    imgbb_id VARCHAR(255) NOT NULL,
                    imgbb_title VARCHAR(255),
                    imgbb_url_viewer TEXT,
                    imgbb_url TEXT,
                    imgbb_display_url TEXT,
                    imgbb_width VARCHAR(50),
                    imgbb_height VARCHAR(50),
                    imgbb_size VARCHAR(50),
                    imgbb_time VARCHAR(50),
                    imgbb_expiration VARCHAR(50),
                    delete_url TEXT,
                    raw_response TEXT,
                    user_id VARCHAR(255)
                );
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON generated_images (generation_timestamp);')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_imgbb_id ON generated_images (imgbb_id);')
            connection.commit()
            logger.info("Database initialized successfully")
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        raise

def retry_with_backoff(func):
    """Decorator for functions to retry with exponential backoff."""
    def wrapper(*args, **kwargs):
        for i in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == MAX_RETRIES - 1:
                    raise
                wait = (2 ** i) * RETRY_DELAY
                logger.warning(f"Attempt {i+1} failed, retrying in {wait} seconds...")
                time.sleep(wait)
    return wrapper

@retry_with_backoff
def upload_to_imgbb(image_bytes: bytes) -> dict:
    """Upload image to ImgBB and return the response."""
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
        "image": img_base64
    }
    response = requests.post(url, payload, timeout=30)
    response.raise_for_status()
    logger.info("Successfully uploaded to ImgBB")
    return response.json()

@retry_with_backoff
def generate_image(prompt: str, width: int, height: int, steps: int) -> Tuple[Image.Image, bytes]:
    """Generate an image using the Together API."""
    if not prompt.strip():
        raise ValueError("Please enter a prompt")
    
    logger.info(f"Generating image with parameters: width={width}, height={height}, steps={steps}")
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",
        width=width,
        height=height,
        steps=steps,
        n=1,
        response_format="b64_json"
    )
    
    image_bytes = base64.b64decode(response.data[0].b64_json)
    image = Image.open(io.BytesIO(image_bytes))
    
    logger.info("Image generated successfully")
    return image, image_bytes

def handle_generation(prompt: str, width: int, height: int, steps: int) -> Tuple[Optional[Image.Image], str]:
    """Handle the entire process of image generation, upload, and database save."""
    try:
        image, image_bytes = generate_image(prompt, width, height, steps)
        imgbb_response = upload_to_imgbb(image_bytes)
        if save_to_database(prompt, width, height, steps, imgbb_response):
            return image, "Image generated successfully!"
        else:
            return image, "Image generated and uploaded, but database save failed!"
    except Exception as e:
        logger.error(f"Error in handle_generation: {str(e)}")
        return None, f"Error: {str(e)}"

def save_to_database(prompt: str, width: int, height: int, steps: int, imgbb_response: dict) -> bool:
    """Save image generation details to the database."""
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                data = imgbb_response['data']
                insert_query = '''
                INSERT INTO generated_images (
                    generation_prompt, generation_timestamp, generation_width, generation_height, 
                    generation_steps, imgbb_id, imgbb_title, imgbb_url_viewer, imgbb_url, 
                    imgbb_display_url, imgbb_width, imgbb_height, imgbb_size, imgbb_time, 
                    imgbb_expiration, delete_url, raw_response, user_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                '''
                values = (
                    prompt, datetime.now(), width, height, steps,
                    data.get('id'), data.get('title'), data.get('url_viewer'),
                    data.get('url'), data.get('display_url'), data.get('width'),
                    data.get('height'), data.get('size'), data.get('time'),
                    data.get('expiration'), data.get('delete_url'),
                    json.dumps(imgbb_response), None  # Assuming user_id is optional
                )
                cursor.execute(insert_query, values)
                connection.commit()
                logger.info("Successfully saved to database")
                return True
    except Error as e:
        logger.error(f"Database error: {e}")
        return False

def create_demo():
    """Create and return the Gradio demo interface."""
    with gr.Blocks(css="style.css", theme="NoCrypt/miku", title="Elixir Craft Image Generator") as demo:
        gr.HTML("""
            <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elixir Craft Image Generator</title>
    <meta name="description" content="Generate stunning AI art from text descriptions with Elixir Craft's FLUX.1 [schnell] model. Unleash your creativity and bring your visions to life.">
    <meta name="keywords" content="AI art generator, text-to-image, AI image generation, FLUX.1, Elixir Craft, digital art, AI art, creative tools, image synthesis, artificial intelligence">
    <meta name="author" content="Vikum Karunathilake">

    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://generator.elixircraft.net/">
    <meta property="og:title" content="Elixir Craft Image Generator">
    <meta property="og:description" content="Generate stunning AI art from text descriptions with Elixir Craft's FLUX.1 [schnell] model. Unleash your creativity and bring your visions to life.">


    <!-- Twitter -->
    <meta name="twitter:card" content="summary">  <!-- Changed to summary for smaller image -->
    <meta name="twitter:creator" content="@VikumKarunathilake">
    <meta name="twitter:title" content="Elixir Craft Image Generator">
    <meta name="twitter:description" content="Generate stunning AI art from text descriptions with Elixir Craft's FLUX.1 [schnell] model. Unleash your creativity and bring your visions to life.">

            </head>
            <div class="text-center p-5" role="main">
                <h1 class="text-3xl sm:text-4xl font-semibold text-gray-800">
                    Elixir Craft Image Generator 🖼️
                </h1>
                <p class="text-base sm:text-lg text-gray-600 max-w-3xl mx-auto mt-2">
                    Welcome to the <strong>AI Image Generator</strong> powered by the <strong>FLUX.1 [schnell]</strong> model! 🎨
                </p>
                <p class="text-sm sm:text-base text-gray-600 max-w-3xl mx-auto mt-4">
                    Enter a description of any scene, character, or object you'd like to see come to life, adjust image dimensions,
                    and select the number of steps to control image detail. Click <strong>"Generate Image"</strong> to create your
                    custom artwork in seconds!
                </p>
                <div class="text-left max-w-3xl mx-auto mt-6 text-gray-800">
                    <h2 class="text-lg sm:text-xl font-semibold">Features:</h2>
                    <ul class="list-disc list-inside text-gray-600 mt-2 space-y-1" role="list">
                        <li>Generate high-quality images from text descriptions</li>
                        <li>Optimized for quick and reliable outputs</li>
                    </ul>
                </div>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Image Description",
                    placeholder="Enter your image description here...",
                    lines=3,
                    elem_id="prompt-input",
                    elem_classes="accessible-input",
                )
                
                with gr.Row():
                    width_input = gr.Slider(
                        minimum=256,
                        maximum=1440,
                        value=832,
                        step=16,
                        label="Image Width",
                    )
                    height_input = gr.Slider(
                        minimum=256,
                        maximum=1440,
                        value=1216,
                        step=16,
                        label="Image Height",
                    )
                
                steps_input = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=4,
                    step=1,
                    label="Generation Steps",
                )
                
                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    elem_id="generate-btn",
                    elem_classes="accessible-button"
                )

            with gr.Column():
                image_output = gr.Image(
                    label="Generated Image",
                    elem_id="generated-image",
                    elem_classes="accessible-image",
                    show_label=True
                )
                status_output = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    elem_id="status-output",
                    elem_classes="accessible-status"
                )

        # Event binding for image generation
        generate_btn.click(
            fn=handle_generation,
            inputs=[prompt_input, width_input, height_input, steps_input],
            outputs=[image_output, status_output]
        )

        gr.HTML("""
            <div class="text-center p-6" role="complementary">
                <h2 class="text-2xl sm:text-3xl font-semibold text-gray-800">Explore the FLUX.1 Gallery</h2>
                <p class="text-sm sm:text-base text-gray-600 max-w-2xl mx-auto mt-2">
                    Discover all images generated with the FLUX.1 AI Image Generator. Each creation is stored in the gallery for
                    you to view, share, or download. Every image includes the prompt details and settings.
                </p>
                <a href="https://gallery.elixircraft.net"
                   target="_blank"
                   rel="noopener noreferrer"
                   class="inline-block mt-4 px-4 py-2 text-base sm:text-lg font-medium text-white bg-blue-500 rounded hover:bg-blue-600 transition"
                   role="button"
                   aria-label="Visit the FLUX.1 Gallery">
                    Visit the Gallery
                </a>
            </div>
            <footer role="contentinfo" class="text-center p-4 mt-8 text-sm text-gray-600">
                <hr class="my-4">
                <p>&copy; 2024 FLUX.1[schnell] AI Image Generator. All rights reserved.</p>
                <p>Contact: <a href="https://discord.com/users/781158548364853270" target="_blank" rel="noopener noreferrer" aria-label="Contact us on Discord">Discord (Vikum_K)</a></p>
                <p>Powered by <a href="https://api.together.xyz/" target="_blank" rel="noopener noreferrer" aria-label="Visit Together.ai">Together.ai</a></p>
            </footer>
        """)
    return demo

if __name__ == "__main__":
    init_db()  # Initialize the database on program start
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True)
