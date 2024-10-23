import gradio as gr
from together import Together
import base64
from PIL import Image
import io
import logging
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Together client
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

# ImgBB API key
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

# Initialize SQLite database
def init_db():
    """Initialize the SQLite database and create a table for storing image data."""
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_prompt TEXT,
            generation_timestamp DATETIME,
            generation_width INTEGER,
            generation_height INTEGER,
            generation_steps INTEGER,
            imgbb_id TEXT,
            imgbb_title TEXT,
            imgbb_url_viewer TEXT,
            imgbb_url TEXT,
            imgbb_display_url TEXT,
            imgbb_width TEXT,
            imgbb_height TEXT,
            imgbb_size TEXT,
            imgbb_time TEXT,
            imgbb_expiration TEXT,
            delete_url TEXT,
            raw_response TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_database(prompt, width, height, steps, imgbb_response):
    """Save generated image data along with ImgBB response to SQLite database."""
    try:
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        data = imgbb_response['data']
        c.execute('''
            INSERT INTO generated_images (
                generation_prompt, generation_timestamp, generation_width, generation_height, generation_steps,
                imgbb_id, imgbb_title, imgbb_url_viewer, imgbb_url, imgbb_display_url, imgbb_width,
                imgbb_height, imgbb_size, imgbb_time, imgbb_expiration, delete_url, raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prompt, datetime.now(), width, height, steps,
            data.get('id'), data.get('title'), data.get('url_viewer'),
            data.get('url'), data.get('display_url'), data.get('width'),
            data.get('height'), data.get('size'), data.get('time'),
            data.get('expiration'), data.get('delete_url'),
            json.dumps(imgbb_response)
        ))
        conn.commit()
        logger.info("Successfully saved to database")
        return True
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False
    finally:
        conn.close()

def upload_to_imgbb(image_bytes):
    """Upload an image to ImgBB and return the complete response."""
    try:
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": IMGBB_API_KEY,
            "image": img_base64
        }
        response = requests.post(url, payload)
        response.raise_for_status()
        logger.info("Successfully uploaded to ImgBB")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"ImgBB upload error: {str(e)}")
        raise Exception("Failed to upload image to ImgBB")

def generate_image(prompt, width, height, steps):
    """Generate an image using Together API and return both the image and the original bytes."""
    if not prompt.strip():
        raise ValueError("Please enter a prompt")
    
    try:
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
        
        # Get the base64 encoded image
        image_bytes = base64.b64decode(response.data[0].b64_json)
        
        # Create PIL Image object
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info("Image generated successfully")
        return image, image_bytes
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise Exception(f"Error generating image: {str(e)}")

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Generator")
    gr.Markdown("Generate and upload images")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter image description...",
                lines=3
            )
            with gr.Row():
                width_input = gr.Slider(minimum=256, maximum=1024, value=1024, step=64, label="Width")
                height_input = gr.Slider(minimum=256, maximum=1024, value=768, step=64, label="Height")
            steps_input = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Steps")
            generate_btn = gr.Button("Generate Image")

        with gr.Column():
            image_output = gr.Image(label="Generated Image")
            status_output = gr.Textbox(label="Status", interactive=False)

    def handle_generation(prompt, width, height, steps):
        """Handle the image generation and upload process."""
        try:
            # Generate the image
            image, image_bytes = generate_image(prompt, width, height, steps)
            
            # Upload to ImgBB
            status_output.value = "Uploading to ImgBB..."
            imgbb_response = upload_to_imgbb(image_bytes)
            
            # Save to database
            if save_to_database(prompt, width, height, steps, imgbb_response):
                return image, "Image generated, uploaded, and saved successfully!"
            else:
                return image, "Image generated and uploaded, but database save failed!"
                
        except Exception as e:
            logger.error(f"Error in handle_generation: {str(e)}")
            return None, f"Error: {str(e)}"

    # Bind the function to the button click
    generate_btn.click(
        fn=handle_generation,
        inputs=[prompt_input, width_input, height_input, steps_input],
        outputs=[image_output, status_output]
    )

if __name__ == "__main__":
    init_db()  # Initialize the database on program start
    demo.launch(share=True)