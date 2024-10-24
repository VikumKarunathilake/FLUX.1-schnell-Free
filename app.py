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
import mysql.connector
from mysql.connector import Error
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Together client
api_key  = os.getenv("TOGETHER_API_KEY")
client = Together(api_key =api_key )

# ImgBB API key
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

# MySQL Connection Configuration
MYSQL_URL = os.getenv("MYSQL_URL")

def get_db_config():
    """Parse MySQL URL and return connection config"""
    parsed = urlparse(MYSQL_URL)
    return {
        'host': parsed.hostname,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.strip('/'),
        'port': parsed.port
    }

def get_db_connection():
    """Create and return a MySQL connection"""
    try:
        config = get_db_config()
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            logger.info('Successfully connected to MySQL database')
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise

def init_db():
    """Initialize the MySQL database and create a table for storing image data."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS generated_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            generation_prompt TEXT,
            generation_timestamp DATETIME,
            generation_width INT,
            generation_height INT,
            generation_steps INT,
            imgbb_id VARCHAR(255),
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
            raw_response TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        '''
        
        cursor.execute(create_table_query)
        connection.commit()
        logger.info("Database initialized successfully")
        
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def save_to_database(prompt, width, height, steps, imgbb_response):
    """Save generated image data along with ImgBB response to MySQL database."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        data = imgbb_response['data']
        insert_query = '''
        INSERT INTO generated_images (
            generation_prompt, generation_timestamp, generation_width, generation_height, 
            generation_steps, imgbb_id, imgbb_title, imgbb_url_viewer, imgbb_url, 
            imgbb_display_url, imgbb_width, imgbb_height, imgbb_size, imgbb_time, 
            imgbb_expiration, delete_url, raw_response
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        '''
        
        values = (
            prompt, datetime.now(), width, height, steps,
            data.get('id'), data.get('title'), data.get('url_viewer'),
            data.get('url'), data.get('display_url'), data.get('width'),
            data.get('height'), data.get('size'), data.get('time'),
            data.get('expiration'), data.get('delete_url'),
            json.dumps(imgbb_response)
        )
        
        cursor.execute(insert_query, values)
        connection.commit()
        logger.info("Successfully saved to database")
        return True
        
    except Error as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

# The rest of the functions remain unchanged
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

# Gradio interface remains the same
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Generator")
    gr.Markdown("Generate and upload images")
    html_part = """
    <script defer data-domain="flux-free.up.railway.app" src="https://plausible.io/js/script.file-downloads.hash.outbound-links.pageview-props.revenue.tagged-events.js"></script>
<script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>
"""
    gr.HTML(html_part)
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter image description...",
                lines=3
            )
            with gr.Row():
                width_input = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                height_input = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
            steps_input = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Steps")
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
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)