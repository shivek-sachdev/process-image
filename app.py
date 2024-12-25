import streamlit as st
import boto3
from PIL import Image
import io
import base64
import os
from pathlib import Path
from typing import Tuple, Union

# Constants
MODEL_ID = "us.meta.llama3-2-90b-instruct-v1:0"
PAYMENT_PROMPT = """Extract only the date and amount from this image. If the date is in Thai calendar (BE), subtract 543 years to convert to Gregorian calendar (CE). Output in exactly this format:
Date: dd/mm/yyyy
Amount: xxx THB"""
MAX_IMAGE_SIZE = 1120  # Llama 3.2 Vision maximum image size

# Configure AWS credentials from Streamlit secrets
if 'aws_credentials' in st.secrets:
    credentials = st.secrets['aws_credentials']
    os.environ['AWS_ACCESS_KEY_ID'] = credentials['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['aws_secret_access_key']
    os.environ['AWS_DEFAULT_REGION'] = credentials['aws_region']

class ImageUtils:
    @staticmethod
    def resize_img(b64imgstr: str, max_size: int = MAX_IMAGE_SIZE) -> str:
        """
        Resize a base64 encoded image to fit within max_size while maintaining aspect ratio.
        
        Args:
            b64imgstr (str): Base64 encoded image string
            max_size (int): Maximum dimension size (default 1120 for Llama 3.2)
            
        Returns:
            str: Base64 encoded resized image
        """
        buffer = io.BytesIO()
        img = base64.b64decode(b64imgstr)
        img = Image.open(io.BytesIO(img))
        
        # Calculate new size maintaining aspect ratio
        width, height = img.size
        if width > height:
            if width > max_size:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_width = width
                new_height = height
        else:
            if height > max_size:
                new_height = max_size
                new_width = int((width * max_size) / height)
            else:
                new_width = width
                new_height = height
                
        rimg = img.resize((new_width, new_height), Image.LANCZOS)
        rimg.save(buffer, format=img.format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def img2base64(image_path: Union[str, Path], resize: bool = False) -> str:
        """
        Convert an image file to base64 string.
        
        Args:
            image_path (str or Path): Path to the image file
            resize (bool): Whether to resize the image
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as img_f:
            img_data = base64.b64encode(img_f.read())
        
        if resize:
            return ImageUtils.resize_img(img_data.decode())
        else:
            return img_data.decode()

    @staticmethod
    def process_image_bytes(image_bytes: bytes, resize: bool = True) -> Tuple[bytes, str]:
        """
        Process image bytes - optionally resize and detect format.
        
        Args:
            image_bytes (bytes): Raw image bytes
            resize (bool): Whether to resize the image
            
        Returns:
            tuple: (processed image bytes, image format)
        """
        try:
            # Try using PIL to determine format
            img = Image.open(io.BytesIO(image_bytes))
            image_format = img.format.lower() if img.format else 'png'
        except Exception:
            # Default to PNG if format detection fails
            image_format = 'png'

        if resize:
            # Convert to base64, resize, and back to bytes
            b64_str = base64.b64encode(image_bytes).decode()
            resized_b64 = ImageUtils.resize_img(b64_str)
            return base64.b64decode(resized_b64), image_format
        
        return image_bytes, image_format

    @staticmethod
    def validate_image(image_bytes: bytes) -> bool:
        """
        Validate if the bytes represent a valid image.
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            return True
        except Exception:
            return False

def init_bedrock_client():
    """Initialize Amazon Bedrock client with credentials."""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION')
    )

def process_image_with_bedrock(client, image_bytes: bytes, image_format: str):
    """
    Process image with Amazon Bedrock API.
    
    Args:
        client: Bedrock client instance
        image_bytes (bytes): Processed image bytes
        image_format (str): Image format (e.g., 'png', 'jpeg')
        
    Returns:
        str: API response text
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                },
                {
                    "text": PAYMENT_PROMPT
                }
            ]
        }
    ]
    
    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def main():
    """Main application function."""
    st.title("Payment Information Extractor")
    
    uploaded_file = st.file_uploader("Upload payment slip/receipt...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read image bytes
            image_bytes = uploaded_file.read()
            
            # Validate image
            if not ImageUtils.validate_image(image_bytes):
                st.error("Invalid image file. Please upload a valid image.")
                return
            
            # Display original image
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Process image with Llama 3.2 Vision size limit (1120x1120)
            processed_bytes, image_format = ImageUtils.process_image_bytes(image_bytes, resize=True)
            
            if st.button('Extract Payment Information'):
                try:
                    client = init_bedrock_client()
                    
                    with st.spinner('Extracting payment information...'):
                        result = process_image_with_bedrock(client, processed_bytes, image_format)
                        
                        if result:
                            st.subheader("Payment Details:")
                            st.write(result)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    
        except Exception as e:
            st.error(f"An error occurred while loading the image: {str(e)}")

if __name__ == "__main__":
    main()