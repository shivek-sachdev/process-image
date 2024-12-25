import streamlit as st
import boto3
from PIL import Image
import io
import base64
import magic
from pathlib import Path
from typing import Tuple, Union

# Constants
MODEL_ID = "us.meta.llama3-2-90b-instruct-v1:0"
PAYMENT_PROMPT = "Extract and list only these details from the image: 1. Payment Date (in format dd/mm/yyyy) 2. Payment Amount (in THB)"

class ImageUtils:
    @staticmethod
    def resize_img(b64imgstr: str, size: Tuple[int, int] = (256, 256)) -> str:
        buffer = io.BytesIO()
        img = base64.b64decode(b64imgstr)
        img = Image.open(io.BytesIO(img))
        rimg = img.resize(size, Image.LANCZOS)
        rimg.save(buffer, format=img.format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def img2base64(image_path: Union[str, Path], resize: bool = False) -> str:
        with open(image_path, "rb") as img_f:
            img_data = base64.b64encode(img_f.read())
        
        if resize:
            return ImageUtils.resize_img(img_data.decode())
        else:
            return img_data.decode()

    @staticmethod
    def process_image_bytes(image_bytes: bytes, resize: bool = True) -> Tuple[bytes, str]:
        mime = magic.Magic(mime=True)
        image_format = mime.from_buffer(image_bytes).split('/')[-1]

        if resize:
            b64_str = base64.b64encode(image_bytes).decode()
            resized_b64 = ImageUtils.resize_img(b64_str)
            return base64.b64decode(resized_b64), image_format
        
        return image_bytes, image_format

    @staticmethod
    def validate_image(image_bytes: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            return True
        except Exception:
            return False

def init_bedrock_client():
    return boto3.client("bedrock-runtime")

def process_image_with_bedrock(client, image_bytes: bytes, image_format: str):
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
    st.title("Payment Information Extractor")
    
    uploaded_file = st.file_uploader("Upload payment slip/receipt...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            
            if not ImageUtils.validate_image(image_bytes):
                st.error("Invalid image file. Please upload a valid image.")
                return
            
            # Display original image
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Process image without displaying the processed version
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