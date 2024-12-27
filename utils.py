import random
import re
from PIL import Image
import os
import boto3
import string

def random_string_name(length=12):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def load_guideline(guideline_path):
    with open(guideline_path, "rb") as file:
        doc_bytes = file.read()
    return doc_bytes

def parse_prompt(text: str, pattern: str = r"<prompt>(.*?)</prompt>") -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError(f"No match found for pattern: {pattern}")

def process_image(image):
    if image is None:
        return None
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get current dimensions
    width, height = image.size
    
    # Calculate scaling factor to ensure the smaller dimension is at least as large as target
    scale_w = TARGET_WIDTH / width
    scale_h = TARGET_HEIGHT / height
    scale = max(scale_w, scale_h)
    
    # Scale image proportionally
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop from center to target dimensions
    left = (new_width - TARGET_WIDTH) // 2
    top = (new_height - TARGET_HEIGHT) // 2
    right = left + TARGET_WIDTH
    bottom = top + TARGET_HEIGHT
    
    image = image.crop((left, top, right, bottom))
    return image

def download_video(s3_uri,output_dir):
    """Download video from S3 to local storage"""
    try:
        s3_client = boto3.client('s3')
        
        if not s3_uri.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        
        path_parts = s3_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1]
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{s3_key.split('/')[0]}.mp4"
        local_path = os.path.join(output_dir, filename)
        
        s3_client.download_file(bucket_name, s3_key, local_path)
        return local_path
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None