import random
import re
from PIL import Image
import os
import boto3
import string
import logging
import tempfile
from botocore.exceptions import NoCredentialsError
import qrcode

aws_region_s3 = os.environ.get("AWS_REGION", "us-east-1")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
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
        s3_client = boto3.client('s3',region_name = aws_region_s3)
        
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

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3',region_name = aws_region_s3)
    try:
        s3.upload_file(local_file, bucket, s3_file)
        logger.info(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        logger.info("The file was not found")
        return False
    except NoCredentialsError:
        logger.info("Credentials not available")
        return False
    
def generate_s3_url(bucket, s3_file):
    s3 = boto3.client('s3', region_name=aws_region_s3)
    url = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': bucket, 'Key': s3_file},
                                    ExpiresIn=3600*24*7)  # URL有效期为7天
    return url

def generate_image_qr_code(selected_image_file,bucket):
    if bucket.startswith("s3://"):
        bucket = bucket[5:]
    if not selected_image_file:
        logger.info("请先选择图片")
        return None
    try:
        print("image file is: " + selected_image_file)
        image_path = selected_image_file
        s3_file = image_path.split('/')[-1]
        
        # 上传到AWS S3
        if not upload_to_s3(image_path, bucket, s3_file):
            return "Failed to upload image to S3."
        
        # 生成S3 URL
        s3_url = generate_s3_url(bucket, s3_file)
        
        if not s3_url:
            return "Failed to generate S3 URL."
        
        # 生成二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(s3_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill='black', back_color='white')
        
        # 将二维码图像保存为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp, format='PNG')
            tmp_path = tmp.name
        
        return tmp_path
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return "Failed to generate QR code."


def generate_qr_code(gradio_video_file,bucket):
    if bucket.startswith("s3://"):
        bucket = bucket[5:]
    if not gradio_video_file:
        logger.info("Please generate video first")
        return None
    try:
        logger.info("video file is:"+gradio_video_file)
        video_path = gradio_video_file
        s3_file = video_path.split('/')[-1]     
        # 上传到AWS S3
        if not upload_to_s3(video_path, bucket, s3_file):
            return "Failed to upload video to S3."        
        # 生成S3 URL
        s3_url = generate_s3_url(bucket, s3_file)       
        # 生成二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(s3_url)
        qr.make(fit=True)    
        img = qr.make_image(fill='black', back_color='white')
        # 将二维码图像保存为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp, format='PNG')
            tmp_path = tmp.name
        return tmp_path
    except Exception as e:
        logger.info(f"Error: {str(e)}")
        return "Failed to generate QR code."
