import boto3
from botocore.config import Config
import base64
import time
import os
import json
import string
import random
import re
from PIL import Image
import io
import sys
import concurrent.futures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    SYSTEM_TEXT_ONLY,
    SYSTEM_IMAGE_TEXT,
    SYSTEM_CANVAS,
    MODEL_OPTIONS,
    DEFAULT_BUCKET,
    DEFAULT_GUIDELINE,
    GENERATED_VIDEOS_DIR,
    CANVAS_SIZE)
from utils import (
    random_string_name,
    load_guideline,
    parse_prompt,
    process_image,
    download_video
)

# Initialize AWS clients
session = boto3.session.Session(region_name='us-east-1')
client = session.client(service_name='bedrock-runtime', 
                       config=Config(connect_timeout=1000, read_timeout=1000))
bedrock_runtime = session.client("bedrock-runtime")

# Constants
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_PROMPT_LENGTH = 512

def optimize_prompt(prompt, guideline_path, model_name=MODEL_OPTIONS["Nova Pro"], image=None):
    if image is not None:
        image = process_image(image)
    
    doc_bytes = load_guideline(guideline_path)
    model_id = MODEL_OPTIONS[model_name]
    
    if image is None:
        # Text-only prompt optimization
        system = [{"text": SYSTEM_TEXT_ONLY}]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "DocumentPDFmessages",
                            "source": {"bytes": doc_bytes}
                        }
                    },
                    {"text": f"Please optimize: {prompt}"},
                ],
            }
        ]
    else:
        # Image + text prompt optimization
        system = [{"text": SYSTEM_IMAGE_TEXT}]
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "DocumentPDFmessages",
                            "source": {"bytes": doc_bytes}
                        }
                    },
                    {"image": {"format": "png", "source": {"bytes": img_bytes}}},
                    {"text": f"Please optimize: {prompt}"},
                ],
            }
        ]

    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    optimized = parse_prompt(text)
    length = len(optimized)
    return optimized, f"{length} chars" + (" (Too Long!)" if length > MAX_PROMPT_LENGTH else "")

def optimize_canvas_prompt(prompt, model_name="Nova Pro"):
    """Optimize prompt for image generation using Canvas model"""
    system = [{"text": SYSTEM_CANVAS}]
    messages = [
        {
            "role": "user",
            "content": [{"text": f"Please optimize: {prompt}"}],
        }
    ]
    
    # Configure inference parameters
    inf_params = {"maxTokens": 512, "topP": 0.9, "temperature": 0.8}
    
    # Get response
    response = client.converse_stream(
        modelId=MODEL_OPTIONS[model_name],
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    # Collect response text
    text = ""
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                text += event["contentBlockDelta"]["delta"]["text"]
    
    # Extract both prompt and negative prompt
    prompt = parse_prompt(text, r"<prompt>(.*?)</prompt>")
    try:
        negative_prompt = parse_prompt(text, r"<negative_prompt>(.*?)</negative_prompt>")
    except ValueError:
        negative_prompt = ""
    
    return prompt, negative_prompt

def generate_image_pair(original_prompt, optimized_prompt, negative_prompt="", quality="standard", num_images=1, height=720, width=1280, seed=0, cfg_scale=6.5):
    """Generate images in parallel for both original and optimized prompts"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both image generation tasks
        future_original = executor.submit(
            generate_single_image,
            original_prompt,
            negative_prompt,
            quality,
            num_images,
            height,
            width,
            seed,
            cfg_scale
        )
        future_optimized = executor.submit(
            generate_single_image,
            optimized_prompt,
            negative_prompt,
            quality,
            num_images,
            height,
            width,
            seed,
            cfg_scale
        )
        
        # Wait for both tasks to complete
        original_images = future_original.result()
        optimized_images = future_optimized.result()
        
        # Combine results
        all_images = []
        if original_images:
            all_images.extend(original_images)
        if optimized_images:
            all_images.extend(optimized_images)
            
        return all_images

def generate_single_image(prompt, negative_prompt="", quality="standard", num_images=1, height=720, width=1280, seed=0, cfg_scale=6.5):
    """Generate image using Nova Canvas model"""
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt
        } if negative_prompt else {
             "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": int(num_images),  # Ensure integer
            "height": int(height),
            "width": int(width),
            "cfgScale": float(cfg_scale),
            "seed":  random.randint(0,858993459) if int(seed) == -1 else int(seed),
            "quality": quality
        }
    })
    
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId='amazon.nova-canvas-v1:0',
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Create temporary directory for saving images
        local_dir = './generated_images'
        os.makedirs(local_dir, exist_ok=True)
        image_paths = []
        
        # Save each image to a temporary file and collect paths
        for i, base64_image in enumerate(response_body.get("images", [])):
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            rand_name = random_string_name()
            path = f"{local_dir}/generated_{rand_name}.png"
            image.save(path)
            image_paths.append(path)
        
        return image_paths if image_paths else None
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def generate_video(prompt, bucket, image=None, seed=0):
    if image is not None:
        image = process_image(image)
    
    # Prepare model input
    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,
            "fps": 24,
            "dimension": "1280x720",
            "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
        },
    }
    
    # Add image if provided
    if image is not None:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        model_input['textToVideoParams']['images'] = [{
            "format": "png",
            "source": {
                "bytes": img_base64
            }
        }]
    
    # Start async video generation
    invocation = bedrock_runtime.start_async_invoke(
        modelId="amazon.nova-reel-v1:0",
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": bucket
            }
        }
    )
    
    # Wait for completion
    while True:
        response = bedrock_runtime.get_async_invoke(
            invocationArn=invocation['invocationArn']
        )
        status = response["status"]
        if status != 'InProgress':
            break
        time.sleep(10)
    
    # Download video
    output_uri = f"{response['outputDataConfig']['s3OutputDataConfig']['s3Uri']}/output.mp4"
    local_path = download_video(output_uri,GENERATED_VIDEOS_DIR)
    
    return local_path



def generate_comparison_videos(original_prompt, optimized_prompt, bucket, image=None, seed=0):
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both video generation tasks
        future_original = executor.submit(generate_video, original_prompt, bucket, image, seed)
        future_optimized = executor.submit(generate_video, optimized_prompt, bucket, image, seed)
        
        # Wait for both tasks to complete
        original_video = future_original.result()
        optimized_video = future_optimized.result()
    
    return original_video, optimized_video