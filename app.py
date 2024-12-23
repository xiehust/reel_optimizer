import gradio as gr
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
    DEFAULT_GUIDELINE
)

# Initialize AWS clients
session = boto3.session.Session(region_name='us-east-1')
client = session.client(service_name='bedrock-runtime', 
                       config=Config(connect_timeout=1000, read_timeout=1000))
bedrock_runtime = session.client("bedrock-runtime")

# Constants
GENERATED_VIDEOS_DIR = "generated_videos"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_PROMPT_LENGTH = 512

def random_string_name(length=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def load_system_prompts(guideline_path):
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

def optimize_prompt(prompt, guideline_path, model_name, image=None):
    if image is not None:
        image = process_image(image)
    
    doc_bytes = load_system_prompts(guideline_path)
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

def optimize_canvas_prompt(prompt, model_name):
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
    local_path = download_video(output_uri)
    
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

def download_video(s3_uri):
    """Download video from S3 to local storage"""
    try:
        s3_client = boto3.client('s3')
        
        if not s3_uri.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        
        path_parts = s3_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1]
        
        os.makedirs(GENERATED_VIDEOS_DIR, exist_ok=True)
        
        filename = f"{s3_key.split('/')[0]}.mp4"
        local_path = os.path.join(GENERATED_VIDEOS_DIR, filename)
        
        s3_client.download_file(bucket_name, s3_key, local_path)
        return local_path
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Nova Reel & Canvas Prompt Optimizer")
        
        # State for storing selected image
        selected_image = gr.State(value=None)
        
        with gr.Tabs() as tabs:
            # Image Generation Tab
            with gr.Tab("Image Generation", id="image_generation"):
                with gr.Row():
                    with gr.Column():
                        canvas_prompt_input = gr.Textbox(label="Enter your prompt")
                        canvas_model_input = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Nova Pro",
                            label="Model",
                            info="Select the model for prompt optimization"
                        )
                        canvas_quality = gr.Radio(
                            choices=["standard", "premium"],
                            value="standard",
                            label="Image Quality",
                            info="Select image generation quality"
                        )
                        canvas_num_images = gr.Number(
                            value=1,
                            minimum=1,
                            maximum=5,
                            step=1,
                            label="Number of Images",
                            info="Number of images to generate (1-5)"
                        )
                        canvas_size = gr.Dropdown(
                            choices=[
                                "1024 x 1024 (1:1)",
                                "2048 x 2048 (1:1)",
                                "1024 x 336 (3:1)",
                                "1024 x 512 (2:1)",
                                "1024 x 576 (16:9)",
                                "1024 x 627 (3:2)",
                                "1024 x 816 (5:4)",
                                "1280 x 720 (16:9)",
                                "2048 x 512 (4:1)",
                                "2288 x 1824 (5:4)",
                                "2512 x 1664 (3:2)",
                                "2720 x 1520 (16:9)",
                                "2896 x 1440 (2:1)",
                                "3536 x 1168 (3:1)",
                                "4096 x 1024 (4:1)",
                                "336 x 1024 (1:3)",
                                "512 x 1024 (1:2)",
                                "512 x 2048 (1:4)",
                                "576 x 1024 (9:16)",
                                "672 x 1024 (2:3)",
                                "720 x 1280 (9:16)",
                                "816 x 1024 (4:5)",
                                "1024 x 4096 (1:4)",
                                "1168 x 3536 (1:3)",
                                "1440 x 2896 (1:2)",
                                "1520 x 2720 (9:16)",
                                "1664 x 2512 (2:3)",
                                "1824 x 2288 (4:5)"

                            ],
                            value="1280 x 720 (16:9)",
                            label="Size (px) / Aspect ratio",
                            info="Select image dimensions"
                        )
                        canvas_seed = gr.Number(
                            value=0,
                            minimum=-1,
                            label="Seed",
                            info="Random seed (-1 for random)"
                        )
                        canvas_cfg_scale = gr.Slider(
                            value=6.5,
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            label="CFG Scale",
                            info="How closely to follow the prompt"
                        )
                        comparison_mode_image = gr.Checkbox(
                            label="Comparison Mode",
                            value=False,
                            info="Generate image with original prompt for comparison"
                        )
                    
                    with gr.Column():
                        canvas_optimized_prompt = gr.Textbox(label="Optimized Prompt")
                        canvas_negative_prompt = gr.Textbox(label="Negative Prompt")
                        canvas_optimize_btn = gr.Button("Optimize Prompt")
                        canvas_generate_btn = gr.Button("Generate Image")
                
                gr.Markdown("## Generated Images")
                gr.Markdown("*Click on an image to select it for video generation*")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original Prompt Images")
                        original_images = gr.Gallery(
                            label="Original Images",
                            show_label=True,
                            elem_id="original_images",
                            columns=[2],
                            rows=[1],
                            height="auto",
                            allow_preview=True
                        )
                    with gr.Column():
                        gr.Markdown("### Optimized Prompt Images")
                        optimized_images = gr.Gallery(
                            label="Optimized Images",
                            show_label=True,
                            elem_id="optimized_images",
                            columns=[2],
                            rows=[1],
                            height="auto",
                            allow_preview=True
                        )
                with gr.Row():
                    selected_image_indicator = gr.Markdown("No image selected")
                    send_to_video_btn = gr.Button("Send Selected Image to Video Generation", interactive=False)

            # Video Generation Tab
            with gr.Tab("Video Generation", id="video_generation"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(label="Enter your prompt")
                        guideline_input = gr.Textbox(
                            label="Guideline PDF path", 
                            value=DEFAULT_GUIDELINE,
                            info="Path to the Nova Reel guideline PDF file"
                        )
                        model_input = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Nova Pro",
                            label="Model",
                            info="Select the model for prompt optimization"
                        )
                        bucket_input = gr.Textbox(
                            label="S3 Bucket",
                            value=DEFAULT_BUCKET,
                            info="S3 bucket for video output"
                        )
                        comparison_mode_video = gr.Checkbox(
                            label="Comparison Mode",
                            value=False,
                            info="Generate video with original prompt for comparison"
                        )
                        image_input = gr.Image(label="Upload an image (optional)", type="pil")
                        video_seed = gr.Number(
                            value=0,
                            minimum=-1,
                            label="Seed",
                            info="Random seed (-1 for random)"
                        )
                    
                    with gr.Column():
                        optimized_prompt = gr.Textbox(
                            label="Optimized Prompt",
                            info="Length: 0 chars"
                        )
                        optimize_btn = gr.Button("Optimize Prompt")
                        generate_comparison_btn = gr.Button("Generate Videos")
                
                gr.Markdown("## Video Comparison")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original Prompt Video")
                        original_video = gr.Video(label="Original")
                    with gr.Column():
                        gr.Markdown("### Optimized Prompt Video")
                        optimized_video = gr.Video(label="Optimized")

        # Video tab event handlers
        def update_optimized_prompt(prompt, guideline_path, model_name, image):
            optimized, length_info = optimize_prompt(prompt, guideline_path, model_name, image)
            return gr.Textbox(value=optimized, info=length_info)
        
        optimize_btn.click(
            fn=update_optimized_prompt,
            inputs=[prompt_input, guideline_input, model_input, image_input],
            outputs=optimized_prompt
        )
        
        def generate_videos_with_comparison(original_prompt, optimized_prompt, bucket, image, comparison_mode, seed):
            if comparison_mode:
                # Generate both original and optimized videos
                return generate_comparison_videos(original_prompt, optimized_prompt, bucket, image, seed)
            else:
                # Generate only optimized video
                optimized = generate_video(optimized_prompt, bucket, image, seed)
                return None, optimized
        
        generate_comparison_btn.click(
            fn=generate_videos_with_comparison,
            inputs=[prompt_input, optimized_prompt, bucket_input, image_input, comparison_mode_video, video_seed],
            outputs=[original_video, optimized_video]
        )

        # Canvas tab event handlers
        def update_canvas_prompts(prompt, model_name):
            optimized, negative = optimize_canvas_prompt(prompt, model_name)
            return optimized, negative
        
        canvas_optimize_btn.click(
            fn=update_canvas_prompts,
            inputs=[canvas_prompt_input, canvas_model_input],
            outputs=[canvas_optimized_prompt, canvas_negative_prompt]
        )
        
        def generate_images_with_comparison(original_prompt, optimized_prompt, negative_prompt, quality, num_images, size, seed, cfg_scale, comparison_mode):
            # Parse dimensions from size string
            dimensions = size.split(" (")[0].split(" x ")
            width = int(dimensions[0])
            height = int(dimensions[1])
            
            if comparison_mode:
                # Generate both original and optimized images
                all_images = generate_image_pair(original_prompt, optimized_prompt, negative_prompt, quality, num_images, height, width, seed, cfg_scale)
                if not all_images:
                    return None, None
                mid = len(all_images) // 2
                return all_images[:mid], all_images[mid:]
            else:
                # Generate only optimized images
                optimized_images = generate_single_image(optimized_prompt, negative_prompt, quality, num_images, height, width, seed, cfg_scale)
                return None, optimized_images

        canvas_generate_btn.click(
            fn=generate_images_with_comparison,
            inputs=[
                canvas_prompt_input,
                canvas_optimized_prompt,
                canvas_negative_prompt,
                canvas_quality,
                canvas_num_images,
                canvas_size,
                canvas_seed,
                canvas_cfg_scale,
                comparison_mode_image
            ],
            outputs=[original_images, optimized_images]
        )

        # Handle image selection
        def on_image_select(evt: gr.SelectData, original_gallery, optimized_gallery):
            """Handle image selection in gallery"""
            if evt is None or (original_gallery is None and optimized_gallery is None):
                return None, gr.update(value="No image selected"), gr.update(interactive=False)
            
            # Determine which gallery was clicked and get the image path
            gallery = original_gallery if evt.target.elem_id == "original_images" else optimized_gallery
            gallery_index = evt.index
            
            # Handle both string paths and tuple/list paths
            if gallery and gallery_index < len(gallery):
                item = gallery[gallery_index]
                selected_path = item[0] if isinstance(item, (list, tuple)) else item
                gallery_name = "Original" if evt.target.elem_id == "original_images" else "Optimized"
                return selected_path, gr.update(value=f"{gallery_name} Image {gallery_index + 1} selected"), gr.update(interactive=True)
            
            return None, gr.update(value="No image selected"), gr.update(interactive=False)

        # Function to send selected image to video tab
        def send_to_video(selected_path, gallery):
            """Send selected image to video tab"""
            if selected_path is None:
                return [
                    None,  # image_input
                    None,  # selected_image
                    gr.update(value="No image selected"),  # indicator
                    gr.update(interactive=True),  # Keep button interactive
                    gr.update(selected="video_generation")  # tabs
                ]
            
            try:
                # Load the image
                image = Image.open(selected_path) if isinstance(selected_path, str) else selected_path
                
                # Return values to update UI
                return [
                    image,  # Update image_input
                    selected_path,   # Maintain selected_image state
                    gr.update(value="No image selected"),  # Reset selection indicator
                    gr.update(interactive=True),  # Keep button interactive
                    gr.update(selected="video_generation")  # Switch to video tab
                ]
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                return [None, None, gr.update(value="Error loading image"), gr.update(interactive=True), gr.update(selected="video_generation")]

        # Add event handlers for image selection and transfer
        original_images.select(
            fn=on_image_select,
            inputs=[original_images, optimized_images],
            outputs=[selected_image, selected_image_indicator, send_to_video_btn]
        )
        optimized_images.select(
            fn=on_image_select,
            inputs=[original_images, optimized_images],
            outputs=[selected_image, selected_image_indicator, send_to_video_btn]
        )

        send_to_video_btn.click(
            fn=send_to_video,
            inputs=[selected_image, optimized_images],
            outputs=[
                image_input,
                selected_image,
                selected_image_indicator,
                send_to_video_btn,
                tabs
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Create interface
    demo = create_interface()
    
    # Launch the interface
    demo.launch(share=True)
