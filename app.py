import gradio as gr
import os
import json
from PIL import Image
import sys
import random
import threading
from datetime import datetime
from argparse import ArgumentParser
import concurrent.futures
import uuid
import shutil
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    SYSTEM_TEXT_ONLY,
    SYSTEM_IMAGE_TEXT,
    SYSTEM_CANVAS,
    MODEL_OPTIONS,
    DEFAULT_BUCKET,
    DEFAULT_GUIDELINE,
    GENERATED_VIDEOS_DIR,
    PROMPT_SAMPLES,
    CANVAS_SIZE)
from shot_video import (
    ReelGenerator,
    generate_shots,
    generate_shot_image,
    generate_reel_prompts,
    generate_shot_vidoes,
    sistch_vidoes,
    extract_last_frame
)
from utils import *
import json
from generation import (
    optimize_prompt,
    optimize_canvas_prompt,
    generate_image_pair,
    generate_single_image,
    generate_video,
    generate_comparison_videos
)

# 全局锁,用于保护共享资源
file_lock = threading.Lock()

class UserSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.work_dir = os.path.join('user_sessions', self.session_id)
        self.setup_directories()
        
    def setup_directories(self):
        """为用户会话创建必要的目录"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'generated_images'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'generated_videos'), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'shot_images'), exist_ok=True)
        
    def get_path(self, *paths):
        """获取用户会话特定的文件路径"""
        return os.path.join(self.work_dir, *paths)
        
    def cleanup(self):
        """清理用户会话数据"""
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

def update_prompt(template_name):
    """更新 prompt 输入框的内容"""
    return PROMPT_SAMPLES.get(template_name, "")

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Nova Reel & Canvas Prompt Optimizer")
        
        # 用户会话状态
        session = gr.State(lambda: UserSession())
        # 选中图片状态
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
                            choices=CANVAS_SIZE,
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
                        canvas_optimize_btn = gr.Button("Optimize Prompt",variant='primary')
                        canvas_generate_btn = gr.Button("Generate Image",variant='primary')
                
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
                    send_to_video_btn = gr.Button("Send Selected Image to Video Generation", interactive=False,variant='primary')

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
                        optimize_btn = gr.Button("Optimize Prompt",variant='primary')
                        generate_comparison_btn = gr.Button("Generate Videos",variant='primary')
                
                gr.Markdown("## Video Comparison")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original Prompt Video")
                        original_video = gr.Video(label="Original")
                    with gr.Column():
                        gr.Markdown("### Optimized Prompt Video")
                        optimized_video = gr.Video(label="Optimized")

            # Shot Video Generation Tab
            with gr.Tab("Long Video Generation", id="shot_video"):
                with gr.Row():
                    with gr.Column():
                        story_input = gr.Textbox(
                            label="Enter your story",
                            lines=5,
                            placeholder="Enter a story to be converted into a long video with multiple scens"
                        )
                        template_dropdown = gr.Dropdown(
                            choices=list(PROMPT_SAMPLES.keys()),
                            label="Sample Prompts"
                        )
                        shot_type_input = gr.Radio(
                            choices=["Breakdown Shot", "Continuous Shot"],
                            value="Breakdown Shot",
                            label="Shot Type",
                            info="Select shot type,(Continuous Shot need to run seqentially, it is much slower"
                        )
                        shot_model_input = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Nova Pro",
                            label="Model",
                            info="Select the model for shot generation"
                        )
                        shot_bucket_input = gr.Textbox(
                            label="S3 Bucket",
                            value=DEFAULT_BUCKET,
                            info="S3 bucket for video output"
                        )
                        num_shot_input = gr.Number(
                            value=3,
                            minimum=2,
                            maximum=10,
                            step=1,
                            label="Number of Shots (Scens)",
                            info="Number of shots to generate (2-10)"
                        )
                        video_seed = gr.Number(
                            value=0,
                            minimum=-1,
                            label="Seed",
                            info="Random seed (-1 for random)"
                        )
                        shot_cfg_scale_input = gr.Slider(
                            value=6.5,
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            label="CFG Scale",
                            info="How closely to follow the prompt"
                        )
                        similarity_strength_input = gr.Slider(
                            value=0.7,
                            minimum=0.2,
                            maximum=1.0,
                            step=0.1,
                            label="Similarity Strength",
                            info="How closely between shot image"
                        )
                    
                    with gr.Column():
                        shots_json = gr.JSON(label="Generated Shots")
                        generate_shots_btn = gr.Button("Generate Shots",variant='primary')
                        status_text = gr.Markdown("Status: Ready")
                        generate_shot_video_btn = gr.Button("Generate Shot Videos", interactive=False,variant='primary')
                        timestamp = gr.Textbox(label="Timestamps", visible=False)
                
                gr.Markdown("## Generated Shot Images")
                shot_images = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="shot_images",
                    columns=[3],
                    rows=[1],
                    height="auto",
                    allow_preview=True
                )
                
                gr.Markdown("## Generated Reel Prompts")
                reel_prompts_json = gr.JSON(label="Reel Prompts")
                
                gr.Markdown("## Generated Long Videos")
                with gr.Row():
                    captioned_video = gr.Video(label="Captioned Video")
                with gr.Row():
                    with gr.Column(scale=1):
                        generate_qr_btn = gr.Button("Generate QR code for Video")
                        generate_image_qr_btn = gr.Button("Generate QR code for selected Image") 
                    with gr.Column(scale=2):
                        qr_output = gr.Image(label="QR code for downloading",type="numpy")

        # Video tab event handlers
        def update_optimized_prompt(prompt, guideline_path, model_name, image):
            optimized, length_info = optimize_prompt(prompt, guideline_path, model_name, image)
            return gr.Textbox(value=optimized, info=length_info)
        
        optimize_btn.click(
            fn=update_optimized_prompt,
            inputs=[prompt_input, guideline_input, model_input, image_input],
            outputs=optimized_prompt,
            concurrency_limit=8
        )

        template_dropdown.change(
            fn=update_prompt,
            inputs=template_dropdown,
            outputs=story_input
        )
        
        def generate_videos_with_comparison(session, original_prompt, optimized_prompt, bucket, image, comparison_mode, seed):
            with file_lock:
                if comparison_mode:
                    # Generate both original and optimized videos
                    return generate_comparison_videos(original_prompt, optimized_prompt, bucket, image, seed, session.work_dir)
                else:
                    # Generate only optimized video
                    optimized = generate_video(optimized_prompt, bucket, image, seed, session.work_dir)
                    return None, optimized
        
        generate_comparison_btn.click(
            fn=generate_videos_with_comparison,
            inputs=[session, prompt_input, optimized_prompt, bucket_input, image_input, comparison_mode_video, video_seed],
            outputs=[original_video, optimized_video],
            concurrency_limit=8
        )

        # Canvas tab event handlers
        def update_canvas_prompts(prompt, model_name):
            optimized, negative = optimize_canvas_prompt(prompt, model_name)
            return optimized, negative
        
        canvas_optimize_btn.click(
            fn=update_canvas_prompts,
            inputs=[canvas_prompt_input, canvas_model_input],
            outputs=[canvas_optimized_prompt, canvas_negative_prompt],
            concurrency_limit=8
        )
        
        def generate_images_with_comparison(session, original_prompt, optimized_prompt, negative_prompt, quality, num_images, size, seed, cfg_scale, comparison_mode):
            # Parse dimensions from size string
            dimensions = size.split(" (")[0].split(" x ")
            width = int(dimensions[0])
            height = int(dimensions[1])
            
            with file_lock:
                if comparison_mode:
                    # Generate both original and optimized images
                    all_images = generate_image_pair(original_prompt, optimized_prompt, negative_prompt, quality, num_images, height, width, seed, cfg_scale, session.work_dir)
                    if not all_images:
                        return None, None
                    mid = len(all_images) // 2
                    return all_images[:mid], all_images[mid:]
                else:
                    # Generate only optimized images
                    optimized_images = generate_single_image(optimized_prompt, negative_prompt, quality, num_images, height, width, seed, cfg_scale, session.work_dir)
                    return None, optimized_images

        canvas_generate_btn.click(
            fn=generate_images_with_comparison,
            inputs=[
                session,
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
            outputs=[original_images, optimized_images],
            concurrency_limit=8
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

        # Shot video tab event handlers
        def update_shots(session, story, model_name, num_shot, shot_type):
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")+'_'+str(random.randint(1000, 9999))
            reel_gen = ReelGenerator(model_id=MODEL_OPTIONS[model_name])
            
            with file_lock:
                if shot_type == 'Breakdown Shot':
                    shots = generate_shots(reel_gen, story, num_shot, False)
                elif shot_type == 'Continuous Shot':
                    shots = generate_shots(reel_gen, story, num_shot, True)
                    
                shot_dir = session.get_path('shot_images', timestamp_str)
                os.makedirs(shot_dir, exist_ok=True)
                with open(os.path.join(shot_dir, 'shots.json'), 'w') as f:
                    json.dump(shots, f, indent=4, ensure_ascii=False)
                    
            return shots, gr.update(interactive=True), timestamp_str
        
        def on_select(gallery_output_images, evt: gr.SelectData):
            selected_index = evt.index
            selected_path = gallery_output_images[selected_index]
            return selected_path[0]

        def generate_shot_videos(session, story, shots, bucket, model_name, seed, cfg_scale, similarity_strength, shot_type, timestamp):
            if not shots or 'shots' not in shots:
                return None, None, None, "Error: No shots data"
                
            reel_gen = ReelGenerator(bucket_name=bucket, model_id=MODEL_OPTIONS[model_name])
            
            try:
                with file_lock:
                    # Generate images for each shot
                    status = gr.update(value="Status: Generating images...")
                    yield None, None, None, status

                    if shot_type == 'Breakdown Shot':
                        image_files = generate_shot_image(reel_gen, shots, timestamp, seed, cfg_scale, similarity_strength, session_dir=session.work_dir)
                        yield None, image_files, None, gr.update(value="Status: Images generated successfully")
                        
                        # Generate optimized prompts for each shot
                        status = gr.update(value="Status: Generating prompts...")
                        yield None, image_files, None, status
                        
                        reel_prompts = generate_reel_prompts(reel_gen, shots, image_files)
                        yield None, image_files, reel_prompts, gr.update(value="Status: Prompts generated successfully")
                        
                        # Generate videos for each shot
                        status = gr.update(value="Status: Generating videos...")
                        yield None, image_files, reel_prompts, status
                        
                        video_files = generate_shot_vidoes(reel_gen, image_files, reel_prompts, session_dir=session.work_dir)
                        
                        # Stitch videos together and add captions
                        status = gr.update(value="Status: Stitching videos...")
                        yield None, image_files, reel_prompts, status
                        
                        _, captioned_video = sistch_vidoes(reel_gen, video_files, shots, timestamp, session_dir=session.work_dir)
                        yield captioned_video, image_files, reel_prompts, gr.update(value="Status: All steps completed successfully")

                    elif shot_type == 'Continuous Shot':
                        # 只生成第一张图片
                        image_files = generate_shot_image(reel_gen, shots, timestamp, seed, cfg_scale, similarity_strength, True, session_dir=session.work_dir)
                        yield None, image_files, None, gr.update(value="Status: The first frame image generated successfully")

                        # Generate optimized prompts for each shot
                        status = gr.update(value="Status: Generating videos...")
                        yield None, image_files, None, status

                        ref_image_files = [image_files[0]]
                        save_path_folder = os.path.split(image_files[0])[0]
                        video_files = []
                        reel_prompts = []
                        for idx, shot in enumerate(shots['shots']):
                            _reel_prompts = generate_reel_prompts(reel_gen, {"shots":[shot]}, ref_image_files)
                            _video_files = generate_shot_vidoes(reel_gen, ref_image_files, _reel_prompts, session_dir=session.work_dir)
                            video_files += _video_files
                            last_frame_image = os.path.join(save_path_folder,f"last_frame_{idx}.png")

                            # extract last frame
                            if idx < len(shots['shots'])-1:
                                extract_last_frame(_video_files[0], last_frame_image)
                                print(f"saved last_frame_image: {last_frame_image}")

                                # update the ref image as last frame
                                ref_image_files = [last_frame_image]
                                image_files.append(last_frame_image)
                                reel_prompts += _reel_prompts
                                status = gr.update(value=f"Status: Generated video segment {idx+1}")
                                yield None, image_files, reel_prompts, status

                        # Stitch videos together and add captions
                        status = gr.update(value="Status: Stitching videos...")
                        yield None, image_files, reel_prompts, status
                        
                        _, captioned_video = sistch_vidoes(reel_gen, video_files, shots, timestamp, session_dir=session.work_dir)
                        yield captioned_video, image_files, reel_prompts, gr.update(value="Status: All steps completed successfully")
                
            except Exception as e:
                yield None, None, None, gr.update(value=f"Error: {str(e)}")

        # Connect shot video tab event handlers
        generate_shots_btn.click(
            fn=update_shots,
            inputs=[session, story_input, shot_model_input, num_shot_input, shot_type_input],
            outputs=[shots_json, generate_shot_video_btn, timestamp],
            concurrency_limit=8
        )
        
        generate_shot_video_btn.click(
            fn=generate_shot_videos,
            inputs=[session, story_input, shots_json, shot_bucket_input, shot_model_input, video_seed, shot_cfg_scale_input, similarity_strength_input, shot_type_input, timestamp],
            outputs=[captioned_video, shot_images, reel_prompts_json, status_text],
            concurrency_limit=8
        )

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
        
        generate_qr_btn.click(
            fn=generate_qr_code,
            inputs=[captioned_video, bucket_input],
            outputs=[qr_output],
            concurrency_limit=8
        )
        
        generate_image_qr_btn.click(
            fn=generate_image_qr_code,
            inputs=[selected_image, bucket_input],
            outputs=[qr_output],
            concurrency_limit=8
        )

        shot_images.select(
            fn=on_select,
            inputs=shot_images,
            outputs=selected_image
        )
        
        # 清理会话数据
        demo.load(lambda: None, None, None).then(
            lambda session: session.cleanup() if session else None,
            inputs=[session],
            outputs=None
        )
    
    return demo

if __name__ == "__main__":
    # Create interface
    demo = create_interface()
    
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    # Launch the interface
    demo.launch(share=True, server_name=args.host, server_port=args.port)
