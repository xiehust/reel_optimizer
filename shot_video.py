import boto3
import json
import base64
import time
import os
import io
import re
import magic
import random
import string
from datetime import datetime
from PIL import Image
from io import BytesIO
from botocore.config import Config
from moviepy import VideoFileClip, CompositeVideoClip, TextClip
from json import JSONDecodeError
import sys
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SHOT_SYSTEM,SYSTEM_TEXT_ONLY,SYSTEM_IMAGE_TEXT,DEFAULT_GUIDELINE,LITE_MODEL_ID,CONTINUOUS_SHOT_SYSTEM
from utils import random_string_name
from generation import optimize_canvas_prompt



class ReelGenerator:
    def __init__(self,model_id = LITE_MODEL_ID,region='us-east-1', bucket_name='s3://bedrock-video-generation-us-east-1-jlvyiv'):
        """Initialize ReelGenerator with AWS credentials and configuration."""
        config = Config(
            connect_timeout=1000,
            read_timeout=1000,
        )
        if not bucket_name.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        path_parts = bucket_name[5:].split('/', 1)
        self.s3_bucket =  path_parts[0]
        self.session = boto3.session.Session(region_name=region)
        self.bedrock_runtime = self.session.client(service_name='bedrock-runtime', config=config)
        self.MODEL_ID =model_id

    def _parse_json(self, pattern: str, text: str) -> str:
        """Parse text using regex pattern."""
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            return text.strip()
        else:
            raise JSONDecodeError("No match found", text, 0)
    
    def _split_caption(self,text: str) -> list:
        """Split caption text into parts."""
        delimiters = [',', '，', '。', '.', ';', '；', '\n', '\t']
        pattern = '|'.join(map(re.escape, delimiters))
        return [part for part in re.split(pattern, text) if part]

    def generate_shots(self, story: str, system_prompt: str) -> dict:
        """Generate shot descriptions from a story."""
        with open(DEFAULT_GUIDELINE, "rb") as file:
            doc_bytes = file.read()
        system = [{"text": system_prompt}]
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
                    {"text": f"Please generate shots for User input:{story}"}],
            },
            {
                "role": "assistant",
                "content": [{"text": "```json"}],
            }
        ]

        response = self.bedrock_runtime.converse_stream(
            modelId=self.MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 2000, "topP": 0.9, "temperature": 0.8}
        )

        text = ""
        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    text += event["contentBlockDelta"]["delta"]["text"]
        
        return json.loads(text[:-3])

    def generate_image(self, body: dict) -> list:
        """Generate images using Amazon Nova Canvas model."""
        accept = "application/json"
        content_type = "application/json"

        response = self.bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId='amazon.nova-canvas-v1:0',
            accept=accept,
            contentType=content_type
        )
        
        response_body = json.loads(response.get("body").read())
        image_bytes_list = []
        
        if "images" in response_body:
            for base64_image in response_body["images"]:
                base64_bytes = base64_image.encode('ascii')
                image_bytes = base64.b64decode(base64_bytes)
                image_bytes_list.append(image_bytes)

        return image_bytes_list

    def generate_variations(self,reference_image_paths,prompt,negative_prompt,save_filepath,seed:int = 0,cfg_scale:float = 6.5,similarity_strength:float = 0.8):
        # Load all reference images as base64.
        images = []
        for path in reference_image_paths:
            with open(path, "rb") as image_file:
                images.append(base64.b64encode(image_file.read()).decode("utf-8"))

        # Configure the inference parameters.
        inference_params = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": images, # Images to use as reference
                "text": prompt, 
                "similarityStrength": similarity_strength,  # Range: 0.2 to 1.0
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,  # Number of variations to generate. 1 to 5.
                "quality": "standard",  # Allowed values are "standard" and "premium"
                "width": 1280,  # See README for supported output resolutions
                "height": 720,  # See README for supported output resolutions
                "cfgScale": cfg_scale,  # How closely the prompt will be followed
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
            },
        }
        if len(negative_prompt):
            inference_params['imageVariationParams']["negativeText"] = negative_prompt
            
        try:
            image_bytes_ret = self.generate_image( inference_params)
            for idx,image_bytes in enumerate(image_bytes_ret):
                image = Image.open(io.BytesIO(image_bytes))
                image.save(save_filepath)
                print(f"image saved to {save_filepath}")
            return image_bytes_ret
        except Exception as err:
            raise ValueError(f"generate_variations:{str(err)}")

    def generate_text2img(self, prompt: str, negative_prompt: str, save_filepath: str,seed:int =0,cfg_scale:float = 6.5) -> str:
        """Generate image from text prompt."""
        inference_params = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 720,
                "width": 1280,
                "cfgScale": cfg_scale,
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed)
            }
        }
        if len(negative_prompt):
            inference_params['textToImageParams']["negativeText"] = negative_prompt

        try:
            image_bytes_list = self.generate_image(inference_params)
            for image_bytes in image_bytes_list:
                image = Image.open(BytesIO(image_bytes))
                image.save(save_filepath)
                print(f"image saved to {save_filepath}")
            return save_filepath
        except Exception as err:
            raise ValueError(f"generate_text2img:{str(err)}")

    def optimize_reel_prompt(self, system_prompt: str, user_prompt: str, ref_image: str, doc_bytes: bytes) -> str:
        """Optimize reel prompt using reference image."""
        with open(ref_image, "rb") as f:
            image = f.read()
        mime_type = magic.Magic(mime=True).from_file(ref_image)

        system = [{"text": system_prompt}]
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
                    {"image": {"format": mime_type.split('/')[1], "source": {"bytes": image}}},
                    {"text": f"Please generate shots for User input:{user_prompt}"},
                ],
            }
        ]

        response = self.bedrock_runtime.converse_stream(
            modelId=self.MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 2000, "topP": 0.9, "temperature": 0.5}
        )

        text = ""
        stream = response.get("stream")
        if stream:
            for event in stream:
                if "contentBlockDelta" in event:
                    text += event["contentBlockDelta"]["delta"]["text"]
        
        return self._parse_json(r"<prompt>(.*?)</prompt>", text)

    def generate_video(self, text_prompt: str, ref_image: str = None,seed:int = 0) -> dict:
        """Generate video from text prompt and optional reference image."""
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": text_prompt,
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": random.randint(0,858993459) if int(seed) == -1 else int(seed),
            },
        }

        if ref_image:
            with open(ref_image, "rb") as f:
                image = f.read()
                input_image_base64 = base64.b64encode(image).decode("utf-8")
                model_input['textToVideoParams']['images'] = [
                    {
                        "format": magic.Magic(mime=True).from_file(ref_image).split('/')[1],
                        "source": {"bytes": input_image_base64}
                    }
                ]

        try:
            invocation = self.bedrock_runtime.start_async_invoke(
                modelId="amazon.nova-reel-v1:0",
                modelInput=model_input,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"s3://{self.s3_bucket}"
                    }
                }
            )
            return invocation
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return None

    def fetch_job_status(self, invocation_arns: list) -> list:
        """Fetch status of video generation jobs."""
        final_responses = []
        for invocation in invocation_arns:
            while True:
                response = self.bedrock_runtime.get_async_invoke(invocationArn=invocation)
                status = response["status"]
                if status != 'InProgress':
                    final_responses.append(response)
                    break
                time.sleep(5)
        return final_responses

    def download_video_from_s3(self, s3_uri: str, local_path: str) -> str:
        """Download generated video from S3."""
        try:
            s3_client = self.session.client('s3')
            
            if not s3_uri.startswith('s3://'):
                raise ValueError("Invalid S3 URI format")
            
            path_parts = s3_uri[5:].split('/', 1)
            if len(path_parts) != 2:
                raise ValueError("Invalid S3 URI format")
            
            bucket_name = path_parts[0]
            s3_key = path_parts[1]
            
            os.makedirs(local_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            fname = timestamp + ''.join(random_string_name()) + '.mp4'
            local_file_path = os.path.join(local_path, fname)
            
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            return local_file_path
            
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return None

    def stitch_videos(self, video1_path: str, video2_path: str, output_path: str) -> str:
        """Stitch two videos together."""
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)

        final_clip = CompositeVideoClip([
            clip1,
            clip2.with_start(clip1.duration),
        ])

        final_clip.write_videofile(output_path)

        clip1.close()
        clip2.close()
        final_clip.close()
        
        return output_path

    def add_timed_captions(self, video_path: str, output_path: str, captions: list, font: str = './yahei.ttf') -> None:
        """Add timed captions to video."""
        video = VideoFileClip(video_path)
        txt_clips = []
        
        for caption in captions:
            text, start_time, end_time = caption
            txt_clip = TextClip(
                text=text,
                font_size=50,
                color='white',
                font=font,
                text_align='center',
                margin=(20, 20)
            )
            txt_clip = txt_clip.with_position('bottom').with_start(start_time).with_end(end_time)
            txt_clips.append(txt_clip)
        
        final_video = CompositeVideoClip([video] + txt_clips)
        final_video.write_videofile(output_path)
        
        video.close()
        final_video.close()



def generate_shots(reel_gen:ReelGenerator,story:str,num_shot:int=3,is_continues_shot = False):
    # Generate shots
    system = CONTINUOUS_SHOT_SYSTEM if is_continues_shot else SHOT_SYSTEM
    shots = reel_gen.generate_shots(story, system.replace("<num_shot>",str(num_shot)))
    return shots

def generate_shot_image(reel_gen:ReelGenerator,shots:dict,timestamp:str, seed:int=0,cfg_scale:float = 6.5, similarity_strength:float = 0.8, is_continues_shot = False):
    # Create directories for outputs
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join('shot_images', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images for each shot
    image_files = []
    prompts = []
    for idx, shot in enumerate(shots['shots']):
        # optimize prompt for canvas
        prompt,negative_prompt = optimize_canvas_prompt(shot['caption'])
        save_path = os.path.join(output_dir, f'shot_{idx}.png')
        
        if not image_files:  # First image
            ret = reel_gen.generate_text2img(prompt,negative_prompt, save_path,seed,cfg_scale)
        else:
            ret = reel_gen.generate_variations(image_files,prompt,negative_prompt,save_path,seed,cfg_scale,similarity_strength)
        if ret:
            image_files.append(save_path)
            prompts.append(prompt)

        if is_continues_shot: #continues_shot only generates the first image
            break
        time.sleep(10)  # Rate limiting
    return image_files

def generate_reel_prompts(reel_gen:ReelGenerator, shots:dict,image_files:list, skip:bool = True):
    # Read PDF document for prompt optimization
    with open(DEFAULT_GUIDELINE, "rb") as file:
        doc_bytes = file.read()
    
    # Optimize prompts for video generation
    reel_prompts = []
    for shot, ref_img in zip(shots['shots'], image_files):
        if skip:
            user_prompt = f"{shot['prompt']} {shot['cinematography']}"
            reel_prompts.append(user_prompt)
        else:
            optimized_prompt = reel_gen.optimize_reel_prompt(
                system_prompt=SYSTEM_IMAGE_TEXT,
                user_prompt=f"{shot['caption']} {shot['cinematography']}",
                ref_image=ref_img,
                doc_bytes=doc_bytes
            )
            reel_prompts.append(optimized_prompt)
        
    return reel_prompts

def generate_shot_vidoes(reel_gen:ReelGenerator,image_files:list,reel_prompts:list):        
    # Generate videos
    invocation_arns = []
    for prompt, image_file in zip(reel_prompts, image_files):
        invocation = reel_gen.generate_video(prompt, image_file)
        if invocation:
            invocation_arns.append(invocation['invocationArn'])
    
    # Wait for video generation to complete
    final_responses = reel_gen.fetch_job_status(invocation_arns)

    # Download generated videos
    video_files = []
    for response in final_responses:
        output_uri = response['outputDataConfig']['s3OutputDataConfig']['s3Uri'] + '/output.mp4'
        video_file = reel_gen.download_video_from_s3(output_uri, './generated_videos')
        if video_file:
            video_files.append(video_file)
    return video_files

def sistch_vidoes(reel_gen:ReelGenerator,video_files:list,shots:dict,timestamp:str):      
    # Stitch videos together
    final_video = None
    prefix = random_string_name()
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(os.path.join('generated_videos',timestamp), exist_ok=True) 
    for idx in range(len(video_files) - 1):
        output_path = os.path.join('generated_videos',timestamp,f'{prefix}_stitched_{idx}.mp4')
        if not final_video:
            final_video = reel_gen.stitch_videos(video_files[idx], video_files[idx + 1], output_path)
        else:
            final_video = reel_gen.stitch_videos(final_video, video_files[idx + 1], output_path)
    
    # Add captions
    if final_video:
        duration = 6  # Duration per shot
        captions = []
        for idx, shot in enumerate(shots['shots']):
            desc_arr = reel_gen._split_caption(shot['caption'])
            for idy, sub_desc in enumerate(desc_arr):
                if sub_desc:  # Only add non-empty captions
                    start_time = idx * duration + (idy * duration / len(desc_arr))
                    end_time = idx * duration + ((idy + 1) * duration / len(desc_arr))
                    captions.append((sub_desc, start_time, end_time))
        
        caption_video_file = os.path.splitext(final_video)[0] + "_caption.mp4"
        reel_gen.add_timed_captions(final_video, caption_video_file, captions)
        print(f"Final video with captions saved to: {caption_video_file}")
    return final_video,caption_video_file

def extract_last_frame(video_path: str, output_path: str):
    """
    Extracts the last frame of a video file.

    Args:
        video_path (str): The local path to the video to extract the last frame from.
        output_path (str): The local path to save the extracted frame to.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Move to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # Read the last frame
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if ret:
        # Save the last frame as an image
        cv2.imwrite(output_path, frame)
        # print(f"Last frame saved as {output_path}")
    else:
        print("Error: Could not read the last frame.")

    # Release the video capture object
    cap.release()