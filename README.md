# Nova Reel Prompt Optimizer

A Gradio web interface for optimizing prompts and generating videos using Amazon's Nova Reel service.

## Features

- Text prompt optimization for video generation
- Support for image-to-video generation with optional image input
- Real-time prompt optimization using Claude
- Video generation using Nova Reel
- User-friendly web interface

## Prerequisites

- Python 3.8+
- AWS credentials configured with access to Bedrock and S3
- An S3 bucket for video output

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Make sure you have AWS credentials configured with appropriate permissions for:
   - Amazon Bedrock (Claude and Nova Reel)
   - S3 bucket access

3. Update the S3 bucket name in `app.py`:
   ```python
   BUCKET = "s3://your-bucket-name"  # Replace with your S3 bucket
   ```

## Usage

1. Start the Gradio interface:

```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

3. In the web interface:
   - Enter your prompt in the text box
   - Optionally upload an image
   - Click "Generate" to start the process
   - View the optimized prompt and generated video

## Notes

- The application uses Claude to optimize prompts for better video generation results
- Video generation may take several minutes to complete
- Generated videos are temporarily stored in the `generated_videos` directory
- Both English and Chinese prompts are supported, but optimized prompts will be in English

## Example Prompts

Text-only example:
```
A serene lake at sunset with gentle ripples on the water
```

Image + text example:
```
Transform this winter scene into a magical snowfall
