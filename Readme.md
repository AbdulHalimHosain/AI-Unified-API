# AI Unified API

AI Unified API is a powerful FastAPI-based application that integrates multiple AI models for tasks such as text generation, image generation, and image classification. This project also manages metadata and user feedback using a SQLite database.

## Features

- **Prompt Generation**: Generate creative prompts for image generation.
- **Image Generation**: Create high-quality images using Stable Diffusion.
- **Image Classification**: Classify uploaded images and generate tags using a ResNet model.
- **Feedback Management**: Submit feedback for generated images and store it in the database.
- **Metadata Retrieval**: Retrieve information about all generated and uploaded images.



## Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai_unified_api
   ```
2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
3. **Install dependencies**
4. **Set up the database**
5. **Run The FastAPI**
```bash
uvicorn main:app --reload
```

## Technologies Used
- **FastAPI:** Web framework for building APIs.
- **Stable Diffusion:** Image generation model.
- **ResNet:** Pre-trained image classification model.
- **SQLite:** Database for storing metadata and feedback.
- **Transformers:** Hugging Face library for text generation.