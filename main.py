from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import random
from torchvision import models, transforms
import torch
import logging
import requests
from fastapi.responses import JSONResponse

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up database
Base = declarative_base()
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define database model
class ImageTest(Base):
    __tablename__ = "image_tests"
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, index=True)
    feedback = Column(String)

Base.metadata.create_all(bind=engine)

# Initialize AI models
logger.info("Initializing AI models...")
try:
    prompt_generator = pipeline("text-generation", model="gpt2")
    image_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    classification_model = models.resnet50(pretrained=True)
    classification_model.eval()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load AI models: {e}")
    raise RuntimeError("Model initialization failed.")

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load labels for classification
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    response = requests.get(LABELS_URL)
    response.raise_for_status()
    labels = response.json()  
    logger.info(f"Labels loaded successfully. Total labels: {len(labels)}")
except Exception as e:
    logger.error(f"Failed to load labels: {e}")
    labels = []

# Helper function to save image metadata
def save_image_metadata(image_path, feedback=None):
    with SessionLocal() as db:
        image_test = ImageTest(image_path=image_path, feedback=feedback)
        db.add(image_test)
        db.commit()
        logger.info(f"Image metadata saved: {image_path}")
        return image_test.id

# Helper function to generate random meaningful prompts
def get_random_prompts():
    prompts = [
        "Illustrate a peaceful countryside with rolling hills and a small cottage.",
        "Draw an enchanted forest with glowing mushrooms and a magical creature.",
        "Sketch a futuristic cityscape with flying cars and towering skyscrapers.",
        "Create an underwater scene with vibrant coral reefs and exotic sea creatures.",
        "Paint a serene desert at sunset with camels walking in the distance."
    ]
    return random.sample(prompts, 2)

# Route to generate prompts
@app.post("/generate_prompt/")
def generate_prompt(user_input: str = Form(None)):
    try:
        if not user_input:
            example_prompts = get_random_prompts()
            return {
                "message": "Here are some example prompts to draw something:",
                "example_prompts": example_prompts
            }

        generated = prompt_generator(
            f"Generate a meaningful drawing prompt based on: {user_input}",
            max_length=50,
            num_return_sequences=1
        )
        return {
            "generated_prompt": generated[0]["generated_text"].strip(),
            "user_input": user_input
        }
    except Exception as e:
        logger.error(f"Failed to generate prompt: {e}")
        raise HTTPException(status_code=500, detail="Prompt generation failed.")

# Route to generate images
@app.post("/generate_image/")
def generate_image(prompt: str = Form(...)):
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

    try:
        image = image_pipeline(prompt).images[0]
        image_path = os.path.join(output_dir, f"{hash(prompt)}.png")
        image.save(image_path)
        image_id = save_image_metadata(image_path)
        logger.info(f"Image generated and saved: {image_path}")
        return {"image_id": image_id, "image_path": image_path}
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed.")

# Route to classify and tag images
@app.post("/classify_image/")
def classify_image(file: UploadFile = File(...)):
    file_path = f"uploaded_images/{file.filename}"
    os.makedirs("uploaded_images", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        # Load and preprocess the image
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as img_err:
            logger.error(f"Failed to open image {file.filename}: {img_err}")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        input_tensor = image_transforms(image).unsqueeze(0)
        logger.info(f"Image tensor shape: {input_tensor.shape}")

        # Perform classification
        with torch.no_grad():
            outputs = classification_model(input_tensor)
            logger.info(f"Model outputs: {outputs}")
            _, predicted = outputs.topk(3, 1, True, True)
            logger.info(f"Predicted indices: {predicted[0].tolist()}")

            # Ensure labels are properly loaded
            if not labels:
                logger.error("Labels are not loaded.")
                raise HTTPException(status_code=500, detail="Labels are not loaded properly.")

            tags = [labels[idx] for idx in predicted[0].tolist()]

        logger.info(f"Image classified: {file_path}, Tags: {tags}")
        return {"file_path": file_path, "tags": tags}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Classification failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Image classification failed.")

# Route to update feedback for images
@app.post("/submit_feedback/")
def submit_feedback(image_id: int = Form(...), feedback: str = Form(...)):
    with SessionLocal() as db:
        image_test = db.query(ImageTest).filter(ImageTest.id == image_id).first()
        if image_test:
            image_test.feedback = feedback
            db.commit()
            logger.info(f"Feedback updated for image ID {image_id}: {feedback}")
            return {"message": "Feedback submitted successfully!"}
        logger.warning(f"Image ID not found for feedback: {image_id}")
        raise HTTPException(status_code=404, detail="Image not found.")

# Route to fetch all images with metadata
@app.get("/get_images/")
def get_images():
    try:
        with SessionLocal() as db:
            images = db.query(ImageTest).all()
            logger.info(f"Fetched {len(images)} images from the database.")
            return [
                {"id": img.id, "image_path": img.image_path, "feedback": img.feedback}
                for img in images
            ]
    except Exception as e:
        logger.error(f"Failed to fetch images: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch images.")
