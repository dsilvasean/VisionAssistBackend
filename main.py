from typing import Union

uvicorn main:main --reload

from fastapi import FastAPI

import asyncio

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.intent_module.train import IntentClassifier



app = FastAPI()

PROJECT_ROOT = "/home/sean/repos/project_repos/VisionAssistBackend"
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data/fine_tuned_model')
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'data/label_encoder.pkl')

classifier = IntentClassifier(MODEL_DIR, LABEL_ENCODER_PATH)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# Updated endpoint to accept a string as item_id
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    # Use the string `item_id` for prediction
    predicted_intent = classifier.predict(item_id)
    return {"item_id": item_id, "predicted_intent": predicted_intent}