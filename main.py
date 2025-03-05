from typing import Union

# uvicorn main:main --reload
# uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel

import asyncio

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.intent_module.train import IntentClassifier



app = FastAPI()

PROJECT_ROOT = "/home/kabir/repos/VisionAssistBackend"
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data/fine_tuned_model')
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'data/label_encoder.pkl')

classifier = IntentClassifier(MODEL_DIR, LABEL_ENCODER_PATH)

class ItemRequest(BaseModel):
    data: str
    q: Union[str, None] = None

@app.get("/")
async def read_root():
    return {"Hello": "World"}


# Updated endpoint to accept a string as item_id
@app.post("/intent/")
async def get_intent(request: ItemRequest):
    # Use the string `item_id` for prediction
    predicted_intent = classifier.predict(request.data)
    return {"item_id": request.data, "predicted_intent": predicted_intent}