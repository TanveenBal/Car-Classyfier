from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image
import cv2
import io

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary for your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_directory = "models/CarStyles"
mapping_file = "mappings/CarStyle map.json"

# Load the car style mappings
with open(mapping_file, "r") as f:
    carStyleMap = json.load(f)

models = {}
for model_file in os.listdir(model_directory):
    if model_file.endswith(".h5"):
        model_name = model_file.split(" ")[-1].split(".")[0]
        models[model_name] = load_model(os.path.join(model_directory, model_file))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if a file is provided
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    try:
        # Open and process the image
        img = Image.open(io.BytesIO(await file.read()))
        img = np.array(img)
        img_resized = cv2.resize(img, (224, 224))
        img_expanded = np.expand_dims(img_resized, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Incorrect file type or image format. Error: {str(e)}")

    results = {}
    try:
        # Predict image on all models
        for model_name, model in models.items():
            yhat = model.predict(img_expanded)
            predicted_class_idx = int(np.argmax(yhat))
            predicted_class = carStyleMap[predicted_class_idx]

            probabilities = {
                carStyleMap[idx]: f"{prob * 100:.2f}%" for idx, prob in enumerate(yhat[0])
            }

            results[model_name] = {
                "predicted_class": predicted_class,
                "probabilities": probabilities
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while predicting. Message: {str(e)}")

    return results

# Run the app with Uvicorn (typically done outside the script)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
