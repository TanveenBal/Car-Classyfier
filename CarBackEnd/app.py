from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import uvicorn
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

car_model_models_dir = "models/CarBrandsModels"
car_model_map_dir = "mappings/Car Model Mappings"

# Load the model prediction mappings
with open("mappings/CarBrandMake map.json", "r") as f:
    carMakeMap = json.load(f)
with open("mappings/CarStyle map.json", "r") as f:
    carStyleMap = json.load(f)

car_make_model = load_model("models/CarBrandsMakes/CarBrandMakeModel 88.76% InceptionResNetV2.h5")
car_style_model = load_model("models/CarStyles/CarStyle 96.63% InceptionResNetV2.h5")

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
        # Predict the make of the car
        make_predictions = car_make_model.predict(img_expanded)
        make_prediction = int(np.argmax(make_predictions))
        make_prediction = carMakeMap[make_prediction]

        # Find the CNN model corresponding to the make that was predicted previously
        car_model_model = None
        for model_file in os.listdir(car_model_models_dir):
            if model_file.startswith(make_prediction):
                car_model_model = load_model(os.path.join(car_model_models_dir, model_file))
                for map_file in os.listdir(car_model_map_dir):
                    if map_file.startswith(make_prediction):
                        with open(os.path.join(car_model_map_dir, map_file), "r") as f:
                            carModelMap = json.load(f)
                        break
                break

        # Debugging
        if not car_model_model:
            raise HTTPException(status_code=400, detail=f"Error: Model of car not found")
        
        # Predict the model of the car
        model_predictions = car_model_model.predict(img_expanded)
        model_prediction = int(np.argmax(model_predictions))
        model_prediction = carModelMap[model_prediction]
 
        # Predict the style of the car
        style_predictions = car_style_model.predict(img_expanded)
        style_prediction = int(np.argmax(style_predictions))
        style_prediction = carStyleMap[style_prediction]

        results = {
            "make": make_prediction,
            "model": model_prediction,
            "style": style_prediction
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error while predicting. Message: {str(e)}")
    return results

# Run the app with Uvicorn (for development mode with reload)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

