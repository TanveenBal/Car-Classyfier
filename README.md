# This is how I created a car make, model, and style guesser!

Car Style Predictor is an application designed to predict the make, model, and style of cars from uploaded images. The backend API leverages machine learning models trained on a variety of car image datasets, while the frontend provides a user-friendly interface for uploading images and displaying predictions.

## Features:
- Upload images via file selection or drag-and-drop interface.
- Predict car make, model, and style from the uploaded image.
- Interactive and responsive UI built with React.
- Backend API developed using FastAPI with trained TensorFlow/Keras models.

Project Structure
```
├── CarBackEnd/
│   ├── app.py                    # API call code for making prediction
│   ├── mappings/                 # JSON mappings for car makes, models, and styles predictions (int -> name)
│   ├── models/                   # Pre-trained car-related ML models
│   └── requirements.txt          # Python library requirements for backend
├── CarFrontEnd/
│   ├── car-prediction-app
│   │   ├── src
│   │   │   ├── App.js            # React-based frontend code
│   │   │   └── App.CSS           # Styles for the frontend    
├── CarBrandMakes.ipynb           # Trains a model on a car's make
├── CarStyle.ipynb                # Trains a model on a car's style
└── CarBrandModels.py             # Trains multiple models to predict a car's model
```
## Backend
### 1. Dependencies

Ensure you have the required libraries installed:

`pip install -r requirements.txt`

### 2. Core Logic
    Model Loading:
    Three TensorFlow/Keras models are loaded:
        A make prediction model.
        A model prediction model.
        A style prediction model.


#### Pipeline:
1. Image is resized and preprocessed.
2. make is predicted using the make model.
3. The corresponding make model's predictions are used to determine the model.
4. Style predictions are inferred using the style prediction model.

```
       +-----------------+
       |   Input Image   |
       +-----------------+
                |
                v
 +-----------------------------+
 |   Predict Car Make Using    |
 |   "Make Prediction Model"   |
 +-----------------------------+
                |
                v
 +-----------------------------+
 |  Get Specific Make's Model  |
 |   Based on Predicted Make   |
 +-----------------------------+
                |
                v
 +-----------------------------+
 |   Predict Car Model Using   |
 |   "Model Prediction Model"  |
 +-----------------------------+  
```

### 3. API Endpoint

```javascript
POST /predict
    Accepts image files (JPEG/PNG).
    Returns a JSON response with predictions:

    {
        "make": "Toyota",
        "model": "Camry",
        "style": "Sedan"
    }
```

Run the Backend:

python app.py

Access the API at: http://127.0.0.1:8000.
## Frontend
### 1. Setup

#### Install React and dependencies:

`npm install`

### 2. Features
- Drag-and-drop or select an image file for uploading.
- Previews the uploaded image.
- Displays predictions retrieved from the backend.

#### Run the Frontend:

`npm start`

Access the app at: http://localhost:3000.

## Training Insights

The backend models were trained using 18 distinct deep learning base models to evaluate their performance. Each model's accuracy and loss metrics were traced across training and validation datasets to identify the best-performing models:

    Base architectures included popular CNNs such as InceptionResNetV2, ResNet, and VGG.
    The highest performance was achieved using InceptionResNetV2 with accuracies:
        88.76% on CarBrandMakeModel.
        96.63% on CarStyleModel.

Training involved fine-tuning and transfer learning techniques to adapt pre-trained models to the car image dataset.
Getting Started
Prerequisites

    Install Python 3.7 or above.
    Install Node.js and npm.
    Ensure TensorFlow is compatible with your system's hardware (e.g., GPU).

