import tensorflow as tf
import os
import json
import logging
from keras.applications import (
    MobileNet, MobileNetV2, MobileNetV3Large,
    InceptionV3, InceptionResNetV2,
    ResNet50, ResNet50V2,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2L, EfficientNetV2S
)
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import Precision, Recall, SparseCategoricalAccuracy

# Setup logging configuration
logging.basicConfig(
    filename='training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Function to build models
def build_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Define available pre-trained models
pretrained_models = {
    'MobileNet': MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'MobileNetV3Large': MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'InceptionResNetV2': InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'ResNet50V2': ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'DenseNet169': DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'DenseNet201': DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'EfficientNetV2B0': EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'EfficientNetV2B3': EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'EfficientNetV2L': EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'EfficientNetV2S': EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(e)

base_dir = "Car Models"
img_size = (224, 224)
batch_size = 16

# Iterate through car brand directories
brand_name = "mazda"
model_dir = os.path.join(base_dir, brand_name)

# Load the image data
data = tf.keras.utils.image_dataset_from_directory(
    model_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
    interpolation='bilinear'
)

class_names = data.class_names
with open(f'CarBackEnd/mappings/Car Model Mappings/Car {brand_name} Model map.json', 'w') as f:
    json.dump(class_names, f)

num_classes = len(class_names)

# Split the dataset into training, validation, and test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

best_accuracy = 75.00
best_model_name = "InceptionResNetV2"

# Iterate through each pre-trained model
for model_name, base_model in pretrained_models.items():
    model = build_model(base_model, num_classes)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{brand_name}_{model_name}')

    logging.info(f"Training {model_name} for {brand_name}...")
    try:
        hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
    except Exception as e:
        logging.error(f"Error training {model_name}: {e}")
        continue

    # Evaluate the model
    pre = Precision()
    re = Recall()
    acc = SparseCategoricalAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)

        yhat_classes = tf.argmax(yhat, axis=1)

        pre.update_state(y, yhat_classes)
        re.update_state(y, yhat_classes)
        acc.update_state(y, yhat)

    current_accuracy = acc.result().numpy() * 100

    logging.info(f"{model_name} for {brand_name} achieved {current_accuracy:.2f}% accuracy.")

    # Check if this model performs better than the previous ones
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model_name = f"{brand_name} Model {current_accuracy:.2f}% {model_name}.h5"

        # Save the best model
        model.save(os.path.join('CarBackEnd/models/CarBrandsModels', best_model_name))

        # Delete the previous worse model, if it exists
        previous_models = [file for file in os.listdir("CarBackEnd/models/CarBrandsModels") if
                           brand_name in file and file != best_model_name]
        for prev_model in previous_models:
            os.remove(os.path.join("CarBackEnd/models/CarBrandsModels", prev_model))
            logging.info(f"Deleted worse model: {prev_model}")

logging.info("Training completed for all models and brands.")
