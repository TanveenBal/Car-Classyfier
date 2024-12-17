import React, { useState, useCallback } from 'react';
import axios from 'axios';
import './App.css';

function CarStylePredictor() {
    const [file, setFile] = useState(null);
    const [imageSrc, setImageSrc] = useState('');
    const [predictions, setPredictions] = useState({});
    const [error, setError] = useState('');
    const [dragging, setDragging] = useState(false);

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragging(false);
        const droppedFile = e.dataTransfer.files[0];
        setFile(droppedFile);
        handleFilePreview(droppedFile);
    };

    // Handle file change from input
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        handleFilePreview(selectedFile);
    };

    // Preview the selected or dropped file
    const handleFilePreview = (selectedFile) => {
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                setImageSrc(e.target.result);
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    // Handle form submission and prediction
    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!file) {
            setError('Please select or drop an image file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            setError('');
            const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            if (response.data.error) {
                setError(response.data.error);
                setPredictions({});
            } else {
                setPredictions(response.data);
            }
        } catch (error) {
            setError('Error in prediction. Please try again.');
            console.error(error);
        }
    };

    return (
        <div className="container">
            <h1 className="title">Car Style Predictor</h1>

            {/* Drag and Drop Area */}
            <div
                className={`drop-zone ${dragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <p>Drag and drop an image file here, or</p>
                <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
            </div>

            <button type="submit" className="submit-button" onClick={handleSubmit}>Predict</button>

            {error && <p className="error-message">{error}</p>}

            {imageSrc && (
                <div className="image-preview">
                    <h2>Uploaded Image</h2>
                    <img src={imageSrc} alt="Uploaded Car" className="uploaded-image" />
                </div>
            )}

            {Object.keys(predictions).length > 0 && (
                <div className="results">
                    <h2>Predictions from Models:</h2>
                    {Object.entries(predictions).map(([modelName, result]) => (
                        <div key={modelName} className="model-result">
                            <h3>{modelName} Prediction</h3>
                            <p>Predicted Class: {result.predicted_class}</p>
                            <h4>Probabilities:</h4>
                            <ul className="probabilities-list">
                                {Object.entries(result.probabilities).map(([className, probability]) => (
                                    <li key={className}>
                                        {className}: {probability}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default CarStylePredictor;
