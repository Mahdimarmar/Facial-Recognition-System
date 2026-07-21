# Facial Age, Gender & Race Prediction System

## Overview
A deep learning system that predicts age, gender, and race from facial images using CNN and TensorFlow. Features real-time inference through an interactive Streamlit web application.

## Features
- ✅ Multi-output CNN for simultaneous predictions
- ✅ Trained on 23,000+ UTKFace images
- ✅ Real-time face detection and preprocessing
- ✅ Live camera capture support
- ✅ Interactive Streamlit web interface
- ✅ High accuracy predictions with confidence scores

## Technologies Used
- **Deep Learning:** TensorFlow, Keras, CNN
- **Computer Vision:** OpenCV
- **Frontend:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Language:** Python

## How to Run

### Prerequisites
```bash
pip install tensorflow opencv-python streamlit numpy pandas
```

### Run the Application
```bash
streamlit run app.py
```

Then open your browser and go to `localhost:8501`

## Model Architecture
- Input: RGB images (224x224)
- Convolutional layers with batch normalization
- Multi-output heads for age, gender, race
- Trained on UTKFace dataset

## Results
- Age prediction: MAE ~3 years
- Gender classification: 95%+ accuracy
- Race classification: 90%+ accuracy

## Future Improvements
- Real-time video stream processing
- Model optimization for mobile deployment
- Additional demographic predictions

## Author
Mahdi Marmar
