# Sign-Language-Recognition
A real-time Sign Language Detection system that recognizes hand gestures and converts them into text using Computer Vision and Deep Learning. The project is designed to provide a stable, accurate, and user-friendly communication aid for hearing and speech-impaired users.

📌 Project Overview

Communication between sign language users and non-signers is often challenging due to the lack of accessible and real-time translation tools. This project addresses the problem by detecting hand gestures through a webcam and translating them into corresponding alphabet or digit outputs in real time.

The system emphasizes accuracy, stability, and real-time performance, making it suitable for practical use on CPU-based systems.

🎯 Features

Real-time hand gesture recognition

Supports alphabets (A–Z) 

Stable predictions using temporal smoothing

Hand jitter reduction using filtering techniques

Interactive UI with:

Space

Backspace

Clear

Period (.)

Optimized for real-time performance on CPU

🧠 System Architecture

Webcam captures real-time video input

Hand landmarks detected using MediaPipe

Preprocessed landmarks passed to a deep learning model

Gesture classified using a CNN-based model

Prediction smoothing and hold-based confirmation

Output displayed as text in the UI

🛠️ Technologies Used

Python

OpenCV – Video capture and image processing

MediaPipe – Hand landmark detection

PyTorch – Model training and inference

Torchvision – Pretrained models

Albumentations – Data augmentation

ONNX & ONNX Runtime – Model optimization

NumPy, Scikit-learn – Data processing and evaluation
