# Sign_Language_Detection_System

Overview


The Sign Language Detection System is a real-time application that uses computer vision and machine learning to recognize and translate sign language gestures into text. The system employs MediaPipe, OpenCV, and a pre-trained machine learning model to detect hand landmarks and predict sign language gestures.

Prerequisites


Python 3.11 or later.
A webcam connected to your system.


How It Works:-

Hand Detection:
The system uses MediaPipe Hands to detect hand landmarks from the video feed.
Feature Extraction:
Normalized x, y coordinates of the hand landmarks are extracted as features.
Prediction:
A pre-trained machine learning model predicts the corresponding gesture.
Visualization:
The predicted gesture is displayed in real-time on the video feed.
