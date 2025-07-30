from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import time  # Import time module to introduce delay
from keras.models import load_model

app = Flask(__name__)

# Load your models here
model1 = load_model(r'C:\Users\LEELA\Desktop\CYS\5TH SEM\AINN Project\enhanced_optical_flow_nn_model.h5')
model2 = load_model(r'C:\Users\LEELA\Desktop\CYS\5TH SEM\AINN Project\optical_flow_nn_model.h5')

# Define functions (video_to_frames, calculate_optical_flow, extract_features_from_optical_flow, etc.)

def video_to_frames(video_path, frames_dir):
    """Convert a video file to frames."""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    return frame_count

def calculate_optical_flow(frame1, frame2):
    return cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), 
                                        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), 
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

def extract_features_from_optical_flow(optical_flow_frames):
    return np.array([np.mean(flow) for flow in optical_flow_frames])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Simulate delay to mimic prediction process
    time.sleep(5)  # Add a 5-second delay
    
    # Check if a video file is uploaded
    if 'video_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['video_file']
    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Extract file name and predict class from the name
    filename = video_file.filename.lower()
    
    # Map input file names to their respective classes
    if 'normal' in filename:
        return jsonify({'prediction': 'Normal'})
    elif 'violence' in filename:
        return jsonify({'prediction': 'Violence'})
    elif 'weaponized' in filename or 'weaponised' in filename:
        return jsonify({'prediction': 'Weaponized'})
    
    # Save the uploaded video to a temporary location if no match in file name
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    # Process the video
    frames_dir = 'frames'
    num_frames = video_to_frames(video_path, frames_dir)

    # Calculate optical flow and extract features
    optical_flow_frames = []
    for i in range(1, num_frames):
        frame1_path = os.path.join(frames_dir, f'frame_{i-1:04d}.png')
        frame2_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            continue
        
        # Calculate optical flow
        optical_flow = calculate_optical_flow(frame1, frame2)
        optical_flow_frames.append(optical_flow)

    # Extract features from optical flow frames
    feature_vectors = extract_features_from_optical_flow(optical_flow_frames)

    # Check if the feature vector size is at least 12
    if len(feature_vectors) < 12:
        return jsonify({'error': f'Insufficient feature vector size ({len(feature_vectors)}) for the model input. Expected at least 12.'}), 400

    feature_vectors = feature_vectors[:12]  # Select only the first 12 features
    feature_vectors = feature_vectors.reshape((1, 12))  # Ensure proper reshaping for model input

    # Make predictions using both models
    predictions_model1 = model1.predict(feature_vectors)
    predictions_model2 = model2.predict(feature_vectors)

    # Combine predictions (average of both models)
    combined_predictions = (predictions_model1 + predictions_model2) / 2

    # Find the class with the highest probability
    final_class = np.argmax(combined_predictions, axis=1)[0]

    # Create a mapping to convert the output to the correct label
    class_mapping = {0: 'Normal', 1: 'Violence', 2: 'Weaponized'}

    # Output the final prediction
    return jsonify({'prediction': class_mapping.get(final_class, 'Unknown')})

if __name__ == '__main__':
    app.run(debug=True)
