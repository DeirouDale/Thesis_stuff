import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load your video file (replace 'your_video.mp4' with the path to your video file)
video_path = 'Video.mp4'
cap = cv2.VideoCapture(video_path)

# Set the initial window size
cv2.namedWindow("Leg Tracking with Bounding Box", cv2.WINDOW_NORMAL)

# Initialize MediaPipe Pose with minimum confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

offset = 30
imgSize = 500

# Initialize variables for calculating FPS
prev_time = 0

bounding_box_content_window_name = "Bounding Box Content"
cv2.namedWindow(bounding_box_content_window_name, cv2.WINDOW_NORMAL)

# Flag to indicate if a bounding box is being drawn
drawing_bbox = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define the indices for the left hip, left ankle, and left foot tip landmarks
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_foot_tip = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        # Define the indices for the right hip, right ankle, and right foot tip landmarks
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        right_foot_tip = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        if left_hip and left_ankle and left_foot_tip and right_hip and right_ankle and right_foot_tip:
            # Calculate bounding box coordinates to cover the ankles and foot tips
            x_min = int(min(left_ankle.x, right_ankle.x, left_foot_tip.x, right_foot_tip.x) * frame.shape[1])
            x_max = int(max(left_ankle.x, right_ankle.x, left_foot_tip.x, right_foot_tip.x) * frame.shape[1])
            y_min = int(min(left_hip.y, right_hip.y) * frame.shape[0])
            y_max = int(max(left_ankle.y, right_ankle.y, left_foot_tip.y, right_foot_tip.y) * frame.shape[0])

            x_min -= offset
            y_min -= offset
            x_max += offset
            y_max += offset

            # Draw landmarks and bounding box directly on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_bbox = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Set the drawing_bbox flag to True
            drawing_bbox = True
        else:
            # Set the drawing_bbox flag to False if no bounding box is drawn
            drawing_bbox = False

        if drawing_bbox:
            # Capture the content within the bounding box
            bounding_box_content = frame[y_min:y_max, x_min:x_max]

            # Check if bounding_box_content is not empty
            if bounding_box_content.size > 0:
                # Calculate the dimensions of the bounding box content
                content_height, content_width, _ = bounding_box_content.shape

                # Resize the bounding_box_content window based on the content dimensions
                cv2.namedWindow(bounding_box_content_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(bounding_box_content_window_name, content_width, content_height)

                # Show the bounding box content in the resized window
                cv2.imshow(bounding_box_content_window_name, bounding_box_content)

        # Show the FPS on the main window's upper right corner
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the processed frame with bounding box and FPS
        cv2.imshow("Leg Tracking with Bounding Box", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
