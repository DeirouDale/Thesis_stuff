import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

#calculate angle between 3 points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

new_model = load_model('model/CNN/GaitPhaseClassifier.h5')

# Load your video file (replace 'your_video.mp4' with the path to your video file)
video_path = "Videos\Video1.mp4"
cap = cv2.VideoCapture(video_path)

# Set the initial window size
cv2.namedWindow("Leg Tracking with Bounding Box", cv2.WINDOW_NORMAL)

# Initialize MediaPipe Pose with minimum confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

x_offset = 30
y_offset = 30
imgSize = 500

# Initialize variables for calculating FPS
prev_time = 0
frame_number = 0  # Initialize frame number

#bounding_box_content_window_name = "Bounding Box Content"
#cv2.namedWindow(bounding_box_content_window_name, cv2.WINDOW_NORMAL)

# Flag to indicate if a bounding box is being drawn
drawing_bbox = False
playing = True

font = cv2.FONT_HERSHEY_SIMPLEX
prediction = ""
font_scale = 1
color = (255, 255, 255)  # Text color in BGR format
thickness = 2

target_directory = "data\Phase1" 
while cap.isOpened():


    if playing:
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

        #get coordinates for right hip, knee, ankle, shoulder
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        rfoot_tip = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        #get coordinates for left hip, knee, ankle, shoulder
        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        lfoor_tip = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        #calculate angle for right side
        rangle_knee = calculate_angle(rhip,rknee,rankle)
        rangle_hip = calculate_angle(rshoulder, rhip, rknee)
        rangle_ankle = calculate_angle(rknee, rankle, rfoot_tip)
        rhip_angle = 180-rangle_hip
        rknee_angle = 180-rangle_knee
        rankle_angle = 180-rangle_ankle
        rhip_angle = round(rhip_angle,2)
        rknee_angle = round(rknee_angle,2)
        rankle_angle = round(rankle_angle,2)
        #calculate angle for left side
        langle_knee = calculate_angle(lhip,lknee,lankle)
        langle_hip = calculate_angle(lshoulder, lhip, lknee)
        langle_ankle = calculate_angle(lknee, lankle, lfoor_tip)
        lhip_angle = 180-langle_hip
        lknee_angle = 180-langle_knee
        lankle_angle = 180-langle_ankle
        lhip_angle = round(lhip_angle,2)
        lknee_angle = round(lknee_angle,2)
        lankle_angle = round(lankle_angle,2)
        if left_hip and left_ankle and left_foot_tip and right_hip and right_ankle and right_foot_tip:
            # Calculate bounding box coordinates to cover the ankles and foot tips
            x_min = int(min(left_ankle.x, right_ankle.x, left_foot_tip.x, right_foot_tip.x) * frame.shape[1])
            x_max = int(max(left_ankle.x, right_ankle.x, left_foot_tip.x, right_foot_tip.x) * frame.shape[1])
            y_min = int(min(left_hip.y, right_hip.y) * frame.shape[0])
            y_max = int(max(left_ankle.y, right_ankle.y, left_foot_tip.y, right_foot_tip.y) * frame.shape[0])

            x_min -= x_offset
            y_min -= y_offset
            x_max += x_offset
            y_max += y_offset

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

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # create a white image

            # Check if bounding_box_content is not empty
            if bounding_box_content.size > 0:
                # Calculate the dimensions of the bounding box content
                content_height, content_width, _ = bounding_box_content.shape
                aspectRatio = content_height / content_width

                if aspectRatio > 1:  # if height is greater than width
                    k = imgSize / content_height  # get the ratio of image size to height
                    wCal = math.ceil(k * content_width)  # calculate width
                    imgResize = cv2.resize(bounding_box_content, (wCal, imgSize))  # resize image
                    imgResizeShape = imgResize.shape  # get shape of resized image

                    wGap = math.ceil((500 - wCal) / 2)  # calculate gap
                    imgWhite[:, wGap:wCal + wGap] = imgResize  # add image to white image
                    

                else:  # if width is greater than height
                    k = imgSize / content_width  # get ratio of image size to width
                    hCal = math.ceil(k * content_height)  # calculate height
                    imgResize = cv2.resize(bounding_box_content, (imgSize, hCal))  # resize image
                    imgResizeShape = imgResize.shape  # get shape of resized image

                    hGap = math.ceil((500 - hCal) / 2)  # calculate gap
                    imgWhite[hGap:hCal + hGap, :] = imgResize  # add image to white image

                
                # Resize the bounding_box_content window based on the content dimensions

                #cv2.namedWindow(bounding_box_content_window_name, cv2.WINDOW_NORMAL)
                #cv2.resizeWindow(bounding_box_content_window_name, content_width, content_height)

                # Show the bounding box content in the resized window
                #cv2.imshow(bounding_box_content_window_name, bounding_box_content)
                #cv2.imshow('ImageWhite', imgWhite)  # show white image

        # Show the FPS on the main window's upper right corner
        resize = cv2.resize(imgWhite, (256, 256))
        normalized_img = resize / 255.0
        yhat_single = new_model.predict(np.expand_dims(normalized_img, axis=0))
        predicted_class = int(np.argmax(yhat_single, axis=1))
        if predicted_class == 0:
            prediction = "Phase 1"
        elif predicted_class == 1:
            prediction = "Phase 2"
        elif predicted_class == 2:
            prediction = "Phase 3"
        elif predicted_class == 3:
            prediction = "Phase 4"
        elif predicted_class == 4:
            prediction = "Phase 5"
        elif predicted_class == 5:
            prediction = "Phase 6"
        elif predicted_class == 6:
            prediction = "Phase 7"
        elif predicted_class == 7:
            prediction = "Phase 8"
        text_size = cv2.getTextSize(prediction, font, font_scale, thickness)[0]
        text_width, text_height = text_size[0], text_size[1]
        text_x = x_min
        text_y = y_max + text_height + 10 
        cv2.putText(frame, prediction, (text_x, text_y), font, font_scale, color, thickness)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle Right Hip: {rhip_angle}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle Right Knee: {rknee_angle}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle Right Ankle: {rankle_angle}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle Left Hip: {lhip_angle}', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Angle Left Knee: {lknee_angle}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  
        cv2.putText(frame, f'Angle Left Ankle: {lankle_angle}', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Show the processed frame with bounding box and FPS
        cv2.imshow("Leg Tracking with Bounding Box", frame)

    # Check for user keyboard input for video navigation
    key = cv2.waitKey(70) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Press 'Space' to toggle play/pause
        playing = not playing
    elif key == ord('d'):
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 5)  # Press 'd' to go forward by 10 frames
    elif key == ord('a'):
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 5, 0))  # Press 'a' to go back by 10 frames
    elif key == ord('s'):  # Press 's' to save the current frame as a JPG image
        frame_filename = os.path.join(target_directory, f'Phase1_{frame_number}.jpg')
    
        # Check if the file already exists in the target directory
        if os.path.isfile(frame_filename):
            # If the file already exists, find a new name with (1), (2), etc.a
            index = 1
            while os.path.isfile(frame_filename):
                frame_filename = os.path.join(target_directory, f'Phase1_{frame_number} ({index}).jpg')
                index += 1

        cv2.imwrite(frame_filename, imgWhite)  # Save the frame as a JPG image
        print(f"Saved frame as {frame_filename}")
cap.release()
cv2.destroyAllWindows()