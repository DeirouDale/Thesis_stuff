import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('image.jpg')

# Set the initial window size
cv2.namedWindow("Leg Tracking with Bounding Box", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Leg Tracking with Bounding Box", 1280, 860)  # Adjust the width and height as needed

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
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

                offset = 30

                x_min -= offset
                y_min -= offset
                x_max += offset
                y_max += offset

                # Draw landmarks and bounding box directly on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Leg Tracking with Bounding Box", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
