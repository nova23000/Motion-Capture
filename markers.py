import cv2
import mediapipe as mp
from tkinter import *
from PIL import Image, ImageTk
import numpy as np  # Add this import statement

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize pose detection
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def detect_and_calculate_angle(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        # Extracting specific points
        joints = {
            'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
            'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
            'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
            'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
            'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
            'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
            'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        }
        
        # Calculate angles
        angles = {
            'left_elbow': calculate_angle(joints['left_shoulder'], joints['left_elbow'], joints['left_wrist']),
            'right_elbow': calculate_angle(joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']),
            'left_knee': calculate_angle(joints['left_hip'], joints['left_knee'], joints['left_ankle']),
            'right_knee': calculate_angle(joints['right_hip'], joints['right_knee'], joints['right_ankle']),
            'left_shoulder': calculate_angle(joints['left_elbow'], joints['left_shoulder'], joints['left_hip']),
            'right_shoulder': calculate_angle(joints['right_elbow'], joints['right_shoulder'], joints['right_hip']),
            'left_hip': calculate_angle(joints['left_shoulder'], joints['left_hip'], joints['left_knee']),
            'right_hip': calculate_angle(joints['right_shoulder'], joints['right_hip'], joints['right_knee'])
        }
        
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        angles = {}
    
    return frame, angles

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame, angles = detect_and_calculate_angle(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    root.after(10, update_frame)

cap = cv2.VideoCapture(0)

root = Tk()
root.title("Real-Time Pose Detection with Angle Calculation")

panel = Label(root)  # Define panel here
panel.pack()

root.after(10, update_frame)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
