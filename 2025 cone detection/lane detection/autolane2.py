import cv2
import numpy as np
import RPi.GPIO as GPIO

# GPIO pins connected to the L298N
enable_pin = 18
in1_pin = 23
in2_pin = 24

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(in1_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)

# Function to move linear actuator forward
def move_linear_actuator_forward():
    GPIO.output(in1_pin, GPIO.HIGH)
    GPIO.output(in2_pin, GPIO.LOW)
    GPIO.output(enable_pin, GPIO.HIGH)

# Function to move linear actuator backward
def move_linear_actuator_backward():
    GPIO.output(in1_pin, GPIO.LOW)
    GPIO.output(in2_pin, GPIO.HIGH)
    GPIO.output(enable_pin, GPIO.HIGH)

# Function to stop linear actuator
def stop_linear_actuator():
    GPIO.output(enable_pin, GPIO.LOW)

# Function for lane detection and actuator control
def lane_detection_and_control(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width // 2, height // 2)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # Control logic for actuator based on lane detection
    if lines is not None:
        # Extract left and right lane lines
        left_lane_lines = []
        right_lane_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if slope < -0.5:  # Left lane line (slope is negative)
                left_lane_lines.append(line)
            elif slope > 0.5:  # Right lane line (slope is positive)
                right_lane_lines.append(line)

        # Draw left lane lines in green
        for line in left_lane_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw right lane lines in green
        for line in right_lane_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate the center between left and right lane lines
        if left_lane_lines and right_lane_lines:
            left_x_avg = np.mean([line[0][0] for line in left_lane_lines])
            right_x_avg = np.mean([line[0][2] for line in right_lane_lines])
            center_x = int((left_x_avg + right_x_avg) / 2)
            cv2.line(frame, (int(width/2), height), (center_x, 0), (0, 0, 255), 3)  # Draw center line
            if left_x_avg < right_x_avg:
                move_linear_actuator_forward()  # Vehicle leaning towards left lane
            else:
                move_linear_actuator_backward()  # Vehicle leaning towards right lane
        elif left_lane_lines:
            move_linear_actuator_forward()  # Vehicle leaning towards left lane
        elif right_lane_lines:
            move_linear_actuator_backward()  # Vehicle leaning towards right lane
        else:
            stop_linear_actuator()  # No lane detected

    else:
        stop_linear_actuator()  # No lane detected

    return frame

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform lane detection and actuator control
    processed_frame = lane_detection_and_control(frame)

    # Display processed frame
    cv2.imshow("Autonomous Vehicle Control", processed_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release video capture, cleanup GPIO, and close all windows
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

