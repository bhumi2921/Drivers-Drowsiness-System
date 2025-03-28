from scipy.spatial import distance
import cv2
import time
import pygame

# Initialize Pygame for buzzer sound
pygame.mixer.init()
pygame.mixer.music.load("music.wav")

# Timer settings
CLOSE_TIME_THRESHOLD = 3  # Time in seconds to trigger buzzer
eye_closed_start_time = None  # Track when eyes first close
buzzer_active = False  # Track if buzzer is already on

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed!")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    eyes_closed = False  # Default state (not closed)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) == 0:  # No eyes detected = closed
            eyes_closed = True

    # Timer-based buzzer logic
    if eyes_closed:
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()  # Start timer
        elif time.time() - eye_closed_start_time >= CLOSE_TIME_THRESHOLD and not buzzer_active:
            print("🚨 ALERT: Eyes closed for 3+ seconds! Triggering buzzer.")
            pygame.mixer.music.play(-1)  # Play buzzer in a loop
            buzzer_active = True
    else:
        eye_closed_start_time = None  # Reset timer if eyes open
        if buzzer_active:
            pygame.mixer.music.stop()  # Stop buzzer
            buzzer_active = False

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
