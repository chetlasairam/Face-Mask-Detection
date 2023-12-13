import cv2
import os

# Load the Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the video stream
cap = cv2.VideoCapture(0)

# Create the output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Save the detected faces to the output directory
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f'output/face_{i}.jpg', face_img)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
