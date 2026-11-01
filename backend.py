"""
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np

#To read an image, detect mask and show it
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.keras')
img=cv2.imread("mask1.png")
face = facemodel.detectMultiScale(img)
for(x,y,l,w) in face:
    crop_face=img[y:y+w,x:x+l]
    cv2.imwrite('temp.jpg',crop_face)  #It will save the image in the pc
    crop_face=load_img('temp.jpg',target_size=(150,150,3))
    crop_face=img_to_array(crop_face)
    crop_face=np.expand_dims(crop_face,axis=0)
    pred=maskmodel.predict(crop_face)[0][0]
    if pred==1:
        cv2.rectangle(img,(x,y),(x+1,y+w),(0,0,255),4)
    else:
        cv2.rectangle(img,(x,y),(x+1,y+w),(0,255,0),4)
cv2.namedWindow("myWindow", cv2.WINDOW_NORMAL)
cv2.imshow("myWindow", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

#For video


import cv2
import os
import time
import numpy as np
from keras.models import load_model

# Load models
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("mask.keras")

# Create folder to save people without masks
save_path = r"C:\projects\Face_Mask\Scripts\Fetched_img"
os.makedirs(save_path, exist_ok=True)

# Open video
vid = cv2.VideoCapture("group.mp4")

if not vid.isOpened():
    raise FileNotFoundError("Could not open mask.mp4")

count = 0
frame_count = 0

while vid.isOpened():

    flag, frame = vid.read()

    if not flag:
        break

    frame_count += 1

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facemodel.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Crop face
        crop_face = frame[y:y+h, x:x+w]

        # Resize face
        crop_face = cv2.resize(crop_face, (150, 150))

        # Convert BGR to RGB
        crop_face = cv2.cvtColor(
            crop_face,
            cv2.COLOR_BGR2RGB
        )

        # Normalize image
        crop_face = crop_face.astype("float32") / 255.0

        # Expand dimensions
        crop_face = np.expand_dims(crop_face, axis=0)

        # Predict mask
        pred = maskmodel.predict(crop_face, verbose=0)[0][0]

        # Assuming class 1 = Without Mask
        if pred >= 0.5:

            label = "No Mask"
            color = (0, 0, 255)

            # Save face every 2 seconds (assuming 30 FPS)
            if frame_count % 60 == 0:

                count += 1

                path = f"No_Mask/person_{count}.jpg"

                cv2.imwrite(path, frame[y:y+h, x:x+w])

                print("Saved:", path)

        else:

            label = "Mask"
            color = (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            3
        )

        # Display prediction
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Display video
    cv2.namedWindow("Mask Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Mask Detection", frame)

    # Press X to stop
    key = cv2.waitKey(20)

    if key == ord('x'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()

print("Video processing completed!")
print("Total saved images:", count)
