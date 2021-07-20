from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys

# Create a cascade classifier to find faces in a frame
classifier = cv.CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")

# Class names and colors to draw rectangle around face
class_names = ["With Mask", "No Mask"]
box_colors = [(0, 255, 0), (0, 0, 255)]

# Parameters of trained model
image_side = 128
channels = 3
model = load_model("model_MNet_test")

def predict_live():
    """
    Uses camera to create a live feed, and performs classification on every frame
    :return: None
    """

    # Create a capture object using default camera ID
    cap = cv.VideoCapture(0)

    while True:

        # Read frame and convert it to grayscale
        ret, frame = cap.read()

        # Find the faces in the image using the cascade classifier
        faces = classifier.detectMultiScale(frame, minSize=(150, 150), maxSize=(250, 250))

        # For each face detected
        for x_start, y_start, width, height in faces:

            # Isolate the face region and process it to be compatible with the trained CNN
            face = frame[y_start: y_start + height, x_start: x_start + width]
            face = cv.resize(face, (image_side, image_side))
            face = face.reshape(image_side, image_side, channels)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            # Get Softmax output probabilities, and prediction from that
            probabilities = model.predict(face)
            prediction = np.argmax(probabilities, axis=1)[0]

            # Draw rectangle with corresponding color, as well as class name and confidence level
            cv.rectangle(frame, (x_start, y_start), (x_start+width, y_start+height), box_colors[prediction], 1)
            cv.putText(frame, class_names[prediction], (x_start, y_start - 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 1)
            cv.putText(frame, str(probabilities[0][prediction]), (x_start, y_start - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 1)

        # Show live feed. Press "q" to quit
        cv.imshow("Feed", frame)
        key = cv.waitKey(10)

        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

def predict_image(image_path):

    # Read image from specified path and convert to grayscale
    frame = cv.imread(image_path)
    dims = frame.shape[:2]

    # Find the faces in the image using the cascade classifier
    faces = classifier.detectMultiScale(frame, minSize=(150, 150), maxSize=(250, 250))

    # For each face detected
    for x_start, y_start, width, height in faces:

        # Isolate the face region and process it to be compatible with the trained CNN
        face = frame[y_start: y_start + height, x_start: x_start + width]
        face = cv.resize(face, (image_side, image_side))
        face = face.reshape(image_side, image_side, channels)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        # Get Softmax output probabilities, and prediction from that
        probabilities = model.predict(face)
        prediction = np.argmax(probabilities, axis=1)[0]

        # Draw rectangle with corresponding color, as well as class name and confidence level
        cv.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), box_colors[prediction], 2)
        cv.putText(frame, class_names[prediction], (x_start, y_start - 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 2)
        cv.putText(frame, str(probabilities[0][prediction]), (x_start, y_start - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 2)

    # Display image and wait for user to exit
    cv.imshow("Image", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def predict_video(video_path):
    """
    Uses camera to create a live feed, and performs classification on every frame
    :return: None
    """

    # Capture video from specified path
    cap = cv.VideoCapture(video_path)

    while True:

        # Read frame and convert it to grayscale
        ret, frame = cap.read()

        if not ret:
            break

        # Find the faces in the image using the cascade classifier
        faces = classifier.detectMultiScale(frame, minSize=(500, 500))

        # For each face detected
        for x_start, y_start, width, height in faces:
            # Isolate the face region and process it to be compatible with the trained CNN
            face = frame[y_start: y_start + height, x_start: x_start + width]
            face = cv.resize(face, (image_side, image_side))
            face = face.reshape(image_side, image_side, channels)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            # Get Softmax output probabilities, and prediction from that
            probabilities = model.predict(face)
            prediction = np.argmax(probabilities, axis=1)[0]

            # Draw rectangle with corresponding color, as well as class name and confidence level
            cv.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), box_colors[prediction], 1)
            cv.putText(frame, class_names[prediction], (x_start, y_start - 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 2)
            cv.putText(frame, str(probabilities[0][prediction]), (x_start, y_start - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[prediction], 2)

        # Show live feed. Press "q" to quit
        cv.imshow("Video", frame)
        key = cv.waitKey(10)

        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def usage():
    """
    Print correct usage of program
    :return:
    """

    print("""
    Syntax:\n
    python Predict_MobileNet_V2 live\n
    OR\n
    python Predict_MobileNet_V2 image <image_path>\n
    OR\n
    python Predict_MobileNet_V2 video <video_path>""")


def main():
    # At least one argument must be provided
    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    # first argument specifies type of data to classify
    command = sys.argv[1]

    if command == "live":
        predict_live()
    elif command == "image":
        if len(sys.argv) < 3:
            usage()
            sys.exit(-2)
        else:
            predict_image(sys.argv[2])
    elif command == "video":
        if len(sys.argv) < 3:
            usage()
            sys.exit(-3)
        else:
            predict_video(sys.argv[2])
    else:
        usage()


if __name__ == "__main__":
    main()