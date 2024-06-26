import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix






def load_and_preprocess_images(clear_skin, pimples_images, target_size=(224, 224)):
    # Initialize lists to store preprocessed images and labels (0 for clear skin, 1 for pimples)
    images = []
    labels = []

    # Load and preprocess images with clear skin
    for filename in os.listdir(clear_skin):
        if filename.endswith(".jpg"):
            image_path = os.path.join(clear_skin, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = image.astype("float") / 255.0
            images.append(image)
            labels.append(0)  # 0 indicates clear skin

    # Load and preprocess images with pimples/acne
    for filename in os.listdir(pimples_images):
        if filename.endswith(".jpg"):
            image_path = os.path.join(pimples_images, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = image.astype("float") / 255.0
            images.append(image)
            labels.append(1)  # 1 indicates presence of pimples/acne

    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Specify the paths to your clear skin and pimples image folders
clear_skin_folder = "clear_skin"
pimples_folder = "pimples_images"

# Load and preprocess images from both folders
images, labels = load_and_preprocess_images(clear_skin_folder, pimples_folder)

# Now, 'images' contains the preprocessed images, and 'labels' contains the corresponding labels.




# Define the CNN architecture
def create_cnn_model(input_shape):
    model = keras.Sequential()

    # Convolutional layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 3 (optional)
    # You can add more convolutional layers for deeper feature extraction

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(1, activation='sigmoid'))  # 1 output node for binary classification

    return model

# Specify the input shape (target size and number of color channels)
input_shape = (224, 224, 3)  # Adjust according to your preprocessing target size and color channels

# Create the CNN model
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()




# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")



# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predicted probabilities to binary labels (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred_binary))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))




#######################################using usb connection###################################3
from flask import Flask, jsonify, render_template
app = Flask(__name__)
# Load the face detection classifier (Haar Cascade or any other)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
@app.route("/")
def index():
    return render_template("index.html")
# Initialize the camera
cap = cv2.VideoCapture(1)  # Use default camera (should automatically use the USB camera)
@app.route("/start_detection")
def start_detection():
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
           return jsonify({"message": "Error capturing frame"})

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    # Loop through detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face = frame[y:y + h, x:x + w]

            # Preprocess the face image for your model
            face = cv2.resize(face, (224, 224))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=0)

            # Make a prediction using your model
            predictions = model.predict(face)

            # Interpret the prediction
            if predictions[0][0] > 0.5:
                label = "Pimples/Acne"
                color = (0, 0, 255)  # Red for pimples
            else:
                label = "Clear Skin"
                color = (0, 255, 0)  # Green for clear skin

            # Draw a bounding box and label on the frame only if pimples are detected
            if label == "Pimples/Acne":
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Facial Health Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Detection completed", "results": detection_results})

    if __name__ == "__main__":
       app.run(debug=True)
