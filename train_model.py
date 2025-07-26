import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path setup
benign_path = 'data/benign'
malignant_path = 'data/malignant'
IMG_SIZE = 128

# Load and preprocess data
def load_images(path, label):
    data = []
    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append((img, label))
        except:
            continue
    return data

benign_data = load_images(benign_path, 0)
malignant_data = load_images(malignant_path, 1)

# Combine & shuffle
dataset = benign_data + malignant_data
np.random.shuffle(dataset)

X = np.array([x for x, _ in dataset]) / 255.0
y = to_categorical([y for _, y in dataset])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("breast_cancer_cnn.h5")
print("✅ Model saved as breast_cancer_cnn.h5")
