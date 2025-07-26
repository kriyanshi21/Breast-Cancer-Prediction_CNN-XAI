# verify_dataset.py
import os
import cv2
import matplotlib.pyplot as plt

path = 'data/benign'
images = os.listdir(path)

img = cv2.imread(os.path.join(path, images[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Sample Image")
plt.axis('off')
plt.show()
