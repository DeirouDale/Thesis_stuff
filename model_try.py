import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

new_model = load_model('model/CNN/GaitPhaseClassifier.h5')

img = cv2.imread('test1.jpg')

resize = cv2.resize(img, (256, 256))
normalized_img = resize / 255.0

yhat_single = new_model.predict(np.expand_dims(normalized_img, axis=0))
predicted_class = int(np.argmax(yhat_single, axis=1))

print("CLASSIFICATION")
print("=============================================================================")
if predicted_class == 0:
    print("Phase 1")
elif predicted_class == 1:
    print("Phase 2")
elif predicted_class == 2:
    print("Phase 3")
elif predicted_class == 3:
    print("Phase 4")
elif predicted_class == 4:
    print("Phase 5")
elif predicted_class == 5:
    print("Phase 6")
elif predicted_class == 6:
    print("Phase 7")
elif predicted_class == 7:
    print("Phase 8")

print("=============================================================================")

#pip install tensorflow tensorflow-gpu opencv-python matplotlib