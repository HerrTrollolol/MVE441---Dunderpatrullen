import numpy as np
from numpy.linalg import svd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
import os



data = np.loadtxt("CATSnDOGS.csv", delimiter=",", skiprows=1)
data_labels = np.loadtxt("Labels.csv", delimiter=",", skiprows=1)

image_width = 64  # Adjust according to your image width
image_height = 64  # Adjust according to your image height

# Reshape each row into an image
num_images = data.shape[0]
for i in range(num_images):
    image = data[i].reshape((image_width, image_height))
    current = i +2

    if data_labels[i] == 0:
        label = "Cat"
    else: 
        label = "Dog"
    # Display the image
    plt.imshow(image, cmap='gray')  # Assuming grayscale images
    plt.title(f"Current number: {current}, Current label: {label}")
    plt.axis('off')  # Turn off axis
    plt.show()