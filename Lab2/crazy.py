import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


random_array = np.random.randint(0, 10, size=(100000, 9))
labels = []
for i in random_array:
    _, counts = np.unique(i, return_counts=True)
    if np.any(counts > 2):
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
print(f"Num of triplets {np.sum(labels) / len(labels)}")

classifier = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    max_samples=0.7,
)
data_train, data_test, labels_train, labels_test = train_test_split(
    random_array, labels, test_size=0.5
)
classifier.fit(data_train, labels_train)
final_test_score = np.sum(classifier.predict(data_test) == labels_test) / len(
    labels_test
)

print(f"Accuracy: {final_test_score}")

# Define the model
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(data_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  # Use 'categorical_crossentropy' for classification problems
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    data_train, labels_train, epochs=100, batch_size=100, validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(data_test, labels_test)
print(f"Test Accuracy (CNN): {test_accuracy}")
