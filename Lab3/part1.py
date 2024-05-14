import argparse
import os
import sys
import io
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from numpy.linalg import svd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from collections import Counter
from itertools import islice
from sklearn.cluster import KMeans


from mpl_toolkits.axes_grid1 import make_axes_locatable

# from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def suppress_print(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = io.StringIO()  # Redirect stdout to a new StringIO object
        result = func(*args, **kwargs)
        sys.stdout = original_stdout  # Restore the original stdout
        return result

    return wrapper


def load_data(flip_data=False):

    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    data = StandardScaler().fit_transform(data)
    indices = np.arange(len(data))

    data_train, data_test, data_labels_train, data_labels_test, _, indices_test = (
        train_test_split(
            data, data_labels, indices, test_size=0.2, stratify=data_labels
        )
    )

    if flip_data:
        # Determine the number of images to flip
        num_images_to_flip_train = len(data_train) // 2
        num_images_to_flip_test = len(data_test) // 2

        # Randomly select indices of images to flip in the training set
        flip_indices_train = np.random.choice(
            len(data_train), num_images_to_flip_train, replace=False
        )

        # Turn selected training images upside down by reversing their order
        data_train[flip_indices_train] = np.flip(data_train[flip_indices_train], axis=1)

        # Randomly select indices of images to flip in the test set
        flip_indices_test = np.random.choice(
            len(data_test), num_images_to_flip_test, replace=False
        )

        # Turn selected test images upside down by reversing their order
        data_test[flip_indices_test] = np.flip(data_test[flip_indices_test], axis=1)
    else:
        flip_indices_test = None

    return (
        data_train,
        data_labels_train,
        data_test,
        data_labels_test,
        indices_test,
        flip_indices_test,
    )


def RF(
    data_train,
    data_labels_train,
    data_test,
    data_labels_test,
    test_indices=None,
    flip_indices_test=None,
):

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

    classifier.fit(data_train, data_labels_train)
    predictions_test = classifier.predict(data_test)
    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    if test_indices is not None:
        misclassified_indices_test = [
            index
            for index, (pred_label, true_label) in zip(
                test_indices, zip(predictions_test, data_labels_test)
            )
            if pred_label != true_label
        ]

        if flip_indices_test is not None:
            true_predictions = predictions_test == data_labels_test
            counter = 0
            for i in flip_indices_test:
                if not true_predictions[i]:
                    counter += 1
            print(counter / len(misclassified_indices_test))

    else:
        misclassified_indices_test = None

    return (
        final_test_score,
        np.array(classifier.feature_importances_),
        misclassified_indices_test,
        data_labels_test,
        predictions_test,
    )


def SVC_(
    data_train,
    data_labels_train,
    data_test,
    data_labels_test,
    test_indices=None,
    rbf=True,
):
    if rbf:
        classifier = SVC(C=10.0, kernel="rbf", gamma=0.0005)
    else:
        classifier = SVC(C=10.0, kernel="linear")
    classifier.fit(data_train, data_labels_train)
    predictions_test = classifier.predict(data_test)
    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    if not rbf:
        params = classifier.coef_[0]
    else:
        params = []

    if test_indices is not None:
        misclassified_indices_test = [
            index
            for index, (pred_label, true_label) in zip(
                test_indices, zip(predictions_test, data_labels_test)
            )
            if pred_label != true_label
        ]
    else:
        misclassified_indices_test = None

    return (
        final_test_score,
        np.array(params),
        misclassified_indices_test,
        data_labels_test,
        predictions_test,
    )


def GBM(
    data_train,
    data_labels_train,
    data_test,
    data_labels_test,
    test_indices=None,
):

    classifier = GradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.8,
        n_estimators=100,
        max_features="sqrt",
        subsample=1.0,
        min_samples_split=2,
        max_depth=4,
    )

    classifier.fit(data_train, data_labels_train)
    predictions_test = classifier.predict(data_test)
    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)

    if test_indices is not None:
        misclassified_indices_test = [
            index
            for index, (pred_label, true_label) in zip(
                test_indices, zip(predictions_test, data_labels_test)
            )
            if pred_label != true_label
        ]
    else:
        misclassified_indices_test = None

    return (
        final_test_score,
        np.array(classifier.feature_importances_),
        misclassified_indices_test,
        data_labels_test,
        predictions_test,
    )


def lasso(
    data_train,
    data_labels_train,
    data_test,
    data_labels_test,
    test_indices=None,
):
    classifier = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=20,
    )
    classifier.fit(data_train, data_labels_train)
    predictions_test = classifier.predict(data_test)
    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)

    params = classifier.coef_[0]

    if test_indices is not None:
        misclassified_indices_test = [
            index
            for index, (pred_label, true_label) in zip(
                test_indices, zip(predictions_test, data_labels_test)
            )
            if pred_label != true_label
        ]
    else:
        misclassified_indices_test = None

    return (
        final_test_score,
        np.array(params),
        misclassified_indices_test,
        data_labels_test,
        predictions_test,
    )


def NN(
    data_train,
    data_labels_train,
    data_test,
    data_labels_test,
    test_indices=None,
):
    # Convert data to PyTorch tensors
    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_labels_train = torch.tensor(data_labels_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    data_labels_test = torch.tensor(data_labels_test, dtype=torch.float32)

    # Define the neural network architecture
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            if data_train.shape[1] == 64 * 64:
                self.fc1 = nn.Linear(64 * 64, 256)
            else:
                self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 16)
            self.fc4 = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            x = torch.relu(x)
            x = self.fc4(x)
            x = self.sigmoid(x)
            return x

    model = NeuralNetwork()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    save_path = "best_model.pth"

    num_items = len(data_train)  # total number of items in the dataset
    num_train = int(num_items * 0.9)  # 90% for training
    num_val = num_items - num_train  # 10% for validation

    full_dataset = TensorDataset(data_train, data_labels_train)
    train_data, val_data = random_split(full_dataset, [num_train, num_val])

    # Training the model with early stopping
    best_val_loss = float("inf")
    patience = 3  # Patience is the number of epochs to tolerate no improvement
    trigger_times = 0  # How many times the trigger condition has been met

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    # Training the model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                v_loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += v_loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss}")
        else:
            trigger_times += 1
            print(f"No improvement in validation loss for {trigger_times} epochs...")
            if trigger_times >= patience:
                print("Stopping early.")
                break

    # Evaluation on test set
    model = NeuralNetwork()

    def load_model_with_retries(model, save_path, max_attempts=5):
        attempt = 0
        while attempt < max_attempts:
            try:
                model.load_state_dict(torch.load(save_path))
                print("Model loaded successfully.")
                return True  # Return True if loading was successful
            except Exception as e:
                print(f"Failed to load model on attempt {attempt+1}: {e}")
                attempt += 1
                # Optionally, add a delay (e.g., using time.sleep(seconds)) if needed

        print("Reached maximum attempt limit, failed to load model.")
        return False  # Return False if all attempts fail

    # model.load_state_dict(torch.load(save_path))
    success = load_model_with_retries(model, save_path)
    model.eval()
    with torch.no_grad():
        outputs = model(data_test)
        predicted = (
            outputs >= 0.5
        ).float()  # Convert probabilities to binary predictions
        accuracy = (predicted == data_labels_test.unsqueeze(1)).float().mean()

        print(f"Test Accuracy: {accuracy.item()}")

    model.eval()

    # Prepare the data loader for the test dataset
    test_loader = DataLoader(
        TensorDataset(data_test, data_labels_test), batch_size=1, shuffle=False
    )

    # Initialize a tensor to store the sum of all saliency maps
    sum_saliency = None
    count = 0

    # Loop through all images in the test set
    for inputs, _ in test_loader:
        inputs.requires_grad = True
        outputs = model(inputs)

        # Assume binary classification and the model outputs a single value per input
        outputs = outputs.squeeze()  # Remove unnecessary dimensions if present
        outputs.backward()  # Compute the gradients with respect to the input image

        # Compute the saliency map as the absolute value of the gradients
        saliency = inputs.grad.data.abs().squeeze()

        # Accumulate the saliency maps
        if sum_saliency is None:
            sum_saliency = saliency
        else:
            sum_saliency += saliency

        count += 1
        inputs.grad.data.zero_()  # Reset gradients to zero for the next iteration

    # Compute the average saliency map
    avg_saliency = sum_saliency / count

    if test_indices is not None:
        misclassified_indices_test = [
            index
            for index, (pred_label, true_label) in zip(
                test_indices, zip(predicted, data_labels_test)
            )
            if pred_label != true_label
        ]
    else:
        misclassified_indices_test = None

    return (
        accuracy.item(),
        np.array(avg_saliency),
        misclassified_indices_test,
        data_labels_test,
        predicted,
    )


def CV_kernel():
    data_train, data_labels_train, _, _, _, _ = load_data()
    alphas = [10**i for i in range(0, 5)]
    gammas = np.round(np.arange(0.00001, 0.001, 0.0001), 5)
    scores = np.zeros((len(alphas), len(gammas)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i, j = 0, 0
    for alpha in alphas:
        j = 0
        for gamma in gammas:
            accuracies = []
            classifier = SVC(C=alpha, kernel="rbf", gamma=gamma)
            for train_index, test_index in kf.split(data_train):
                # Split data
                X_train, X_test = data_train[train_index], data_train[test_index]
                y_train, y_test = (
                    data_labels_train[train_index],
                    data_labels_train[test_index],
                )

                # Train the model
                classifier.fit(X_train, y_train)

                # Predict on the test set
                predictions = classifier.predict(X_test)

                # Calculate accuracy
                acc = accuracy_score(y_test, predictions)
                accuracies.append(acc)
            score = np.mean(accuracies)
            print(f"For {alpha} and {gamma} we get the score of {score}.")
            scores[i][j] = score
            j += 1
        i += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        scores,
        annot=True,
        fmt=".2f",
        xticklabels=gammas,
        yticklabels=alphas,
        cmap="coolwarm",
    )
    plt.title("CV kernel")
    plt.xlabel("Gammas")
    plt.ylabel("Alphas")
    plt.show()

    return scores


def CV_lasso():
    data_train, data_labels_train, _, _, _, _ = load_data()
    alphas = np.arange(0.1, 30.0, 0.1).tolist()
    scores = np.zeros(len(alphas))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i = 0
    for alpha in alphas:
        accuracies = []
        classifier = classifier = LogisticRegression(
            penalty="l1", solver="liblinear", C=alpha
        )
        for train_index, test_index in kf.split(data_train):
            # Split data
            X_train, X_test = data_train[train_index], data_train[test_index]
            y_train, y_test = (
                data_labels_train[train_index],
                data_labels_train[test_index],
            )

            # Train the model
            classifier.fit(X_train, y_train)

            # Predict on the test set
            predictions = classifier.predict(X_test)

            # Calculate accuracy
            acc = accuracy_score(y_test, predictions)
            accuracies.append(acc)
        score = np.mean(accuracies)
        print(f"For {alpha} we get the score of {score}.")
        scores[i] = score
        i += 1
    plt.figure(figsize=(10, 8))
    plt.plot(alphas, scores)
    plt.title("CV Lasso")
    plt.xlabel("Alphas")
    plt.ylabel("Validation score")
    plt.show()

    return scores


def plot_confusion_matrix(flip_data=False):
    averages = 10
    classifiers = {
        "RF": ([], []),
        "GBM": ([], []),
        "SVC": ([], []),
        "NN": ([], []),
        "Lasso": ([], []),
    }

    for av in range(averages):
        print(f"average [{av+1}/{averages}]")
        data_train, data_labels_train, data_test, data_labels_test, _, _ = load_data(
            flip_data
        )
        _, _, _, RF_true, RF_pred = RF(
            data_train, data_labels_train, data_test, data_labels_test
        )
        classifiers["RF"][0].extend(RF_true)
        classifiers["RF"][1].extend(RF_pred)
        _, _, _, GBM_true, GBM_pred = GBM(
            data_train,
            data_labels_train,
            data_test,
            data_labels_test,
        )
        classifiers["GBM"][0].extend(GBM_true)
        classifiers["GBM"][1].extend(GBM_pred)
        _, _, _, SVC_true, SVC_pred = SVC_(
            data_train, data_labels_train, data_test, data_labels_test
        )
        classifiers["SVC"][0].extend(SVC_true)
        classifiers["SVC"][1].extend(SVC_pred)
        _, _, _, NN_true, NN_pred = suppress_print(NN)(
            data_train, data_labels_train, data_test, data_labels_test
        )
        classifiers["NN"][0].extend(NN_true)
        classifiers["NN"][1].extend(NN_pred)
        _, _, _, Lasso_true, Lasso_pred = lasso(
            data_train, data_labels_train, data_test, data_labels_test
        )
        classifiers["Lasso"][0].extend(Lasso_true)
        classifiers["Lasso"][1].extend(Lasso_pred)

    # Plotting all confusion matrices
    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=1.2)
    for idx, (name, (true, pred)) in enumerate(classifiers.items(), 1):
        cm = confusion_matrix(true, pred)
        plt.subplot(2, 3, idx)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            annot_kws={"size": 12},
            xticklabels=["Class 0 - Catty", "Class 1 - Doggie"],
            yticklabels=["Class 0 - Catty", "Class 1 - Doggie"],
        )
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

    plt.tight_layout()
    plt.show()


def plot_performance(scores):
    print(scores)

    # Creating the plot
    # Get the keys and values from the dictionary
    labels = list(scores.keys())
    values = list(scores.values())

    # Creating the bar plot
    plt.figure(figsize=(10, 6))  # Optional: specifies the size of the figure
    plt.boxplot(values, labels=labels)  # Creates the bar plot

    # Adding titles and labels
    plt.title("Accuracy different models")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")

    # Show the plot
    plt.show()

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))

    # Create a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df)

    # Adding titles and labels
    plt.title("Comparison of Model Performances")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")

    # Show the plot
    plt.show()


def plot_features(model_feature_importances, flip_data):
    _, _, data_test, data_labels, _, _, _ = load_data(flip_data)

    # Filter data based on labels
    label0_data = data_test[data_labels == 0]
    label1_data = data_test[data_labels == 1]

    # Calculate average images for each label
    if len(label0_data) > 0:
        average_image0 = np.mean(label0_data, axis=0)
        model_feature_importances["Average image Label Cat"] = average_image0

    if len(label1_data) > 0:
        average_image1 = np.mean(label1_data, axis=0)
        model_feature_importances["Average image Label Dog"] = average_image1

    model_feature_importances["Average image Label difference"] = abs(
        average_image1 - average_image0
    )

    num_models = len(model_feature_importances)

    # Setup the matplotlib figure and axes
    cols = 3
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Normalize and plot each image with its own colorbar
    for ax, (model_name, image_array) in zip(axes, model_feature_importances.items()):
        # Normalize feature importance maps
        if "Average" not in model_name:
            image_array = np.abs(image_array) / np.sum(np.abs(image_array))
        image_matrix = image_array.reshape(64, 64).T  # Assuming image shape of 64x64

        # Choose color map
        cmap = "gray" if "Average" in model_name else "hot"

        # Display the image
        im = ax.imshow(image_matrix, cmap=cmap, aspect="auto")
        ax.set_title(model_name)
        ax.axis("off")

        # Create a colorbar for each image subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Hide any unused axes
    for i in range(num_models, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_misclassified_indices(misclassified_indices, test_indices_counter):
    print(misclassified_indices)
    classifiers = list(misclassified_indices.keys())
    color_map = plt.get_cmap("tab10")
    bar_width = 0.6
    total_images = 198
    tick_interval = 25

    # Convert test_indices_counter list to a Counter object for easier counting
    test_index_counts = Counter(test_indices_counter)

    # Initialize counters for all image indices (0 to total_images - 1)
    misclassified_counts = {
        classifier: Counter(misclassified_indices[classifier])
        for classifier in classifiers
    }

    # Create a complete range of image indices
    all_indices = list(range(total_images))
    index_positions = np.arange(total_images)

    # Prepare the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 6))
    bottoms = np.zeros(total_images)
    for idx, classifier in enumerate(classifiers):
        y_values = [
            misclassified_counts[classifier].get(image_idx, 0)
            / test_index_counts.get(image_idx, 1)
            for image_idx in all_indices
        ]
        ax.bar(
            index_positions,
            y_values,
            bar_width,
            label=classifier,
            color=color_map(idx % 10),
            bottom=bottoms,
        )
        bottoms += np.array(y_values)

    # Adjust x-axis ticks and labels
    visible_indices = range(0, total_images, tick_interval)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Normalized Misclassified Count")
    ax.set_title("Normalized Number of Times Each Classifier Misclassifies Each Image")
    ax.set_xticks(visible_indices)
    ax.set_xticklabels(visible_indices)
    ax.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust y-axis to fit all stacked bars
    max_count = np.max(bottoms)
    ax.set_ylim(0, max_count + 1)
    ax.grid(axis="y")  # Add horizontal grid lines for better readability
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()
    np.save("accuarcies.npy", bottoms)


# lista med indices och plottar bilderna
def plot_misclassified_images():
    accuracies = np.load("accuarcies.npy")
    indices_sorted = np.argsort(accuracies)[::-1]
    image_shape = (64, 64)  # Define the shape of images
    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    for i in indices_sorted:
        actual_label = data_labels[i]
        predicted_label = "Dog" if actual_label == 0 else "Cat"
        misclassified_image = data[i].reshape(image_shape).T

        # Convert numerical labels to text for the title
        actual_label_text = "Cat" if actual_label == 0 else "Dog"

        # Display the misclassified image with predicted and actual labels
        plt.imshow(misclassified_image, cmap="gray")
        plt.title(
            f" Index: {i}, Predicted: {predicted_label}, Actual: {actual_label_text}, Error rate: {round(accuracies[i]/5,2)}"
        )
        plt.show()


def CV_block():
    # Load your data
    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    # Constants
    block_size = 16
    num_blocks = 4
    total_blocks = num_blocks**2
    k = 5  # Number of folds in k-fold cross-validation

    # Prepare the data in blocks
    def prepare_blocks(data):
        blocks = np.empty((data.shape[0], total_blocks, block_size * block_size))
        for img_index in range(data.shape[0]):
            square_matrix = data[img_index].reshape(64, 64)
            block_counter = 0
            for i in range(num_blocks):
                for j in range(num_blocks):
                    row_start = i * block_size
                    col_start = j * block_size
                    blocks[img_index, block_counter] = square_matrix[
                        row_start : row_start + block_size,
                        col_start : col_start + block_size,
                    ].flatten()
                    block_counter += 1
        return blocks

    blocks = prepare_blocks(data)

    fig, axes = plt.subplots(nrows=num_blocks, ncols=num_blocks, figsize=(8, 8))

    for i in range(num_blocks):
        for j in range(num_blocks):
            ax = axes[i, j]
            block_index = i * num_blocks + j
            block_image = blocks[5, block_index].reshape(block_size, block_size)
            ax.imshow(block_image, cmap="gray", interpolation="none")
            ax.axis("off")  # Turn off axis numbering

    plt.tight_layout()
    plt.show()

    # Initialize scores
    scores_methods = {
        name: np.zeros((total_blocks, k))
        for name in ["RF", "GBM", "NN", "Lasso", "SVC"]
    }

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold_index, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Processing fold {fold_index + 1}/{k}")
        for block_index in range(total_blocks):
            X_train, X_test = (
                blocks[train_index, block_index],
                blocks[test_index, block_index],
            )

            y_train, y_test = data_labels[train_index], data_labels[test_index]

            # RF
            rf_score, _, _, _, _ = RF(X_train, y_train, X_test, y_test)
            scores_methods["RF"][block_index, fold_index] = rf_score

            # GBM
            gbm_score, _, _, _, _ = GBM(X_train, y_train, X_test, y_test)
            scores_methods["GBM"][block_index, fold_index] = gbm_score

            # NN (modify for cross-validation)
            NN_suppressed = suppress_print(NN)
            nn_score, _, _, _, _ = NN_suppressed(X_train, y_train, X_test, y_test)
            scores_methods["NN"][block_index, fold_index] = nn_score

            # Lasso
            lasso_score, _, _, _, _ = lasso(X_train, y_train, X_test, y_test)
            scores_methods["Lasso"][block_index, fold_index] = lasso_score

            # SVC
            svc_score, _, _, _, _ = SVC_(X_train, y_train, X_test, y_test)
            print(svc_score)
            scores_methods["SVC"][block_index, fold_index] = svc_score

    # Compute average scores across folds for each method
    average_scores = {
        name: np.mean(scores, axis=1) for name, scores in scores_methods.items()
    }

    # Visualization
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
    titles = [
        "Heatmap of RF",
        "Heatmap of GBM",
        "Heatmap of NN",
        "Heatmap of Lasso",
        "Heatmap of SVC",
    ]
    for ax, (name, scores) in zip(axes.flat, average_scores.items()):
        heatmap = ax.imshow(scores.reshape(4, 4).T, cmap="viridis", aspect="auto")
        ax.set_title(titles.pop(0))
        fig.colorbar(heatmap, ax=ax)
    # Hide the last subplot if it's not used
    if len(average_scores) < axes.size:
        for i in range(len(average_scores), axes.size):
            axes.flat[i].axis("off")

    plt.tight_layout()
    plt.show()


def load_data_split():
    scores_RF = np.zeros((16))
    scores_GBM = np.zeros((16))
    scores_NN = np.zeros((16))
    scores_lasso = np.zeros((16))
    scores_SVC = np.zeros((16))
    averages = 2

    for av in range(averages):
        print(f"average: [{av+1}/{averages}]")

        data_train, data_labels_train, data_test, data_labels_test, _, _ = load_data()
        # plot_subimages(data_train[0].reshape(64, 64))

        # Parameters
        block_size = 16
        num_blocks = 4

        # Calculate the total number of blocks per image
        total_blocks = num_blocks * num_blocks

        # Prepare to collect blocks, adjusting to store flat blocks
        blocks_train = np.empty(
            (data_train.shape[0], total_blocks, block_size * block_size)
        )
        blocks_test = np.empty(
            (data_test.shape[0], total_blocks, block_size * block_size)
        )

        # Iterate over each image in the training set
        for img_index in range(data_train.shape[0]):
            square_matrix_train = data_train[img_index].reshape(64, 64)
            block_counter = 0
            for i in range(num_blocks):
                for j in range(num_blocks):
                    # Calculate the start and end indices for rows and columns
                    row_start = i * block_size
                    row_end = row_start + block_size
                    col_start = j * block_size
                    col_end = col_start + block_size

                    # Slice the block, flatten it, and add to the array
                    blocks_train[img_index, block_counter] = square_matrix_train[
                        row_start:row_end, col_start:col_end
                    ].flatten()
                    block_counter += 1

        # Iterate over each image in the test set
        for img_index in range(data_test.shape[0]):
            square_matrix_test = data_test[img_index].reshape(64, 64)
            block_counter = 0
            for i in range(num_blocks):
                for j in range(num_blocks):
                    row_start = i * block_size
                    row_end = row_start + block_size
                    col_start = j * block_size
                    col_end = col_start + block_size

                    # Slice the block, flatten it, and add to the array
                    blocks_test[img_index, block_counter] = square_matrix_test[
                        row_start:row_end, col_start:col_end
                    ].flatten()
                    block_counter += 1

        for i in range(16):
            print(f"block [{i+1}/{16}]")
            current_block_train_data = blocks_train[:, i, :]
            current_block_test_data = blocks_test[:, i, :]
            scores_RF[i] += RF(
                current_block_train_data,
                data_labels_train,
                current_block_test_data,
                data_labels_test,
            )[0]
            scores_GBM[i] += GBM(
                current_block_train_data,
                data_labels_train,
                current_block_test_data,
                data_labels_test,
            )[0]
            NN_suppressed = suppress_print(NN)
            scores_NN[i] += NN_suppressed(
                current_block_train_data,
                data_labels_train,
                current_block_test_data,
                data_labels_test,
            )[0]
            scores_lasso[i] += lasso(
                current_block_train_data,
                data_labels_train,
                current_block_test_data,
                data_labels_test,
            )[0]
            scores_SVC[i] += SVC_(
                current_block_train_data,
                data_labels_train,
                current_block_test_data,
                data_labels_test,
            )[0]
            print(
                SVC_(
                    current_block_train_data,
                    data_labels_train,
                    current_block_test_data,
                    data_labels_test,
                )[0]
            )

        scores = [scores_RF, scores_GBM, scores_NN, scores_lasso, scores_SVC]

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(10, 15)
    )  # Adjust the layout and size as needed

    # List of scores and titles

    global_max = max(np.max(score) for score in scores) / averages
    global_min = max(np.min(score) for score in scores) / averages
    scores.append(np.mean(scores))
    titles = [
        "Heatmap of RF",
        "Heatmap of GBM",
        "Heatmap of NN",
        "Heatmap of Lasso",
        "Heatmap of SVC",
        "Heatmap total",
    ]

    # Create each subplot
    for ax, score, title in zip(axes.flat, scores, titles):
        heatmap = ax.imshow(
            score.reshape(4, 4).T / averages,
            cmap="viridis",
            aspect="auto",
            vmax=global_max,
            vmin=global_min,
        )
        ax.set_title(title)
        fig.colorbar(heatmap, ax=ax)  # Add a colorbar to each subplot within its

    # If there's an extra subplot (odd number), hide it
    if len(scores) < axes.size:
        for i in range(len(scores), axes.size):
            axes.flat[i].axis("off")

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()


def plot_subimages(original_image):
    # Parameter
    block_size = 16
    num_blocks = 4

    # Prepare to collect subimages
    subimages = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            row_start = i * block_size
            row_end = row_start + block_size
            col_start = j * block_size
            col_end = col_start + block_size

            subimage = original_image[row_start:row_end, col_start:col_end]
            subimages.append(subimage)

    # Plot subimages in a grid
    plt.figure(figsize=(10, 10))
    for idx, subimage in enumerate(subimages):
        plt.subplot(num_blocks, num_blocks, idx + 1)
        plt.imshow(subimage, cmap="gray")
        plt.title(f"Block {idx+1}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def cluster():
    print("Hej")
    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    data = StandardScaler().fit_transform(data)

    data_transposed = data.T
    print(data_transposed.shape)

    # Choose the number of clusters
    num_clusters = 2

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_transposed)
    clusters = kmeans.predict(data_transposed)

    new_clusters = clusters.reshape(64, 64)

    plt.imshow(new_clusters, cmap="gray")
    plt.show()


def cluster(num_clusters = 2 , transpose = False):
    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    data = StandardScaler().fit_transform(data)

    if transpose:   #This is for the task 2e
        data_ew = data.T
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data_ew)
        clusters = kmeans.predict(data_ew)

        new_clusters = clusters.reshape(64, 64)
        new_clusters = np.rot90(new_clusters, k=-1)
        
        image = data[0].reshape(64, 64)
        image = np.rot90(image, k=-1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap="gray")
        plt.imshow(new_clusters, cmap="plasma", alpha=0.2)
        plt.show()
    
    else:   #This is for task 1c
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        clusters = kmeans.predict(data)
        
        corresponding = []
        for i in range(num_clusters):
            indices_list = np.where(clusters == i)[0]
            label_list = []
            for j in indices_list:
                label_list.append(data_labels[j])
            corresponding.append(np.array(label_list))
        
        i = 0
        plot_matrix = np.zeros((2, num_clusters))
        for list_item in corresponding:
            plot_matrix[0][i] = np.sum(list_item == 0)
            plot_matrix[1][i] = np.sum(list_item == 1)
            i += 1
        
        plt.imshow(plot_matrix, cmap='viridis')
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                plt.text(j, i, str(plot_matrix[i, j]), ha='center', va='center', color='white', fontsize=12)
        plt.show()
        
        correct_guess = np.max(plot_matrix, axis=0)
        print(np.sum(correct_guess) / np.sum(plot_matrix))

def main(args):

    scores = {"RF": [], "GBM": [], "NN": [], "LASSO": [], "SVC": []}
    feature_importances = {"RF": [], "GBM": [], "NN": [], "LASSO": [], "SVC": []}
    misclassified_indices = {"RF": [], "GBM": [], "NN": [], "LASSO": [], "SVC": []}
    test_indices_counter = []

    averages = 10 if (args.plot_performance or args.plot_misclassified_indices) else 1
    for i in range(averages):
        (
            data_train,
            data_labels_train,
            data_test,
            data_labels_test,
            test_indices,
            flip_indices_test,
        ) = load_data(args.flip_data)
        test_indices_counter.extend(test_indices)
        if "RF" in args.classifier:
            RF_score, feature_importance, misclassified_indices_test, _, _ = RF(
                data_train,
                data_labels_train,
                data_test,
                data_labels_test,
                test_indices,
                flip_indices_test,
            )
            scores["RF"].append(RF_score)
            feature_importances["RF"] = feature_importance
            misclassified_indices["RF"].extend(misclassified_indices_test)
            print(f"RF iteration [{i+1}/{averages}] Done")

        if "GBM" in args.classifier:
            GBM_score, feature_importance, misclassified_indices_test, _, _ = GBM(
                data_train,
                data_labels_train,
                data_test,
                data_labels_test,
                test_indices,
            )
            scores["GBM"].append(GBM_score)
            feature_importances["GBM"] = feature_importance
            misclassified_indices["GBM"].extend(misclassified_indices_test)
            print(f"GBM iteration [{i+1}/{averages}] Done")

        if "NN" in args.classifier:
            NN_suppressed = suppress_print(NN)
            NN_score, feature_importance, misclassified_indices_test, _, _ = (
                NN_suppressed(
                    data_train,
                    data_labels_train,
                    data_test,
                    data_labels_test,
                    test_indices,
                )
            )
            scores["NN"].append(NN_score)
            feature_importances["NN"] = feature_importance
            misclassified_indices["NN"].extend(misclassified_indices_test)

            print(f"NN iteration [{i+1}/{averages}] Done")

        if "LASSO" in args.classifier:
            Lasso_score, feature_importance, misclassified_indices_test, _, _ = lasso(
                data_train,
                data_labels_train,
                data_test,
                data_labels_test,
                test_indices,
            )
            scores["LASSO"].append(Lasso_score)
            feature_importances["LASSO"] = feature_importance
            misclassified_indices["LASSO"].extend(misclassified_indices_test)
            print(f"LASSO iteration [{i+1}/{averages}] Done")

        if "SVC" in args.classifier:
            SVC_score, _, misclassified_indices_test, _, _ = SVC_(
                data_train,
                data_labels_train,
                data_test,
                data_labels_test,
                test_indices,
                rbf=True,
            )
            _, feature_importance, _, _, _ = SVC_(
                data_train,
                data_labels_train,
                data_test,
                data_labels_test,
                test_indices,
                rbf=False,
            )
            feature_importances["SVC"] = feature_importance
            scores["SVC"].append(SVC_score)
            misclassified_indices["SVC"].extend(misclassified_indices_test)
            print(f"SVC iteration [{i+1}/{averages}] Done")

    if args.CV == "kernel":
        CV_kernel()
        print("Done")
    if args.CV == "lasso":
        CV_lasso()
        print("Done")
    if args.plot_performance:
        plot_performance(scores)
    if args.plot_features:
        plot_features(feature_importances, args.flip_data)  #
    if args.plot_misclassified_indices:
        plot_misclassified_indices(misclassified_indices, test_indices_counter)
    if args.plot_misclassified_images:
        plot_misclassified_images()  # plots missclassified images
    if args.load_data_split:
        # load_data_split()
        CV_block()  # loads data split
    if args.cluster:
        cluster(args.num_clusters, args.transpose_cluster_data)

    if args.confusion:
        plot_confusion_matrix(
            args.flip_data
        )  # if args.confusion == True, then plots confusion matrix


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        nargs="+",
        type=str,
        help="List of classifiers",
        default=[],
    )
    parser.add_argument("--CV", type=str)
    parser.add_argument("--plot_performance", action="store_true", default=False)
    parser.add_argument("--plot_features", action="store_true", default=False)
    parser.add_argument(
        "--plot_misclassified_indices", action="store_true", default=False
    )
    parser.add_argument(
        "--plot_misclassified_images", action="store_true", default=False
    )
    parser.add_argument("--load_data_split", action="store_true", default=False)
    parser.add_argument("--confusion", action="store_true", default=False)
    parser.add_argument("--flip_data", action="store_true", default=False)
    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--transpose_cluster_data", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
