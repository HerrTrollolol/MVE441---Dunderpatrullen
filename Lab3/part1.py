import argparse
import os
import sys
import io

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


def load_data():

    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    data = StandardScaler().fit_transform(data)

    data_train, data_test, data_labels_train, data_labels_test = train_test_split(
        data, data_labels, test_size=0.2
    )

    return data_train, data_labels_train, data_test, data_labels_test


def RF(data_train, data_labels_train, data_test, data_labels_test):

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
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_test
    )
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)

    return final_test_score, np.array(classifier.feature_importances_)


def SVC_(data_train, data_labels_train, data_test, data_labels_test, rbf=True):
    if rbf:
        classifier = SVC(C=10.0, kernel="rbf", gamma=0.0005)
    else:
        classifier = SVC(C=10.0, kernel="linear")
    classifier.fit(data_train, data_labels_train)
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_test
    )
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    if not rbf:
        params = classifier.coef_[0]
    else:
        params = []

    return final_test_score, np.array(params)


def GBM(data_train, data_labels_train, data_test, data_labels_test):

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
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_test
    )
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)

    return final_test_score, np.array(classifier.feature_importances_)


def lasso(data_train, data_labels_train, data_test, data_labels_test):
    classifier = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=20,
    )
    classifier.fit(data_train, data_labels_train)

    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_test
    )

    params = classifier.coef_[0]

    return final_test_score, np.array(params)


def NN(data_train, data_labels_train, data_test, data_labels_test):
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
            self.fc1 = nn.Linear(64 * 64, 256)
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
    model.load_state_dict(torch.load(save_path))
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

    return accuracy.item(), np.array(avg_saliency)


def CV_kernel():
    data_train, data_labels_train, _, _ = load_data()
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
    data_train, data_labels_train, _, _ = load_data()
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


def plot_confusion_matrix(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted)

    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 14},
        xticklabels=["Class 0 - Catty", "Class 1 - Doggie"],
        yticklabels=["Class 0 - Catty", "Class 1 - Doggie"],
    )

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def plot_misclassified_images(classifier, data_test, data_labels_test):
    predictions_test = classifier.predict(data_test)
    misclassified_indices = np.where(predictions_test != data_labels_test)[0]

    # Plot misclassified images
    for idx in misclassified_indices:
        predicted_label = predictions_test[idx]
        actual_label = data_labels_test[idx]

        image_shape = (64, 64)
        misclassified_image = data_test[idx].reshape(image_shape)

        if predicted_label == 0:
            predicted_label = "Cat"
        else:
            predicted_label = "Dog"
        if actual_label == 0:
            actual_label = "Cat"
        else:
            actual_label = "Dog"
        # Show the misclassified image
        plt.imshow(misclassified_image, cmap="gray")
        plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
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


def plot_features(model_feature_importances):
    _, _, data_test, _ = load_data()
    # Append test images to the dictionary with specific keys
    model_feature_importances["Test image 1"] = data_test[0]
    model_feature_importances["Test image 2"] = data_test[1]
    model_feature_importances["Test image 3"] = data_test[2]

    num_models = len(model_feature_importances)

    # Setup the matplotlib figure and axes, 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust figsize as needed

    # Flatten the axes array for easier iteration if there are more than one row and column
    axes = axes.flatten()

    # Store a reference image for the colorbar, preferably not a test image
    reference_image = None

    # Iterate over the images and their axes to plot each image
    for ax, (model_name, image_array) in zip(
        axes, reversed(list(model_feature_importances.items()))
    ):
        # Ensure the image data is in NumPy array format
        if isinstance(image_array, list):
            image_array = np.array(image_array)

        # Reshape the flat array to 64x64 and transpose it
        image_matrix = image_array.reshape(64, 64).T

        # Display the image
        cmap = "gray" if "Test image" in model_name else "hot"
        im = ax.imshow(image_matrix, cmap=cmap, aspect="auto")
        ax.set_title(model_name)
        ax.axis("off")  # Turn off axis ticks and labels

        # Save the first non-test image reference for the colorbar
        if reference_image is None and "Test image" not in model_name:
            reference_image = im

    # Add a colorbar using the reference image
    if reference_image is not None:
        cax = fig.add_axes([0.67, 0.01, 0.01, 0.305])  # Position for the colorbar
        fig.colorbar(reference_image, cax=cax)

    # Hide any unused axes
    for i in range(num_models, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main(args):

    scores = {"RF": [], "GBM": [], "NN": [], "LASSO": [], "SVC": []}
    averages = 100 if args.plot_performance else 1
    feature_importances = {"RF": [], "GBM": [], "NN": [], "LASSO": [], "SVC": []}
    for i in range(averages):
        data_train, data_labels_train, data_test, data_labels_test = load_data()
        if "RF" in args.classifier:
            RF_score, feature_importance = RF(
                data_train, data_labels_train, data_test, data_labels_test
            )
            scores["RF"].append(RF_score)
            feature_importances["RF"] = feature_importance
            print(f"RF iteration [{i+1}/{averages}] Done")

        if "GBM" in args.classifier:
            GBM_score, feature_importance = GBM(
                data_train, data_labels_train, data_test, data_labels_test
            )
            scores["GBM"].append(GBM_score)
            feature_importances["GBM"] = feature_importance
            print(f"GBM iteration [{i+1}/{averages}] Done")

        if "NN" in args.classifier:
            NN_suppressed = suppress_print(NN)
            NN_score, feature_importance = NN_suppressed(
                data_train, data_labels_train, data_test, data_labels_test
            )
            scores["NN"].append(NN_score)
            feature_importances["NN"] = feature_importance
            print(f"NN iteration [{i+1}/{averages}] Done")

        if "LASSO" in args.classifier:
            Lasso_score, feature_importance = lasso(
                data_train, data_labels_train, data_test, data_labels_test
            )
            scores["LASSO"].append(Lasso_score)
            feature_importances["LASSO"] = feature_importance
            print(f"LASSO iteration [{i+1}/{averages}] Done")

        if "SVC" in args.classifier:
            SVC_score, _ = SVC_(
                data_train, data_labels_train, data_test, data_labels_test
            )
            _, feature_importance = SVC_(
                data_train, data_labels_train, data_test, data_labels_test, False
            )
            feature_importances["SVC"] = feature_importance
            scores["SVC"].append(SVC_score)
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
        plot_features(feature_importances)


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

    args = parser.parse_args()
    main(args)
