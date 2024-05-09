import argparse
import os

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


def load_data():

    data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
    data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    data = StandardScaler().fit_transform(data)

    indices = np.arange(len(data))
    data_train, data_test, data_labels_train, data_labels_test, indices_train, indices_test = train_test_split(
        data, data_labels, indices, test_size=0.2
    )

    return data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test


def RF(data_train, data_labels_train, data_test, data_labels_test,  indices_train, indices_test):

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
    predictions_train = classifier.predict(data_train)
    predictions_test = classifier.predict(data_test)
    
    misclassified_indices_train = [index for index, (pred_label, true_label) in zip(indices_train, zip(predictions_train, data_labels_train)) if pred_label != true_label]
    
    misclassified_indices_test = [index for index, (pred_label, true_label) in zip(indices_test, zip(predictions_test, data_labels_test)) if pred_label != true_label]

    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(predictions_train == data_labels_train) / len(data_train)

    return (final_test_score, final_train_score, classifier.feature_importances_, misclassified_indices_test, misclassified_indices_train)


def kernel_regression(data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test):
    classifier = SVC(C=10.0, kernel="rbf", gamma=0.0005)
    classifier.fit(data_train, data_labels_train)
    
    predictions_train = classifier.predict(data_train)
    predictions_test = classifier.predict(data_test)
    
    misclassified_indices_train = [index for index, (pred_label, true_label) in zip(indices_train, zip(predictions_train, data_labels_train)) if pred_label != true_label]
    
    misclassified_indices_test = [index for index, (pred_label, true_label) in zip(indices_test, zip(predictions_test, data_labels_test)) if pred_label != true_label]

    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(predictions_train == data_labels_train) / len(data_train)

    return final_test_score, final_train_score, misclassified_indices_test, misclassified_indices_train


def GBM(data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test):
    classifier = GradientBoostingClassifier(
        loss="log_loss", # same as deviant. avoiding error message.
        learning_rate=0.8,
        n_estimators=100,
        max_features="sqrt",
        subsample=1.0,
        min_samples_split=2,
        max_depth=4,
    )

    classifier.fit(data_train, data_labels_train)
    predictions_train = classifier.predict(data_train)
    predictions_test = classifier.predict(data_test)
    
    misclassified_indices_train = [index for index, (pred_label, true_label) in zip(indices_train, zip(predictions_train, data_labels_train)) if pred_label != true_label]
    misclassified_indices_test = [index for index, (pred_label, true_label) in zip(indices_test, zip(predictions_test, data_labels_test)) if pred_label != true_label]

    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)
    final_train_score = np.sum(predictions_train == data_labels_train) / len(data_train)

    return final_test_score, final_train_score, misclassified_indices_test, misclassified_indices_train


def lasso(data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test):
    classifier = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=20,
    )
    classifier.fit(data_train, data_labels_train)

    predictions_train = classifier.predict(data_train)
    predictions_test = classifier.predict(data_test)

    misclassified_indices_train = [index for index, (pred_label, true_label) in zip(indices_train, zip(predictions_train, data_labels_train)) if pred_label != true_label]
    
    misclassified_indices_test = [index for index, (pred_label, true_label) in zip(indices_test, zip(predictions_test, data_labels_test)) if pred_label != true_label]

    final_train_score = np.sum(predictions_train == data_labels_train) / len(data_train)
    final_test_score = np.sum(predictions_test == data_labels_test) / len(data_test)

    return final_test_score, final_train_score, misclassified_indices_test, misclassified_indices_train


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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

    # Lists to store misclassified indices
    misclassified_indices_train = []
    misclassified_indices_test = []

    save_path = "best_model.pth"

    num_items = len(data_train)  # total number of items in the dataset
    num_train = int(num_items * 0.9)  # 90% for training
    num_val = num_items - num_train  # 10% for validation

    full_dataset = TensorDataset(data_train, data_labels_train)
    train_data, val_data = random_split(full_dataset, [num_train, num_val])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Training the model with early stopping
    best_val_loss = float("inf")
    patience = 3  # Patience is the number of epochs to tolerate no improvement
    trigger_times = 0  # How many times the trigger condition has been met

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

    # Load the best model for evaluation
    model = NeuralNetwork()
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        # Evaluation on training set
        train_outputs = model(data_train)
        train_predicted = (train_outputs >= 0.5).float()
        misclassified_indices_train = [i for i, (pred, true) in enumerate(zip(train_predicted, data_labels_train)) if pred != true]

        # Evaluation on test set
        test_outputs = model(data_test)
        test_predicted = (test_outputs >= 0.5).float()
        misclassified_indices_test = [i for i, (pred, true) in enumerate(zip(test_predicted, data_labels_test)) if pred != true]

        train_accuracy = (train_predicted == data_labels_train.unsqueeze(1)).float().mean()
        test_accuracy = (test_predicted == data_labels_test.unsqueeze(1)).float().mean()

        print(f"Train Accuracy: {train_accuracy.item()}, Test Accuracy: {test_accuracy.item()}")

    return test_accuracy.item(), misclassified_indices_test, misclassified_indices_train



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


def plot_misclassified_indices(misclassified_indices):
    plt.figure(figsize=(15, 6))  # Increase figure width to accommodate more image indices
    misclassified_counts = {}
    classifiers = list(misclassified_indices.keys())
    colors = plt.cm.get_cmap('tab10', len(classifiers))  # Get a colormap with distinct colors for each classifier
    
    for classifier in classifiers:
        counts = {}
        for index in misclassified_indices[classifier]:
            counts[index] = counts.get(index, 0) + 1
        
        x = list(counts.keys())
        y = list(counts.values())
        
        plt.bar(x, y, color=colors(classifiers.index(classifier)), label=classifier)

    plt.xlabel('Image Index')
    plt.ylabel('Misclassified Count')
    plt.title('Number of Times Each Image is Misclassified by Different Classifiers')
    plt.legend(title='Classifier')
    plt.grid(axis='y')  # Add gridlines along y-axis for better readability
    plt.show()


def plot_performance(scores):
    print(scores)

    # Creating the plot
    # Get the keys and values from the dictionary
    labels = list(scores.keys())
    values = list(scores.values())
    print(values)

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



def main(args):

    scores = {"RF": [], "GBM": [], "NN": [], "LAS": [], "KR": []}
    averages = 30 if args.plot_performance else 110
    misclassified_indices = {classifier: [] for classifier in args.classifier}
    
    for i in range(averages):
        data_train, data_labels_train, data_test, data_labels_test,  indices_train, indices_test = load_data()
       

        for classifier in args.classifier:

            if "RF" in args.classifier:
                RF_score, _, _,misclassified_indices_test, misclassified_indices_train = RF(
                    data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test
                )
                scores["RF"].append(RF_score)
                misclassified_indices[classifier].extend(misclassified_indices_test)
                misclassified_indices[classifier].extend(misclassified_indices_train)
                print(f"RF iteration [{i+1}/{averages}] Done")

            if "GBM" in args.classifier:
                GBM_score, _, misclassified_indices_test, misclassified_indices_train = GBM(
                    data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test 
                )
                misclassified_indices[classifier].extend(misclassified_indices_test)
                misclassified_indices[classifier].extend(misclassified_indices_train)
                scores["GBM"].append(GBM_score)
                print(f"GBM iteration [{i+1}/{averages}] Done")

            if "NN" in args.classifier:
                NN_score = NN(data_train, data_labels_train, data_test, data_labels_test)
                scores["NN"].append(NN_score)
                print(f"NN iteration [{i+1}/{averages}] Done")

            if "LAS" in args.classifier:
                Lasso_score, _, misclassified_indices_test, misclassified_indices_train = lasso(
                    data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test
                )
                scores["LAS"].append(Lasso_score)
                misclassified_indices[classifier].extend(misclassified_indices_test)
                misclassified_indices[classifier].extend(misclassified_indices_train)
                print(f"LAS iteration [{i+1}/{averages}] Done")

            if classifier == "KR":
                KR_score, _, misclassified_indices_test, misclassified_indices_train  = kernel_regression(
                    data_train, data_labels_train, data_test, data_labels_test, indices_train, indices_test
                )
                scores["KR"].append(KR_score)
                misclassified_indices[classifier].extend(misclassified_indices_test)
                misclassified_indices[classifier].extend(misclassified_indices_train)
                print(f"KR iteration [{i+1}/{averages}] Done")

    if args.CV == "kernel":
        CV_kernel()
        print("Done")
    if args.CV == "lasso":
        CV_lasso()
        print("Done")
    if args.plot_performance:
        plot_performance(scores)
    if args.plot_misclassified_indices:
        plot_misclassified_indices(misclassified_indices)


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
    parser.add_argument("--plot_misclassified_indices", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
