import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import scipy.stats as stats
import seaborn as sns
import tensorflow as tf
from numpy.linalg import svd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras import layers, models


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

    return (final_train_score, final_test_score, classifier.feature_importances_)

def kernel_regression(data_train, data_labels_train, data_test, data_labels_test):
    classifier = KernelRidge(alpha=1.0, kernel= "rbf", gamma=0.1)
    classifier.fit(data_train,data_labels_train)
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_test
    )
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    return final_train_score, final_test_score

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

    return (final_train_score, final_test_score, classifier.feature_importances_)

def lasso(data_train, data_labels_train, data_test, data_labels_test):
    classifier = LogisticRegression(penalty="l1", solver = "liblinear",C = 0.1,
    )
    classifier.fit(data_train,data_labels_train)
    
    final_train_score = np.sum(
        classifier.predict(data_train) == data_labels_train
    ) / len(data_train)
    final_test_score = np.sum(classifier.predict(data_test) == data_labels_test) / len(
        data_train
    )
    return final_train_score,final_test_score


def CV_kernel():
    data_train, data_labels_train, _, _ = load_data()
    alphas = [10E-4,10E-3,10E-2,10E-1,10E-0]
    gammas = [10E-4,10E-3,10E-2,10E-1,10E-0]
    scores = np.zeros((len(alphas), len(gammas)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i,j = 0
    for alpha in alphas:
        for gamma in gammas:
            accuracies = []
            classifier = KernelRidge(alpha=alpha, kernel= "rbf", gamma=gamma)
            for train_index, test_index in kf.split(data_train):
                # Split data
                X_train, X_test = data_train[train_index], data_train[test_index]
                y_train, y_test = data_labels_train[train_index], data_labels_train[test_index]
                
                # Train the model
                classifier.fit(X_train, y_train)
                
                # Predict on the test set
                predictions = classifier.predict(X_test)
                
                # Calculate accuracy
                accuracies = accuracy_score(y_test, predictions)
                accuracies.append(accuracies)
            score = np.mean(accuracies)
            print(f"For {alpha} and {gamma} we get the score of {score}.")
            scores[i][j] = score
            j+=1
        i += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores, annot=True, fmt=".2f", xticklabels=alphas, yticklabels=gammas, cmap='coolwarm')
    plt.title('Heatmap of Scores')
    plt.xlabel('Alphas')
    plt.ylabel('Gammas')
    plt.show()
    
    return scores

def CV_lasso():
    data_train, data_labels_train, _, _ = load_data()
    alphas = [10E-4,10E-3,10E-2,10E-1,10E-0]
    scores = np.zeros(len(alphas))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i = 0
    for alpha in alphas:
        accuracies = []
        classifier = classifier = LogisticRegression(penalty="l1", solver = "liblinear",C = alpha)
        for train_index, test_index in kf.split(data_train):
            # Split data
            X_train, X_test = data_train[train_index], data_train[test_index]
            y_train, y_test = data_labels_train[train_index], data_labels_train[test_index]
            
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
        i+=1
    plt.figure(figsize=(10, 8))
    plt.plot(alphas, scores)
    plt.title('Heatmap of Scores')
    plt.xlabel('Alphas')
    plt.ylabel('Gammas')
    plt.show()
    
    return scores

def NN(data_train, data_labels_train, data_test, data_labels_test):
    model = models.Sequential([
        layers.Flatten(input_shape=(64**2,)), 
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'), 
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # model.fit(data_train, data_labels_train)
    # score = model.predict(data_test, data_labels_test)

    # model.summary()
    return("hi")
            
def plot_confusion_matrix(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted)

    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14}, 
                xticklabels=['Class 0 - Catty', 'Class 1 - Doggie'], 
                yticklabels=['Class 0 - Catty', 'Class 1 - Doggie'])
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()     

def play_heureka_sound():
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("Sounds\\Heureka.mp3")  
    pygame.mixer.music.play()

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

def main(args):
    
    data_train, data_labels_train, data_test, data_labels_test = load_data()
    scores = {"RF":[], "GBM":[], "NN":[], "LAS":[], "KR":[]}
    
    if "RF" in args.classifier:
        RF_score = RF(data_train, data_labels_train, data_test, data_labels_test)
        scores["RF"] = RF_score
        print("Done")
        
    if "GBM" in args.classifier:
        GBM_score = GBM(data_train, data_labels_train, data_test, data_labels_test)
        scores["GBM"] = GBM_score
        print("Done")
        
    if "NN" in args.classifier:
        NN_score = NN(data_train, data_labels_train, data_test, data_labels_test)
        scores["NN"] = NN_score
        print("Done")
    
    if "LAS" in args.classifier:
        Lasso_score = lasso(data_train, data_labels_train, data_test, data_labels_test)
        scores["LAS"] = Lasso_score
        print("Done")
    
    if "KR" in args.classifier:
        KR_score = kernel_regression(data_train, data_labels_train, data_test, data_labels_test)
        scores["KR"] = KR_score
        print("Done")
        
    if args.CV == "kernel":
        CV_kernel()
        print("Done")
    if args.CV == "lasso":
        CV_lasso()
        print("Done")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", nargs='+', type=str, help='List of classifiers', default=["RF", "GBM", "NN", "LAS", "KR"])
    parser.add_argument("--CV", type=str)
    
    args = parser.parse_args()
    main(args)