import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ortho_group
from numpy.linalg import qr
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def main():
    TCGAdata = np.loadtxt('TCGAdata.txt', skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt('TCGAlabels', skiprows=1, usecols=1, dtype=str)
    
    #print(TCGAdata[0][1999])
    #print(TCGAdata.shape)

    # standardize (i.e., scale to unit variance) and center the data (i.e., subtract the mean)
    X_pp = StandardScaler().fit_transform(TCGAdata)
    X_svd = svd(X_pp)
    X_dat = X_pp @ X_svd[2].T
    
    end_accuracy = []
    x_axis = []
    
    for i in range(1, 50, 1):
        
        # Define your dataset and labels
        X = X_dat[:, :i]  # Your dataset
        x_axis.append(i)

        # Define the number of folds
        k_f = 2
        k_n = 100
        # Initialize a KFold object
        kf = KFold(n_splits=k_f, shuffle=True, random_state=69) #nice
        
        if i%10 == 1:
            print(i)

        # Initialize a list to store the accuracies
        accuracies = []
        knn = KNeighborsClassifier(n_neighbors=k_n)
        
        # Loop through each fold
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = TCGAlabels[train_index], TCGAlabels[test_index]

            # Create KNN classifier with k=1 (for finding the closest neighbor)
            knn.fit(X_train, y_train)

            # Predict the class label for the query point
            y_pred = knn.predict(X_test)

            accuracies.append((np.sum(y_pred == y_test)))  # Output: [1
        total_accuracy = np.sum(accuracies)/len(TCGAlabels)
        
        end_accuracy.append(total_accuracy)
    print(end_accuracy)
    plt.plot(x_axis, end_accuracy)
    plt.show()

if __name__ == "__main__":
    main()