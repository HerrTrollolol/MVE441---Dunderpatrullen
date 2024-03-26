import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold
import argparse


def main(args):
    TCGAdata = np.loadtxt('TCGAdata.txt', skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt('TCGAlabels', skiprows=1, usecols=1, dtype=str)

    X_pp = StandardScaler().fit_transform(TCGAdata)
    X_svd = svd(X_pp)
    X_dat = X_pp @ X_svd[2].T  #Principel component
    
    end_accuracy = []
    end_train_accuracy = []
    x_axis = []
    
    for i in range(1, args.max_dim, args.dim_step):
        X = X_dat[:, :i]  
        x_axis.append(i)
        accuracies = []
        train_accuracies = []

        k_f = args.folds
        k_n = args.neighbours
        
        if i%10 == 1:
            print(i)

        kf = KFold(n_splits=k_f, shuffle=True, random_state=69) #nice
        knn = KNeighborsClassifier(n_neighbors=k_n)

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = TCGAlabels[train_index], TCGAlabels[valid_index]

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_valid)
            y_train_pred = knn.predict(X_train)
            accuracies.append((np.sum(y_pred == y_valid))) 
            train_accuracies.append((np.sum(y_train_pred == y_train)))
            
        end_train_accuracy.append(np.sum(train_accuracies)/len(TCGAlabels))
        end_accuracy.append(np.sum(accuracies)/len(TCGAlabels))
    plt.title(f"k_f = {k_f}, k_n = {k_n}, max accuracy = {round(np.max(end_accuracy), 3)} at i = {x_axis[np.argmax(end_accuracy)]}")
    plt.plot(x_axis, end_accuracy, label="valid")
    plt.plot(x_axis, end_train_accuracy, label="train")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default = 10)
    parser.add_argument('--neighbours', type=int, default=10)
    parser.add_argument('--max_dim', type=int, default=2000)
    parser.add_argument('--dim_step', type=int, default=1)
    args = parser.parse_args()
    main(args)