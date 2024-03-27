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


    # Shuffle the combined data
    suffle_indecies = np.arange(0, 2887, 1)
    np.random.shuffle(suffle_indecies)

    # Split the shuffled data and labels into training and test sets
    split_indexies = int(len(suffle_indecies) * args.train_share)

    TCGAdata_train = TCGAdata[suffle_indecies[:split_indexies]]
    TCGAdata_test = TCGAdata[suffle_indecies[split_indexies:]]

    TCGAlabels_train = TCGAlabels[suffle_indecies[:split_indexies]]
    TCGAlabels_test = TCGAlabels[suffle_indecies[split_indexies:]]
    
    #This part sorts TCGAdata_train after each columns variance, from biggest to smallest
    variance = np.var(TCGAdata_train, axis=0)
    sorted_indices = np.argsort(variance)[::-1]
    sorted_vector = TCGAdata_train[:, sorted_indices]
    TCGAdata_test = TCGAdata_test[:, sorted_indices]
    
    end_accuracy = []
    end_train_accuracy = []
    end_test_accuracy = []
    x_axis = []
    
    for i in range(1, args.max_dim, args.dim_step):
        X = sorted_vector[:, :i]  
        x_axis.append(i)
        accuracies = []
        train_accuracies = []

        k_f = args.folds
        k_n = args.neighbours
        
        if i%10 == 1:
            print(i)

        kf = KFold(n_splits=k_f, shuffle=False) #nice
        knn = KNeighborsClassifier(n_neighbors=k_n)

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = TCGAlabels_train[train_index], TCGAlabels_train[valid_index]

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_valid)
            y_train_pred = knn.predict(X_train)
            accuracies.append((np.sum(y_pred == y_valid))) 
            train_accuracies.append((np.sum(y_train_pred == y_train)))
            
        end_train_accuracy.append(np.sum(train_accuracies)/(len(TCGAlabels_train)*(k_f-1)))
        end_accuracy.append(np.sum(accuracies)/(len(TCGAlabels_train)))
        
        knn.fit(X, TCGAlabels_train)
        test_predictions = np.sum(knn.predict(TCGAdata_test[:, :i]) == TCGAlabels_test)
        end_test_accuracy.append(test_predictions/len(TCGAlabels_test))
        
        
    plt.title(f"k_f = {k_f}, k_n = {k_n}, max accuracy = {round(np.max(end_test_accuracy), 3)} at i = {x_axis[np.argmax(end_test_accuracy)]}")
    plt.plot(x_axis, end_accuracy, label="valid")
    plt.plot(x_axis, end_train_accuracy, label="train")
    plt.plot(x_axis, end_test_accuracy, label="test")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default = 10)
    parser.add_argument('--neighbours', type=int, default=10)
    parser.add_argument('--max_dim', type=int, default=2000)
    parser.add_argument('--dim_step', type=int, default=1)
    parser.add_argument('--train_share', type=float, default=0.8)
    args = parser.parse_args()
    main(args)