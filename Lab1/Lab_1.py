import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold

def main():
    TCGAdata = np.loadtxt('TCGAdata.txt', skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt('TCGAlabels', skiprows=1, usecols=1, dtype=str)

    X_pp = StandardScaler().fit_transform(TCGAdata)
    X_svd = svd(X_pp)
    X_dat = X_pp @ X_svd[2].T
    
    end_accuracy = []
    x_axis = []
    
    for i in range(1, 50, 1):
        X = X_dat[:, :i]  
        x_axis.append(i)
        accuracies = []

        k_f = 2
        k_n = 100
        
        if i%10 == 1:
            print(i)

        kf = KFold(n_splits=k_f, shuffle=True, random_state=69) #nice
        knn = KNeighborsClassifier(n_neighbors=k_n)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = TCGAlabels[train_index], TCGAlabels[test_index]

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracies.append((np.sum(y_pred == y_test))) 
            
        total_accuracy = np.sum(accuracies)/len(TCGAlabels)
        end_accuracy.append(total_accuracy)
    plt.plot(x_axis, end_accuracy)
    plt.show()

if __name__ == "__main__":
    main()