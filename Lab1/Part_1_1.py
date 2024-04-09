import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold
import argparse
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def sampling(TCGAdata, TCGAlabels, sampling_type):
    TCGAdata_resampled, TCGAlabels_resampled = TCGAdata, TCGAlabels
    if sampling_type == "SMOTE":
        smote = SMOTE()

        TCGAdata_resampled, TCGAlabels_resampled = smote.fit_resample(
            TCGAdata, TCGAlabels
        )
    elif sampling_type == "OVER":
        ros = RandomOverSampler()

        TCGAdata_resampled, TCGAlabels_resampled = ros.fit_resample(
            TCGAdata, TCGAlabels
        )

    elif sampling_type == "UNDER":
        rus = RandomUnderSampler()

        TCGAdata_resampled, TCGAlabels_resampled = rus.fit_resample(
            TCGAdata, TCGAlabels
        )

    return TCGAdata_resampled, TCGAlabels_resampled


def main(args):
    TCGAdata = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

    TCGAdata, TCGAlabels = sampling(TCGAdata, TCGAlabels, args.sampling_type)

    # Normerar data
    TCGAdata = StandardScaler().fit_transform(TCGAdata)

    X_svd = svd(TCGAdata)
    TCGAdata = TCGAdata @ X_svd[2].T  # Principel component

    # Shuffle the combined data
    print(f"number of data points: {TCGAlabels.shape[0]}")
    suffle_indecies = np.arange(0, TCGAlabels.shape[0] - 1, 1)
    np.random.shuffle(suffle_indecies)

    # Split the shuffled data and labels into training and test sets
    split_indexies = int(len(suffle_indecies) * args.train_share)

    TCGAdata_train = TCGAdata[suffle_indecies[:split_indexies]]
    TCGAdata_test = TCGAdata[suffle_indecies[split_indexies:]]

    TCGAlabels_train = TCGAlabels[suffle_indecies[:split_indexies]]
    TCGAlabels_test = TCGAlabels[suffle_indecies[split_indexies:]]

    end_accuracy = []
    end_train_accuracy = []
    end_test_accuracy = []
    x_axis = []

    for i in range(1, args.max_dim, args.dim_step):
        X = TCGAdata_train[:, :i]
        x_axis.append(i)
        accuracies = []
        train_accuracies = []

        k_f = args.folds
        k_n = args.neighbours

        if i % 10 == 1:
            print(i)

        kf = KFold(n_splits=k_f, shuffle=True, random_state=69)  # nice
        knn = KNeighborsClassifier(n_neighbors=k_n)

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = (
                TCGAlabels_train[train_index],
                TCGAlabels_train[valid_index],
            )

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_valid)
            y_train_pred = knn.predict(X_train)
            accuracies.append((np.sum(y_pred == y_valid)))
            train_accuracies.append((np.sum(y_train_pred == y_train)))

        end_train_accuracy.append(
            np.sum(train_accuracies) / (len(TCGAlabels_train) * (k_f - 1))
        )
        end_accuracy.append(np.sum(accuracies) / (len(TCGAlabels_train)))

        knn.fit(X, TCGAlabels_train)
        test_predictions = np.sum(knn.predict(TCGAdata_test[:, :i]) == TCGAlabels_test)
        end_test_accuracy.append(test_predictions / len(TCGAlabels_test))
    print(f"test accuracy: {end_test_accuracy[np.argmax(end_accuracy)]}")
    plt.title(
        f"k_f = {k_f}, k_n = {k_n}, max accuracy (val) = {round(np.max(end_accuracy), 3)} at i = {x_axis[np.argmax(end_accuracy)]}"
    )
    plt.plot(x_axis, end_accuracy, label="Valid")
    plt.plot(x_axis, end_train_accuracy, label="Train")
    plt.plot(x_axis, end_test_accuracy, label="Test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--max_dim", type=int, default=2000)
    parser.add_argument("--dim_step", type=int, default=1)
    parser.add_argument("--train_share", type=float, default=0.8)
    parser.add_argument("--sampling_type", type=str, default="None")
    args = parser.parse_args()
    main(args)


# Category "GBM": 172 occurrences, 5.96% of total data
# Category "BC": 1215 occurrences, 42.09% of total data
# Category "OV": 266 occurrences, 9.21% of total data
# Category "LU": 571 occurrences, 19.78% of total data
# Category "KI": 606 occurrences, 20.99% of total data
# Category "U": 57 occurrences, 1.97% of total data
