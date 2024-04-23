import numpy as np
from numpy.linalg import svd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def RF(args):
    if args.data_set != "Cancer":
        TCGAdata = pd.read_csv("CATSnDOGS.csv", header=True, dtype=int).values
        TCGAlabels = pd.read_csv("Labels.csv", header=True, dtype=int).values

    else:
        TCGAdata = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
        TCGAlabels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

    if args.noice != 0:
        TCGAdata = TCGAdata + np.random.normal(
            0, args.noice, TCGAdata.shape
        )  # Adjust noise_std_dev as needed

    TCGAdata = StandardScaler().fit_transform(TCGAdata)

    TCGA_train, TCGA_test, TCGAlabels_train, TCGAlabels_test = train_test_split(
        TCGAdata, TCGAlabels, test_size=0.25
    )

    classifier = RandomForestClassifier(
        n_estimators=1000,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.7,
    )

    classifier.fit(TCGA_train, TCGAlabels_train)
    final_test_score = np.sum(classifier.predict(TCGA_test) == TCGAlabels_test) / len(
        TCGA_test
    )
    final_train_score = np.sum(
        classifier.predict(TCGA_train) == TCGAlabels_train
    ) / len(TCGA_train)

    return (final_train_score, final_test_score, classifier.feature_importances_)


def GBM(args):
    if args.data_set != "Cancer":
        TCGAdata = pd.read_csv("CATSnDOGS.csv", header=True, dtype=int).values
        TCGAlabels = pd.read_csv("Labels.csv", header=True, dtype=int).values

    else:
        TCGAdata = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
        TCGAlabels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

    if args.noice != 0:
        TCGAdata = TCGAdata + np.random.normal(0, args.noice, TCGAdata.shape)

    TCGAdata = StandardScaler().fit_transform(TCGAdata)

    TCGA_train, TCGA_test, TCGAlabels_train, TCGAlabels_test = train_test_split(
        TCGAdata, TCGAlabels, test_size=0.25
    )

    classifier = GradientBoostingClassifier(
        loss="log_loss",  # Specifies the loss function to be used as logarithmic loss
        learning_rate=0.8,  # Controls the contribution of each tree in the ensemble
        n_estimators=1000,  # Number of boosting stages or trees to be built
        max_features="sqrt",
        subsample=1.0,  # Fraction of samples used for fitting the individual base learners
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        max_depth=3,  # Maximum depth of the individual decision trees
    )

    classifier.fit(TCGA_train, TCGAlabels_train)
    final_test_score = np.sum(classifier.predict(TCGA_test) == TCGAlabels_test) / len(
        TCGA_test
    )
    final_train_score = np.sum(
        classifier.predict(TCGA_train) == TCGAlabels_train
    ) / len(TCGA_train)
    return (final_train_score, final_test_score, classifier.feature_importances_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noice", type=int, default=0)
    parser.add_argument("--noice_iteration_max", type=int, default=15)
    parser.add_argument("--noice_iteration_jump", type=int, default=1)
    parser.add_argument("--classifier", type=str, default="RF")
    parser.add_argument("--plot_noice", action="store_true", default=False)
    parser.add_argument("--plot_features", action="store_true", default=False)
    parser.add_argument("--data_set", type=str, default="Cancer")
    args = parser.parse_args()

    if args.plot_noice:
        x_axis = []
        all_test_RF = []
        all_train_RF = []
        all_test_GBM = []
        all_train_GBM = []
        for i in range(0, args.noice_iteration_max, args.noice_iteration_jump):
            print(f"Iteration: {i} of {args.noice_iteration_max}")
            args.noice = i  # Setting the current noise level

            # RF model scores
            final_train_score_RF, final_test_score_RF, _ = RF(args)
            all_train_RF.append(final_train_score_RF)
            all_test_RF.append(final_test_score_RF)

            # GBM model scores
            final_train_score_GBM, final_test_score_GBM, _ = GBM(args)
            all_train_GBM.append(final_train_score_GBM)
            all_test_GBM.append(final_test_score_GBM)

            x_axis.append(i)
        # Plotting
        plt.figure(figsize=(12, 6))  # Set the overall figure size

        # Subplot for RF
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.plot(x_axis, all_test_RF, label="Test Score RF", color="red")
        plt.plot(x_axis, all_train_RF, label="Train Score RF", color="blue")
        plt.title("Random Forest Scores")
        plt.xlabel("Noise STD")
        plt.ylabel("Accuraccy")
        plt.legend()

        # Subplot for GBM
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.plot(x_axis, all_test_GBM, label="Test Score GBM", color="green")
        plt.plot(x_axis, all_train_GBM, label="Train Score GBM", color="purple")
        plt.title("Gradient Boosting Machine Scores")
        plt.xlabel("Noise STD")
        plt.ylabel("Accuraccy")
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    elif args.plot_features:
        # Assume feature_importance_RF and feature_importance_GBM are populated correctly
        _, _, feature_importance_RF = RF(args)
        _, _, feature_importance_GBM = GBM(args)

        # Calculate the total importance for each feature
        total_importance = feature_importance_RF + feature_importance_GBM

        # Filter out features where the total importance is zero
        non_zero_indices = np.where(total_importance != 0)[0]

        # Apply the filter to keep only non-zero importance features
        filtered_feature_importance_RF = feature_importance_RF[non_zero_indices]
        filtered_feature_importance_GBM = feature_importance_GBM[non_zero_indices]

        # Get the sorted indices based on non-zero total importance
        sorted_indices = np.argsort(
            filtered_feature_importance_RF + filtered_feature_importance_GBM
        )

        # Sort the importance arrays using the sorted indices
        sorted_feature_importance_RF = filtered_feature_importance_RF[sorted_indices]
        sorted_feature_importance_GBM = filtered_feature_importance_GBM[sorted_indices]

        # Number of features after filtering and sorting
        num_features = np.arange(len(sorted_feature_importance_RF))

        # Set the width of each bar
        bar_width = 0.5

        # Plotting
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
        plt.bar(
            num_features,
            sorted_feature_importance_RF,
            width=bar_width,
            label="RF",
            color="b",
        )
        plt.bar(
            num_features,
            sorted_feature_importance_GBM,
            width=bar_width,
            bottom=sorted_feature_importance_RF,
            label="GBM",
            color="r",
        )

        # Adding labels and title
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.title("Sorted and Filtered Feature Importance Summed with Contributions")

        # Adding legend
        plt.legend()

        # Show plot
        plt.show()
        plt.plot(
            np.sort(feature_importance_GBM)[::-1],
            label="GBM",
            color="r",
        )
        plt.plot(
            np.sort(feature_importance_RF)[::-1],
            label="RF",
            color="b",
        )
        plt.show()

    else:
        if args.classifier == "RF":
            final_train_score, final_test_score, _ = RF(args)
        else:
            final_train_score, final_test_score, _ = GBM(args)
        print(f"Train accuracy: {final_train_score}, Test accuracy: {final_test_score}")
