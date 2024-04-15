import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold
import argparse
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


def imbalance(TCGAdata, TCGAlabels):
    # orignial data (172,1215,266,571,606,57)
    sampling_strategy = {
        '"GBM"': 50,
        '"BC"': 1200,
        '"OV"': 50,
        '"LU"': 50,
        '"KI"': 50,
        '"U"': 50,
    }
    # Initialize the RandomUnderSampler with the defined strategy
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)

    # Resample the dataset
    TCGAdata_resampled, TCGAlabels_resampled = rus.fit_resample(TCGAdata, TCGAlabels)
    return TCGAdata_resampled, TCGAlabels_resampled


def main(args, ax):
    TCGAdata = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

    if args.imbalance:
        TCGAdata, TCGAlabels = imbalance(TCGAdata, TCGAlabels)

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
    best_end_accuracy, best_test_prediction = 0, 0

    for i in range(1, args.max_dim + 1, args.dim_step):
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
        prediction_test_data = knn.predict(TCGAdata_test[:, :i])
        test_predictions = np.sum(prediction_test_data == TCGAlabels_test)
        dim_test_accuracy = test_predictions / len(TCGAlabels_test)
        end_test_accuracy.append(dim_test_accuracy)

        if end_accuracy[-1] > best_end_accuracy:
            best_end_accuracy = end_accuracy[-1]
            best_test_prediction = prediction_test_data

    print(f"test accuracy: {end_test_accuracy[np.argmax(end_accuracy)]}")
    ax.set_title(
        f"Sampling type: {args.sampling_type} - Test error: {round(1-end_test_accuracy[np.argmax(end_accuracy)],3)}"
    )
    ax.plot(x_axis, [1 - x for x in end_accuracy], label="Valid")
    ax.plot(x_axis, [1 - x for x in end_train_accuracy], label="Train")
    ax.plot(x_axis, [1 - x for x in end_test_accuracy], label="Test")

    ax.plot(
        x_axis,
        [1 - end_test_accuracy[np.argmax(end_accuracy)]] * len(x_axis),
        "-.",
        linewidth=0.5,
        color="black",
        label="Best model",
    )

    ax.set_ylabel("Error")
    ax.set_xlabel("Dimension")
    ax.set_yscale("log")
    ax.legend()

    if not (args.plot_all or args.plot_all_2 or args.plot_variance):
        cm = confusion_matrix(TCGAlabels_test, best_test_prediction, normalize="pred")
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            cmap=plt.cm.Blues,
            xticklabels=np.unique(TCGAlabels_test),
            yticklabels=np.unique(TCGAlabels_test),
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.show()
    return [1 - x for x in end_test_accuracy]


if __name__ == "__main__":
    alt = ["NONE", "SMOTE", "OVER", "UNDER"]
    ns = [3, 10, 100]
    ss = [0.5, 0.7, 0.9]

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--max_dim", type=int, default=2000)
    parser.add_argument("--dim_step", type=int, default=1)
    parser.add_argument("--train_share", type=float, default=0.8)
    parser.add_argument("--sampling_type", type=str, default="None")
    parser.add_argument("--plot_all", action="store_true", default=False)
    parser.add_argument("--plot_all_2", action="store_true", default=False)
    parser.add_argument("--plot_variance", action="store_true", default=False)
    parser.add_argument("--imbalance", action="store_true", default=False)
    args = parser.parse_args()
    if args.plot_all:
        y_limits = (10**-3, 1)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs_flat = axs.flatten()

        for i, ax in enumerate(axs_flat):
            args.sampling_type = alt[i]
            # Setting the same y-axis limits
            main(args, ax)
            ax.set_ylim(y_limits)
        plt.tight_layout(pad=3.0)
        plt.show()

    elif args.plot_all_2:
        # Create a 3x3 grid for plotting, adjust figsize as needed
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs_flat = axs.flatten()

        # Set y-axis limits
        y_limits = (10**-3, 1)

        # Iterate through each combination of elements from list1 and list2
        i = 0  # Index to access flattened axes
        for elem1 in ns:
            for elem2 in ss:
                ax = axs_flat[i]
                # Assuming you modify args or call main() based on elem1 and elem2
                # Example modification, replace with actual usage:
                args.neighbours = elem1
                args.train_share = elem2
                main(args, ax)  # Call your main plotting function
                ax.set_ylim(y_limits)
                ax.set_title(
                    f"neighbours: {elem1}, train share: {elem2}"
                )  # Optional: set title for subplot
                i += 1  # Move to the next subplot

        # Ensure there's no overlap in the layout
        plt.tight_layout(pad=6.0)

        # Display the plot
        plt.show()
    elif args.plot_variance:
        colors = ["blue", "green", "red"]
        share_values = [
            0.5,
            0.7,
            0.9,
        ]  # Different share values for training or other uses
        alpha_values = [0.2, 0.4, 0.6]  # Different alpha values for visibility

        all_case_results = []

        # Prepare results for each share value
        for index, share_value in enumerate(share_values):
            args.train_share = share_value  # Modify args accordingly if necessary
            test_result = []

            for i in range(10):  # Run each case 5 times
                fig, ax = plt.subplots(figsize=(10, 7))
                result = main(args, ax)[
                    10:20
                ]  # Assuming main() returns a list, and you're interested in elements 10 to 20
                test_result.append(result)
                plt.close(fig)

            all_results_array = np.array(test_result)
            means = np.mean(all_results_array, axis=0)
            std_devs = np.std(all_results_array, axis=0)
            all_case_results.append((means, std_devs))

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 1 row, 3 columns

        for ax, (means, std_devs), color, alpha, share_value in zip(
            axes, all_case_results, colors, alpha_values, share_values
        ):
            iterations = range(len(means))
            ax.plot(
                iterations, means, label=f"Mean Train Share {share_value}", color=color
            )
            ax.fill_between(
                iterations,
                means - std_devs,
                means + std_devs,
                color=color,
                alpha=alpha,
                label=f"Mean ± STD Train Share {share_value}",
            )
            ax.set_xlabel("PCA dimensions")
            ax.set_ylabel("Test Error")
            ax.set_title(f"Train Share {share_value}")
            ax.legend()

        plt.suptitle("Mean and Standard Deviation of Test error")
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust the layout to make room for the global title
        plt.show()
    else:
        y_limits = (10**-4, 1)
        fig, ax = plt.subplots(figsize=(10, 7))
        main(args, ax)
        ax.set_ylim(y_limits)
        plt.tight_layout(pad=3.0)
        plt.title("VAD I HELVETE")
        plt.show()
