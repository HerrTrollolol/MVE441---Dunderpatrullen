import numpy as np
from numpy.linalg import svd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
import os


# cleaned betyder:

# bytt namn till data istället för TCGA_data.
# gjort en funktion av hela augmentation-delen.
# lagt till en funktion som plottar mislabeled images. obeservation att viss data redan är mislabeled.
# flyttat TCGAdata/ labels och CATSnDOGS /labels till en folder som heter data


def RF_break(mode):
    depth_range = range(1, 21, 1)
    tree_range = range(1, 21, 1)
    if mode == "create":
        data = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
        data_labels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

        data = StandardScaler().fit_transform(data)

        data_train, data_test, data_labels_train, data_labels_test = train_test_split(
            data, data_labels, test_size=0.3
        )
        # This classifier is for the Cancer data_set
        result = np.zeros((len(list(depth_range)), len(list(tree_range))))
        i, j = 0, 0
        for depth in depth_range:
            j = 0
            for n_trees in tree_range:
                print(depth, n_trees)
                classifier = RandomForestClassifier(
                    n_estimators=n_trees,
                    criterion="gini",
                    max_depth=depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    bootstrap=True,
                    max_samples=0.7,
                )
                # n_estimators=1000,
                # criterion="gini",
                # max_depth=1000,
                # min_samples_split=2,
                # min_samples_leaf=1,
                # max_features="sqrt",
                # bootstrap=True,
                # max_samples=0.7,
                classifier.fit(data_train, data_labels_train)
                final_test_score = np.sum(
                    classifier.predict(data_test) == data_labels_test
                ) / len(data_test)
                result[i][j] = final_test_score
                j += 1
            i += 1

        file_path = os.path.join("data", "break_method_combination.npy")
        np.save(file_path, result)

    elif mode == "plot":
        result = np.load("data/break_method_combination.npy")
        sns.heatmap(
            result,
            xticklabels=list(tree_range),
            yticklabels=(np.array(list(depth_range))),
        )
        plt.yticks(rotation=0)
        plt.ylabel("Depth")
        plt.xlabel("n_estimators")
        plt.show()


def RF(args):
    if args.data_set != "Cancer":

        print(os.getcwd())
        data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
        data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    else:
        data = np.loadtxt("data/TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
        data_labels = np.loadtxt("data/TCGAlabels", skiprows=1, usecols=1, dtype=str)

    if args.noice != 0:
        data = data + np.random.normal(
            0, args.noice, data.shape
        )  # Adjust noise_std_dev as needed

    data = StandardScaler().fit_transform(data)

    # if args.data_set != "Cancer":
    #     X_svd = svd(data)
    #     data = data @ X_svd[2].T  # Makes data into principal components

    data_train, data_test, data_labels_train, data_labels_test = train_test_split(
        data, data_labels, test_size=0.3
    )

    if args.augmentation:
        data_train, data_labels_train = augment_data(data_train, data_labels_train)

    if args.data_set == "Cancer":
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

    else:  # CatsNDogs
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

    # plots the mislabeld images.
    if args.plot_mislabeled:
        plot_misclassified_images(classifier, data_test, data_labels_test)

    return (final_train_score, final_test_score, classifier.feature_importances_)


def GBM(args):
    if args.data_set != "Cancer":
        data = np.loadtxt("data/CATSnDOGS.csv", delimiter=",", skiprows=1)
        data_labels = np.loadtxt("data/Labels.csv", delimiter=",", skiprows=1)

    else:
        data = np.loadtxt("data/TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
        data_labels = np.loadtxt("data/TCGAlabels", skiprows=1, usecols=1, dtype=str)

    if args.noice != 0:
        data = data + np.random.normal(0, args.noice, data.shape)

    data = StandardScaler().fit_transform(data)

    data_train, data_test, data_labels_train, data_labels_test = train_test_split(
        data, data_labels, test_size=0.25
    )

    if args.augmentation:
        data_train, data_labels_train = augment_data(data_train, data_labels_train)

    if args.data_set == "Cancer":  # This classifier is for the Cancer data_set
        classifier = GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.8,
            n_estimators=1000,
            max_features="sqrt",
            subsample=1.0,
            min_samples_split=2,
            max_depth=3,
        )

    else:  # This classifier is for the cats_and_dogs data_set
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

    if args.plot_mislabeled:
        plot_misclassified_images(classifier, data_test, data_labels_test)

    return (final_train_score, final_test_score, classifier.feature_importances_)


def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(scale=noise_factor, size=data.shape)
    augmented_data = data + noise
    return augmented_data


def augment_data(data_train, data_labels_train):
    augmented_data = data_train
    augmented_data_labels = data_labels_train
    for i in range(1, 11):
        augmented_data = np.concatenate((augmented_data, add_noise(data_train, i / 10)))
        augmented_data_labels = np.concatenate(
            (augmented_data_labels, data_labels_train)
        )

    augmented_data = np.concatenate((augmented_data, flip_data(augmented_data)))
    augmented_data_labels = np.concatenate(
        (augmented_data_labels, augmented_data_labels)
    )

    augmented_data = np.concatenate((augmented_data, add_black_pixels(augmented_data)))
    augmented_data_labels = np.concatenate(
        (augmented_data_labels, augmented_data_labels)
    )

    data_train, data_labels_train = augmented_data, augmented_data_labels

    return augmented_data, augmented_data_labels


def scale_data(data, scale_factor=0.1):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data) * np.random.uniform(
        low=1 - scale_factor, high=1 + scale_factor, size=data.shape
    )
    return scaled_data


def flip_data(data, axis=0):
    flipped_data = np.flip(data, axis=axis)
    return flipped_data


def add_black_pixels(data, percent_pixels=0.05):
    augmented_data = np.copy(data)

    num_pixels = int(np.ceil(data.size * percent_pixels))
    random_indices = np.random.choice(data.size, size=num_pixels, replace=False)
    augmented_data.flat[random_indices] = 0

    return augmented_data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noice", type=int, default=0)
    parser.add_argument("--noice_iteration_max", type=int, default=15)
    parser.add_argument("--noice_iteration_jump", type=int, default=1)
    parser.add_argument("--classifier", type=str, default="RF")
    parser.add_argument("--data_set", type=str, default="Cancer")
    parser.add_argument("--plot_noice", action="store_true", default=False)
    parser.add_argument("--plot_features", action="store_true", default=False)
    parser.add_argument("--augmentation", action="store_true", default=False)
    parser.add_argument("--plot_feature_noice", action="store_true", default=False)
    parser.add_argument("--break_method", action="store_true", default=False)
    parser.add_argument("--plot_break", action="store_true", default=False)
    parser.add_argument("--create_break", action="store_true", default=False)
    parser.add_argument("--plot_mislabeled", action="store_true", default=False)

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

    elif args.plot_feature_noice:
        noice = [0, 4]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
        fig.suptitle("Feature Importance by Model with Increasing Noise")

        for i in noice:
            print(i / len(noice))
            args.noice = i

            _, _, feature_importance_RF = RF(args)
            print("hi")
            _, _, feature_importance_GBM = GBM(args)
            print("då")

            base_color_intensity = 0  # Starting intensity for no noise (0 would be white, 1 would be full color)
            intensity_step = 0.15
            # Calculate current intensity level
            current_intensity = base_color_intensity + i * intensity_step

            # Define colors
            darker_red = (
                1,
                current_intensity,
                current_intensity,
            )  # Keep red constant, decrease green and blue slightly
            darker_blue = (
                current_intensity,
                current_intensity,
                1,
            )  # Keep blue constant, decrease red and green slightly

            # Plot
            axes[0].plot(
                np.sort(feature_importance_GBM)[::-1],
                label=f"GBM Noise={i}",
                color=darker_red,
            )
            axes[1].plot(
                np.sort(feature_importance_RF)[::-1],
                label=f"RF Noise={i}",
                color=darker_blue,
            )

            # Final plot adjustments
        # Adding labels and legend
        axes[0].set_title("GBM")
        axes[0].set_ylabel("Importance")
        axes[0].legend()

        axes[1].set_title("RF")
        axes[1].set_xlabel("Features")
        axes[1].set_ylabel("Importance")
        axes[1].legend()

        # Show plot
        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make room for the main title
        plt.show()
    elif args.break_method:
        if args.create_break:
            RF_break("create")
        if args.plot_break:
            RF_break("plot")

    else:
        if args.classifier == "RF":
            final_train_score, final_test_score, _ = RF(args)
        else:
            final_train_score, final_test_score, _ = GBM(args)
        print(f"Train accuracy: {final_train_score}, Test accuracy: {final_test_score}")
