import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# from numpy.linalg import svd
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def sampling(TCGAdata, TCGAlabels, sampling_type):
    TCGAdata_resampled, TCGAlabels_resampled, sampler = TCGAdata, TCGAlabels, False
    
    if sampling_type == "SMOTE":
        sampler = SMOTE()
    elif sampling_type == "OVER":
        sampler = RandomOverSampler()
    elif sampling_type == "UNDER":
        sampler = RandomUnderSampler()

    if sampler != False:
        TCGAdata_resampled, TCGAlabels_resampled = sampler.fit_resample(
            TCGAdata, TCGAlabels
        )

    return TCGAdata_resampled, TCGAlabels_resampled


def imbalance(TCGAdata, TCGAlabels):
    # orignial data (172,1215,266,571,606,57)
    sampling_split = {
        '"GBM"': 30,
        '"BC"': 1215,
        '"OV"': 30,
        '"LU"': 571,
        '"KI"': 606,
        '"U"': 30,
    }
    
    sampler = RandomUnderSampler(sampling_strategy=sampling_split)

    TCGAdata_resampled, TCGAlabels_resampled = sampler.fit_resample(TCGAdata, TCGAlabels)
    return TCGAdata_resampled, TCGAlabels_resampled


def main(args, ax):
    TCGAdata = np.loadtxt("TCGAdata.txt", skiprows=1, usecols=range(1, 2001))
    TCGAlabels = np.loadtxt("TCGAlabels", skiprows=1, usecols=1, dtype=str)

    # If true, then imbalances the data based on input
    if args.imbalance:
        TCGAdata, TCGAlabels = imbalance(TCGAdata, TCGAlabels)

    TCGAdata, TCGAlabels = sampling(TCGAdata, TCGAlabels, args.sampling_type)

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

    # This sorts TCGAdata_train after each columns variance
    variance = np.var(TCGAdata_train, axis=0)
    sorted_indices = np.argsort(variance)[::-1]
    sorted_vector = TCGAdata_train[:, sorted_indices]
    TCGAdata_test = TCGAdata_test[:, sorted_indices]

    end_accuracy, end_train_accuracy, end_test_accuracy = [], [], []
    x_axis = []
    best_end_accuracy, best_test_prediction = 0, 0

    # Main loop in which we find the most effective usage of principal components
    for i in range(1, args.max_dim + 1, args.dim_step):
        X = sorted_vector[:, :i]
        x_axis.append(i)
        accuracies, train_accuracies = [], []

        k_f, k_n = args.folds, args.neighbours

        if i % (5*args.dim_step) == 1:
            print(i)

        kf = KFold(n_splits=k_f, shuffle=False)  # nice
        knn = KNeighborsClassifier(n_neighbors=k_n)

        # This loop fits a knn-model for i principal components and saves the accuracy
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

        # Here we try to predict the test_data based on i principal components
        knn.fit(X, TCGAlabels_train)
        prediction_test_data = knn.predict(TCGAdata_test[:, :i])
        test_predictions = np.sum(prediction_test_data == TCGAlabels_test)
        dim_test_accuracy = test_predictions / len(TCGAlabels_test)
        end_test_accuracy.append(dim_test_accuracy)

        # And here we save the prediction-accuracy for TEST-data for the number of 
        # principal components that achieved the best prediction on the VALID-data
        if end_accuracy[-1] > best_end_accuracy:
            best_end_accuracy = end_accuracy[-1]
            best_test_prediction = prediction_test_data

    # Everything below here until "return-statement" is just plotting
    print(f"test accuracy: {end_test_accuracy[np.argmax(end_accuracy)]}")
    ax.set_title(
        f"Sampling type: {args.sampling_type} - Test error: {round(1-end_test_accuracy[np.argmax(end_accuracy)],3)}"
    )
    ax.plot(x_axis, [1 - x for x in end_accuracy], label="Valid")
    ax.plot(x_axis, [1 - x for x in end_train_accuracy], label="Train")
    # ax.plot(x_axis, [1 - x for x in end_test_accuracy], label="Test")
    # ax.plot(
    #     x_axis,
    #     [1 - end_test_accuracy[np.argmax(end_accuracy)]] * len(x_axis),
    #     "-.",
    #     linewidth=0.5,
    #     color="black",
    #     label="Best model",
    # )
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
    print(end_test_accuracy)
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
            main(args, ax)
            ax.set_ylim(y_limits)
        plt.tight_layout(pad=3.0)
        plt.show()

    elif args.plot_all_2:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs_flat = axs.flatten()
        y_limits = (10**-3, 1)

        i = 0  
        for elem1 in ns:
            for elem2 in ss:
                ax = axs_flat[i]
                args.neighbours = elem1
                args.train_share = elem2
                main(args, ax) 
                ax.set_ylim(y_limits)
                ax.set_title(
                    f"neighbours: {elem1}, train share: {elem2}"
                )  
                i += 1  

        plt.tight_layout(pad=6.0)
        plt.show()
        
    elif args.plot_variance:
        colors = ["blue", "green", "red"]
        share_values = [0.5, 0.7, 0.9] 
        alpha_values = [0.2, 0.4, 0.6]  
        all_case_results = []

        for index, share_value in enumerate(share_values):
            args.train_share = share_value  
            test_result = []

            for i in range(10):  
                fig, ax = plt.subplots(figsize=(10, 7))
                result = main(args, ax)[30:50] # Works as intended if max_dim/dim_step > 50. 
                test_result.append(result)
                plt.close(fig)

            all_results_array = np.array(test_result)
            means = np.mean(all_results_array, axis=0)
            std_devs = np.std(all_results_array, axis=0)
            all_case_results.append((means, std_devs))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) 

        for ax, (means, std_devs), color, alpha, share_value in zip(
            axes, all_case_results, colors, alpha_values, share_values
        ):
            iterations = np.arange(len(means)) * 5
            ax.plot(
                iterations,
                means,
                label=f"Mean Train Share {share_value}",
                color=color,
            )
            ax.fill_between(
                iterations,
                means - std_devs,
                means + std_devs,
                color=color,
                alpha=alpha,
                label=f"Mean ± STD Train Share {share_value}",
            )
            ax.set_xlabel("Number of features")
            ax.set_ylabel("Test Error")
            ax.set_title(f"Train Share {share_value}")
            ax.legend()

        plt.suptitle("Mean and Standard Deviation of Test error")
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        ) 
        plt.show()
        
    else:
        y_limits = (10**-4, 1)
        fig, ax = plt.subplots(figsize=(10, 7))
        main(args, ax)