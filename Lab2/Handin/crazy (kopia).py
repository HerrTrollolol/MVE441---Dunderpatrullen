import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

random_array = np.random.randint(0, 10, size=(100000, 9))
labels = []
for i in random_array:
    _, counts = np.unique(i, return_counts=True)
    if np.any(counts > 2):
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
print(f"Num of triplets {np.sum(labels) / len(labels)}")

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
accurcies = []
for i in range(100):
    data_train, data_test, labels_train, labels_test = train_test_split(
        random_array, labels, test_size=0.5
    )
    classifier.fit(data_train, labels_train)
    final_test_score = np.sum(classifier.predict(data_test) == labels_test) / len(
        labels_test
    )

    print(f"Accuracy: {final_test_score}")
    accurcies.append(final_test_score)

plt.boxplot(accurcies)

# Adding titles and labels
plt.title("boxplot of Accuracies")
plt.ylabel("Accuracy")

# Display the plot
plt.show()
