import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(12345)


def get_data():
    permute = rng.permutation(200)

    data = np.vstack([
        rng.normal((1.0, 2.0, -3.0), 1.0, size=(50, 3)),
        rng.normal((-1.0, 1.0, 1.0), 1.0, size=(50, 3)),
        rng.normal((0.0, -1.0, -1.0), 1.0, size=(50, 3)),
        rng.normal((-1.0, -1.0, -2.0), 1.0, size=(50, 3))
    ])
    labels = np.hstack([
        [1]*50, [2]*50, [3]*50, [4]*50
    ])

    X = pd.DataFrame(np.take(data, permute, axis=0), columns=["A", "B", "C"])
    y = pd.Series(np.take(labels, permute, axis=0))
    return X, y


data, labels = get_data()

data.to_csv("data.csv")
labels.to_csv("labels.csv")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=23456)

print(X_train.index.size, X_test.index.size)

X_train.index.to_series().to_csv("train_index.csv", index=False, header=False)
X_test.index.to_series().to_csv("test_index.csv", index=False, header=False)

classifier  = DecisionTreeClassifier(random_state=34567)
classifier.fit(X_train, y_train)

feature_importance = pd.DataFrame(classifier.feature_importances_, index=classifier.feature_names_in_, columns=["Importance"])
feature_importance.to_csv("feature_importance.csv")

train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

pd.Series(train_predictions, index=X_train.index, name="Predicted labels").to_csv("train_predictions.csv")
pd.Series(test_predictions, index=X_test.index, name="Predicted labels").to_csv("test_predictions.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
ax1.set_title("Confusion matrix for training data")
ax2.set_title("Confusion matrix for test data")
ConfusionMatrixDisplay.from_predictions(y_train, train_predictions, ax=ax1, cmap="Greys", colorbar=False)
ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, ax=ax2, cmap="Greys", colorbar=False)

print(f"Train accuracy {accuracy_score(y_train, train_predictions)}",
      f"Test accuracy {accuracy_score(y_test, test_predictions)}", sep="\n")

plt.show()