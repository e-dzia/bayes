import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, model_selection, metrics
from bayes import Bayes


def unpack_data(filename):
    dataset = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        dataset.columns = ["petalLength", "petalWidth", "sepalLength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        dataset.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
        dataset = dataset.set_index('id')

    if filename == 'files/pima-diabetes.csv':
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age",
                           "class"]

    if filename == 'files/wine.csv':
        dataset.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                           "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue",
                           "OD280/OD315", "Proline"]

    return dataset


def preprocess_data(dataset):
    # TODO: digitize etc
    return dataset


def split_data(dataset):
    # split into train and test sets
    # TODO: stratify?
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['class'])

    # extract labels
    train_set_labels = train_set["class"].copy()
    train_set = train_set.drop("class", axis=1)

    test_set_labels = test_set["class"].copy()
    test_set = test_set.drop("class", axis=1)

    return train_set, train_set_labels, test_set, test_set_labels


def cross_validation(dataset, labels, model):
    seed = 7
    X = dataset
    Y = labels

    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_weighted')
    # TODO: accuracy? f1_weighted?

    print("Cross-validation mean score: {} ({})".format(cv_results.mean(), cv_results.std()))


def evaluate(labels_true, labels_predicted):
    accuracy = metrics.accuracy_score(y_true=labels_true, y_pred=labels_predicted)
    f1 = metrics.f1_score(y_true=labels_true, y_pred=labels_predicted, average='weighted')

    count = 0
    for label_bayes, label_true in zip(labels_predicted, labels_true):
        # print(label_bayes, label_true)
        if label_bayes == label_true:
            count += 1
    print("Correct labels: {}/{}".format(count, len(labels_predicted)))
    print("Accuracy: {}, F1: {}".format(accuracy, f1))


def show_data(dataset):
    print(dataset.head(), "\n")
    print(dataset.corr(), "\n")

    # dataset.hist(bins=50, figsize=(20, 15))
    # plt.show()


def main(filename, show_mode):
    # 0: unpack the data from .csv
    # 1: preprocess the data (ex. disceretize, polish (?))
    # 2: split the data (cross-validation)
    # 3: bayes classifier (on learning dataset)
    # 4: evaluate the classifier (on test dataset)
    dataset = unpack_data(filename)

    if show_mode:  # optional
        show_data(dataset)

    dataset = preprocess_data(dataset)

    # TODO: my own Bayes algorithm model
    model = naive_bayes.GaussianNB()

    # TODO: split the dataset before cross-validation?
    train_set, train_set_labels, test_set, test_set_labels = split_data(dataset)

    cross_validation(train_set, train_set_labels, model)

    model.fit(train_set, train_set_labels)
    labels = model.predict(test_set)

    evaluate(test_set_labels, labels)


if __name__ == "__main__":
    show_mode = True
    filename = 'files/iris.csv'

    main(filename, show_mode)
