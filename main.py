import csv

import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, model_selection, metrics, preprocessing
from gaussianbayes import GaussianBayes
from multinomialbayes import MultinomialBayes, EQUAL_BINS, K_MEANS, EQUAL_SIZE_BINS


def unpack_data(filename):
    dataset = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        dataset.columns = ["petalLength", "petalWidth", "sepalLength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        dataset.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
        dataset = dataset.set_index('id')

    if filename == 'files/pima-diabetes.csv':
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc",
                           "Age",
                           "class"]

    if filename == 'files/wine.csv':
        dataset.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                           "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue",
                           "OD280/OD315", "Proline"]

    return dataset


def preprocess_data(dataset):
    # TODO?
    # if filename == 'files/iris.csv':
    #     type = {'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}
    #     dataset['class'] = [type[item] for item in dataset['class']]

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


def extract_labels(dataset):
    # extract labels
    dataset_labels = dataset["class"].copy()
    dataset = dataset.drop("class", axis=1)

    return dataset, dataset_labels


def cross_validation(dataset, labels, model, splits):
    seed = 7
    X = dataset
    Y = labels

    # kfold = model_selection.KFold(n_splits=10, random_state=seed)
    kfold = model_selection.StratifiedKFold(n_splits=splits, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_weighted')
    # TODO: accuracy? f1_weighted?
    return cv_results


def evaluate(labels_true, labels_predicted):
    accuracy = metrics.accuracy_score(y_true=labels_true, y_pred=labels_predicted)
    f1 = metrics.f1_score(y_true=labels_true, y_pred=labels_predicted, average='weighted')

    count = 0
    for label_bayes, label_true in zip(labels_predicted, labels_true):
        if show_mode:
            print(label_true, label_bayes)
        if label_bayes == label_true:
            count += 1
    print("Correct labels: {}/{}".format(count, len(labels_predicted)))
    print("Accuracy: {}, F1: {}".format(accuracy, f1))


def show_data(dataset):
    print(dataset.head(), "\n")
    print(dataset.corr(), "\n")

    # dataset.hist(bins=50, figsize=(20, 15))
    # plt.show()


def main_single(filename, show_mode, discretization_method=K_MEANS, num_of_bins=4, splits=10):
    # unpack the data from .csv
    dataset = unpack_data(filename)

    if show_mode:  # optional
        show_data(dataset)

    # preprocess the data (ex. disceretize, polish (?))
    dataset = preprocess_data(dataset)

    # my own Bayes algorithm model
    # model = GaussianBayes()
    # model = MultinomialBayes(discretization_method=EQUAL_BINS, num_of_bins=num_of_bins)  # naive_bayes.GaussianNB()
    model = MultinomialBayes(discretization_method=discretization_method, num_of_bins=num_of_bins)

    # split the data
    # TODO: split the dataset before cross-validation?
    train_set, train_set_labels = extract_labels(dataset)

    cv_results = cross_validation(train_set, train_set_labels, model, splits)

    if show_mode:
        print("Cross-validation F1 score: {} ({})".format(cv_results.mean(), cv_results.std()))

    return cv_results

    # train_set, train_set_labels, test_set, test_set_labels = split_data(dataset)
    #
    # # bayes classifier (on learning dataset)
    # model.fit(train_set, train_set_labels)
    # labels = model.predict(test_set)
    #
    # # evaluate the classifier (on test dataset)
    # evaluate(test_set_labels, labels)


def main_tests(filenames, discretization_methods, bins_sizes, splits_sizes):
    for filename in filenames:
        f = open("res-" + filename, "w+")
        writer = csv.writer(f)
        for splits in splits_sizes:
            for discretization_method in discretization_methods:
                for num_of_bins in bins_sizes:
                    results = main_single('files/' + filename, False, discretization_method=discretization_method,
                                num_of_bins=num_of_bins, splits=splits)
                    data = [filename, splits, discretization_method, num_of_bins, results.mean()]
                    writer.writerow(data)
                    print(data)
        f.close()


if __name__ == "__main__":
    show_mode = True
    filenames = ['iris.csv', 'pima-diabetes.csv', 'glass.csv', 'wine.csv']
    discretization_methods = [EQUAL_BINS, K_MEANS, EQUAL_SIZE_BINS]
    bins_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    splits_sizes = [2, 3, 4, 5, 7, 9, 10, 12, 15]

    filename = filenames[1]
    discretization_method = K_MEANS
    num_of_bins = 6
    splits = 10

    main_single('files/' + filename, show_mode, discretization_method=discretization_method, num_of_bins=num_of_bins, splits=splits)
    # main_tests(filenames, discretization_methods, bins_sizes, splits_sizes)

# TODO: w glass jest SPORO podmianek (co z tym zrobić?) w Gaussian
# TODO: czy metoda gaussian to osobna metoda dyskretyzacji?
