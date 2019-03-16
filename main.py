import csv
import warnings
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, model_selection, metrics, preprocessing

from gaussianbayes import GaussianBayes
from multinomialbayes import MultinomialBayes
from discretization import discretize_equal_width, discretize_k_means, discretize_equal_frequency, \
    NO_DISCRETIZATION, EQUAL_FREQUENCY, K_MEANS, EQUAL_WIDTH
from printer import show_data


def unpack_data(filename):
    dataset = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        dataset.columns = ["petalLength", "petalWidth", "sepalLength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        dataset.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
        dataset = dataset.set_index('id')

    if filename == 'files/pima-diabetes.csv':
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc",
                           "Age", "class"]

    if filename == 'files/wine.csv':
        dataset.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                           "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue",
                           "OD280/OD315", "Proline"]

    return dataset


def preprocess_data(X, discretization_method, num_of_bins):
    # discretize X
    if discretization_method == EQUAL_WIDTH:
        X = discretize_equal_width(X, num_of_bins)

    if discretization_method == K_MEANS:
        X = discretize_k_means(X, num_of_bins)

    if discretization_method == EQUAL_FREQUENCY:
        X = discretize_equal_frequency(X, num_of_bins)

    return X


def split_data(dataset):
    # split into train and test sets
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


def cross_validation(X, y, kfold, model):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for train, test in kfold.split(X, y):
        train = train.tolist()
        test = test.tolist()
        params = model.get_params()
        t = type(model)
        model = t(*params.values())
        model.fit(X.iloc[train], y.iloc[train])
        labels_predicted = model.predict(X.iloc[test])
        labels_true = y.iloc[test]
        accuracy, precision, recall, f1 = evaluate(labels_true, labels_predicted)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    return accuracy, precision, recall, f1


def evaluate(labels_true, labels_predicted):
    accuracy = metrics.accuracy_score(y_true=labels_true, y_pred=labels_predicted)
    precision = metrics.precision_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    recall = metrics.recall_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    f1 = metrics.f1_score(y_true=labels_true, y_pred=labels_predicted, average='macro')

    return accuracy, precision, recall, f1


def main_single(filename, show_mode, discretization_method=K_MEANS, num_of_bins=4, splits=10, stratified=True):
    # unpack the data from .csv
    dataset = unpack_data(filename)

    # choose the model
    if discretization_method == NO_DISCRETIZATION:
        model = GaussianBayes()
    else:
        model = MultinomialBayes()

    # split the data
    dataset, dataset_labels = extract_labels(dataset)
    dataset = preprocess_data(dataset, discretization_method, num_of_bins)

    if stratified:
        kfold = model_selection.StratifiedKFold(n_splits=splits, random_state=7)
    else:
        kfold = model_selection.KFold(n_splits=10, random_state=7)

    accuracy, precision, recall, f1 = cross_validation(dataset, dataset_labels, kfold, model)

    if show_mode:
        print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(accuracy, precision, recall, f1))

    return accuracy, precision, recall, f1


def main_tests(filenames, discretization_methods, bins_sizes, splits_sizes):
    for filename in filenames:
        f = open("results/res-" + filename, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(['file', 'splits', 'disc', 'bins', 'acc', 'pr', 'rec', 'f1'])
        for splits in splits_sizes:
            for discretization_method in discretization_methods:
                if discretization_method == NO_DISCRETIZATION:
                    acc, pr, rec, f1 = main_single('files/' + filename, False,
                                                   discretization_method=discretization_method,
                                                   num_of_bins=0, splits=splits)
                    data = [filename, splits, discretization_method, 0, acc, pr, rec, f1]
                    writer.writerow(data)
                    print(data)
                else:
                    for num_of_bins in bins_sizes:
                        acc, pr, rec, f1 = main_single('files/' + filename, False,
                                                       discretization_method=discretization_method,
                                                       num_of_bins=num_of_bins, splits=splits)
                        data = [filename, splits, discretization_method, num_of_bins, acc, pr, rec, f1]
                        writer.writerow(data)
                        print(data)
        f.close()


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    show_mode = True
    filenames = ['iris.csv', 'pima-diabetes.csv', 'glass.csv', 'wine.csv']
    discretization_methods = [NO_DISCRETIZATION, EQUAL_WIDTH, EQUAL_FREQUENCY, K_MEANS]
    bins_sizes = [2, 4, 7, 10, 15]
    splits_sizes = [2, 3, 5, 10]
    stratified = [True, False]

    filename = filenames[0]
    discretization_method = EQUAL_WIDTH
    num_of_bins = 6
    splits = 10

    # main_single('files/' + filename, show_mode, discretization_method=discretization_method, num_of_bins=num_of_bins, splits=splits)
    main_tests([filenames[0]], discretization_methods, bins_sizes, splits_sizes)

