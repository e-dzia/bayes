import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
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
    return dataset


def split_data(dataset):
    # TODO -> it's not cross-validation yet, tmp: test_size=0.2
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['class'])
    train_set_labels = train_set["class"].copy()
    train_set = train_set.drop("class", axis=1)

    test_set_labels = test_set["class"].copy()
    test_set = test_set.drop("class", axis=1)

    return train_set, train_set_labels, test_set, test_set_labels


def cross_validation(dataset):
    # for i in range(1):
    #     train_set, train_set_labels, test_set, test_set_labels = split_data(dataset)
    #     bayes(train_set)
    #     evaluate(test_set)
    train_set, train_set_labels, test_set, test_set_labels = split_data(dataset)

    clf = naive_bayes.MultinomialNB()  # Bayes()
    clf.fit(train_set, train_set_labels)
    labels = clf.predict(test_set)

    count = 0
    for label_bayes, label_true in zip(labels, test_set_labels):
        print(label_bayes, label_true)
        if label_bayes == label_true:
            count += 1
    print(len(labels), count)


def bayes(dataset):
    pass


def evaluate(dataset):
    pass


def show_data(dataset):
    print(dataset.head(), "\n")
    print(dataset.corr())

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
    cross_validation(dataset)


if __name__ == "__main__":
    show_mode = True
    filename = 'files/iris.csv'

    main(filename, show_mode)
