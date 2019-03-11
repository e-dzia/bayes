import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, model_selection, metrics, preprocessing
from gaussianbayes import GaussianBayes
from multinomialbayes import MultinomialBayes, EQUAL_BINS, K_MEANS


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


def cross_validation(dataset, labels, model):
    seed = 7
    X = dataset
    Y = labels

    # kfold = model_selection.KFold(n_splits=10, random_state=seed)
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='f1_weighted')
    # TODO: accuracy? f1_weighted?

    print("Cross-validation F1 score: {} ({})".format(cv_results.mean(), cv_results.std()))


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


def main(filename, show_mode):
    # unpack the data from .csv
    dataset = unpack_data(filename)

    if show_mode:  # optional
        show_data(dataset)

    # preprocess the data (ex. disceretize, polish (?))
    dataset = preprocess_data(dataset)

    # my own Bayes algorithm model
    # model = GaussianBayes()
    # model = MultinomialBayes(discretization_method=EQUAL_BINS)  # naive_bayes.GaussianNB()
    model = MultinomialBayes(discretization_method=K_MEANS)

    # split the data
    # TODO: split the dataset before cross-validation?
    train_set, train_set_labels = extract_labels(dataset)

    cross_validation(train_set, train_set_labels, model)

    train_set, train_set_labels, test_set, test_set_labels = split_data(dataset)

    # bayes classifier (on learning dataset)
    model.fit(train_set, train_set_labels)
    labels = model.predict(test_set)

    # evaluate the classifier (on test dataset)
    evaluate(test_set_labels, labels)


if __name__ == "__main__":
    show_mode = False
    filename = 'files/iris.csv'

    main(filename, show_mode)

# TODO: w glass jest SPORO podmianek (co z tym zrobić?) w Gaussian
# TODO: czy metoda gaussian to osobna metoda dyskretyzacji?
