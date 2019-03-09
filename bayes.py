# 0: unpack the data from .csv
# 1: preprocess the data (ex. disceretize, polish (?))
# 2: split the data (cross-validation)
# 3: bayes classifier (on learning dataset)
# 4: evaluate the classifier (on test dataset)
import pandas


def unpack_data(filename):
    data = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        data.columns = ["petalLength", "petalWidth", "sepalLlength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        data.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]

    if filename == 'files/pima-diabetes.csv':
        data.columns = [
            "NumTimesPrg", "PlGlcConc", "BloodP",
            "SkinThick", "TwoHourSerIns", "BMI",
            "DiPedFunc", "Age", "class"]

    if filename == 'files/wine.csv':
        data.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue", "OD280/OD315", "Proline"]

    print(data)
    return data


def preprocess_data(data):
    return data


def split_data(data):

    train = data
    test = data

    return train, test


def cross_validation(data):
    for i in range(10):
        train, test = split_data(data)
        bayes(train)
        evaluate(test)
    pass


def bayes(data):
    pass


def evaluate(data):
    pass


def main(filename):
    data = unpack_data(filename)
    data = preprocess_data(data)
    cross_validation(data)


if __name__ == "__main__":
    filename = 'files/iris.csv'
    main(filename)