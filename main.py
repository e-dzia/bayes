import warnings

from discretization import discretize_equal_width, discretize_k_means, discretize_equal_frequency, \
    NO_DISCRETIZATION, EQUAL_FREQUENCY, K_MEANS, EQUAL_WIDTH

from utils import main_single
from tests import main_tests


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    show_mode = True
    filenames = ['iris.csv', 'pima-diabetes.csv', 'glass.csv', 'wine.csv']
    discretization_methods = [NO_DISCRETIZATION, EQUAL_WIDTH, EQUAL_FREQUENCY, K_MEANS]
    bins_sizes = [2, 4, 7, 10, 15]
    splits_sizes = [2, 3, 5, 10]
    stratified = [False, True]

    filename = filenames[3]
    discretization_method = NO_DISCRETIZATION
    num_of_bins = 6
    splits = 10

    main_single('files/' + filename, show_mode, discretization_method=discretization_method, num_of_bins=num_of_bins, splits=splits)
    # main_tests(10, filenames, discretization_methods, bins_sizes, splits_sizes, stratified)

