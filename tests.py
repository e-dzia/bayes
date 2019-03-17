import csv
import datetime
import numpy as np

from discretization import discretize_equal_width, discretize_k_means, discretize_equal_frequency, \
    NO_DISCRETIZATION, EQUAL_FREQUENCY, K_MEANS, EQUAL_WIDTH
from utils import main_single


def main_single_tests(num_of_tests, filename, show_mode, discretization_method=K_MEANS, num_of_bins=4, splits=10, stratified=True):
    accs = []
    prs = []
    recs = []
    f1s = []
    for _ in range(num_of_tests):
        acc, pr, rec, f1 = main_single(filename, show_mode,
                                       discretization_method=discretization_method,
                                       num_of_bins=num_of_bins, splits=splits, stratified=stratified)
        accs.append(acc)
        prs.append(pr)
        recs.append(rec)
        f1s.append(f1)

    acc = np.mean(accs)
    pr = np.mean(prs)
    rec = np.mean(recs)
    f1 = np.mean(f1s)

    return acc, pr, rec, f1


def main_tests_all(filenames, discretization_methods, bins_sizes, splits_sizes, stratified):
    start = datetime.datetime.now()
    print("time start: {}".format(start))

    for filename in filenames:
        file = 'results/res-{}-all-{}.csv'.format(filename.split(".")[0], start).replace(' ', '_').replace(':', '.')
        f = open(file, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(['file', 'stratified', 'splits', 'disc', 'bins', 'acc', 'pr', 'rec', 'f1'])
        for stratify in stratified:
            for splits in splits_sizes:
                for discretization_method in discretization_methods:
                    if discretization_method == NO_DISCRETIZATION:
                        acc, pr, rec, f1 = main_single_tests('files/' + filename, False,
                                                       discretization_method=discretization_method,
                                                       num_of_bins=0, splits=splits, stratified=stratify)
                        data = [filename, 0 if stratify is False else 1, splits, discretization_method, 0, acc, pr, rec, f1]
                        writer.writerow(data)
                        print(data)
                    else:
                        for num_of_bins in bins_sizes:
                            acc, pr, rec, f1 = main_single_tests('files/' + filename, False,
                                                           discretization_method=discretization_method,
                                                           num_of_bins=num_of_bins, splits=splits, stratified=stratify)
                            data = [filename, 0 if stratify is False else 1, splits, discretization_method, num_of_bins, acc, pr, rec, f1]
                            writer.writerow(data)
                            print(data)

        end = datetime.datetime.now()
        print("done, time: {}, elapsed: {}".format(end, end - start))
        f.close()


def main_tests(num_of_tests, filenames, discretization_methods, bins_sizes, splits_sizes, stratified):
    start = datetime.datetime.now()
    print("time start: {}".format(start))

    for filename in filenames:
        file = 'results/res-{}-{}.csv'.format(start, filename.split(".")[0]).replace(' ', '_').replace(':', '.')
        f = open(file, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(['file', 'stratified', 'splits', 'disc', 'bins', 'acc', 'pr', 'rec', 'f1'])

        discretization_method = 1
        num_of_bins = 4

        for stratify in stratified:
            for splits in splits_sizes:
                acc, pr, rec, f1 = main_single_tests(num_of_tests, 'files/' + filename, False,
                                                   discretization_method=discretization_method,
                                                   num_of_bins=num_of_bins, splits=splits, stratified=stratify)
                data = [filename, 0 if stratify is False else 1, splits, discretization_method, num_of_bins, acc, pr, rec, f1]
                writer.writerow(data)
                print(data)

        stratify = True
        splits = 10

        for discretization_method in discretization_methods:
            if discretization_method == NO_DISCRETIZATION:
                acc, pr, rec, f1 = main_single_tests(num_of_tests, 'files/' + filename, False,
                                               discretization_method=discretization_method,
                                               num_of_bins=0, splits=splits, stratified=stratify)
                data = [filename, 0 if stratify is False else 1, splits, discretization_method, 0, acc, pr, rec, f1]
                writer.writerow(data)
                print(data)
            else:
                for num_of_bins in bins_sizes:
                    acc, pr, rec, f1 = main_single_tests(num_of_tests, 'files/' + filename, False,
                                                   discretization_method=discretization_method,
                                                   num_of_bins=num_of_bins, splits=splits, stratified=stratify)
                    data = [filename, 0 if stratify is False else 1, splits, discretization_method, num_of_bins,
                            acc, pr, rec, f1]
                    writer.writerow(data)
                    print(data)

        end = datetime.datetime.now()
        print("done, time: {}, elapsed: {}".format(end, end - start))
        f.close()
