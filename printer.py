import matplotlib.pyplot as plt
import seaborn as sns


def show_data(dataset_name, dataset):
    fig, ax = plt.subplots()
    dataset.hist(bins=50, figsize=(10, 15), ax=ax)
    fig.savefig('visualization/' + dataset_name + '-hist.pdf')
    plt.clf()

    plt.figure(figsize=(12, 12))
    corr = dataset.corr()
    sns_plot = sns.heatmap(corr, cmap="PiYG", annot=True, center=0, linewidths=.5, fmt=".2f")
    sns_plot.figure.savefig('visualization/' + dataset_name + '-corr.pdf')
    plt.clf()

    plt.figure(figsize=(12, 12))
    cov = dataset.cov()
    sns_plot = sns.heatmap(cov, cmap="PiYG", annot=True, center=0, linewidths=.5, fmt=".2f")
    sns_plot.figure.savefig('visualization/' + dataset_name + '-cov.pdf')
    plt.clf()

    sns_plot = sns.pairplot(dataset, kind="scatter", diag_kind="hist", hue="class")
    sns_plot.savefig('visualization/' + dataset_name + '-scatter.pdf')
    plt.clf()


def show_data_example(show_mode, filename, dataset):
    if show_mode:  # do this only once for every dataset - no necessity to do it more
         res = filename.split('/')
         dataset_name = res[1].split('.')[0]
         show_data(dataset_name, dataset)
