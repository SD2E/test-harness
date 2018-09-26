import pandas as pd
import matplotlib.pyplot as plt
import ast
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from random import randint
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():

    # # RFR splits:
    # csvs = ["114k_stabilityscore_RFR_performances.csv",
    #         "114k_stabilityscore_calibrated_RFR_performances.csv",
    #         "114k_stabilityscore_cnn_RFR_performances.csv",
    #         "114k_stabilityscore_cnn_calibrated_RFR_performances.csv"]
    # for csv in csvs:
    #     df = pd.read_csv(csv)
    #     bar_plot_new_splits(df, 'test_split', 'R Squared', csv)

    # CNN splits:
    csvs = ["114k_stabilityscore_CNN_performances.csv",
            "114k_stabilityscore_calibrated_CNN_performances.csv",
            "114k_stabilityscore_cnn_CNN_performances.csv",
            "114k_stabilityscore_cnn_calibrated_CNN_performances.csv"]
    for csv in csvs:
        df = pd.read_csv(csv)
        bar_plot_new_splits(df, 'test_split', 'R Squared', csv)

def bar_plot(df, x, y):
    separated = pd.DataFrame([ast.literal_eval(i) for i in df[x].values])
    df['test_library'] = separated['library']
    df['test_topology'] = separated['topology']
    df['test_info'] = df['test_library'] + ', ' + df['test_topology']
    print(df.shape)

    ax = df.plot.bar(x='test_info', y=y, legend=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Test Split Filter')
    ax.set_ylabel(y)
    plt.title()
    plt.tight_layout()
    plt.show()


def bar_plot_new_splits(df, x, y, title=""):
    separated = pd.DataFrame([ast.literal_eval(i) for i in df[x].values])
    df['test_info'] = df['test_split']
    print(df.shape)

    ax = df.plot.bar(x='test_info', y=y, legend=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Test Split Filter')
    ax.set_ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def percent_overlap_two_lists_same_size(list1, list2):
    list_size = len(list1)
    if list_size != len(list2):
        raise ValueError("Both lists must be of the same size.")
    count = 0
    for x in list1:
        if x in list2:
            count = count + 1
    return float(count) / float(list_size)


def percent_overlap_heatmap(features_df, cols_to_compare, top_n=10):
    heatmap_df = pd.DataFrame(np.ones((len(cols_to_compare), len(cols_to_compare))), index=cols_to_compare,
                              columns=cols_to_compare)
    heatmap_combos = list(itertools.combinations(cols_to_compare, 2))

    for c1, c2 in heatmap_combos:
        c1_importances = list(features_df.sort_values(by=c1, ascending=False)[:top_n]['Feature'])
        c2_importances = list(features_df.sort_values(by=c2, ascending=False)[:top_n]['Feature'])
        percent_overlap = percent_overlap_two_lists_same_size(c1_importances, c2_importances)
        heatmap_df.loc[c1, c2] = percent_overlap
        heatmap_df.loc[c2, c1] = percent_overlap

    sns.heatmap(heatmap_df, annot=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
