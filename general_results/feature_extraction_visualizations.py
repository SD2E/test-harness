import pandas as pd
import matplotlib.pyplot as plt
import os
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
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = [f for f in files if '.csv' in f]
    performance_files = [f for f in files if 'performances' in f]
    feature_files = [f for f in files if 'features' in f]

    pimportance_df = None
    for f in feature_files:
        dataset = f.split("_")[1].split("-")[0]
        model_used = f.split("_")[1].split("-")[1]
        df = pd.read_csv(f, comment="#")
        df.rename(columns={"Importance": "{}_Dataset_{}".format(model_used, dataset)}, inplace=True)
        if pimportance_df is None:
            pimportance_df = df.copy()
        else:
            pimportance_df = pd.merge(pimportance_df, df, on="Feature")
    print(pimportance_df)

    pimportance_cols = list(pimportance_df.columns.values)
    pimportance_cols.remove("Feature")

    percent_overlap_heatmap(pimportance_df, pimportance_cols, 10)


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

    sns.clustermap(heatmap_df, annot=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
