import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from test_harness.utils.names import Names

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def vis_barplot(data_df, x_colname, y_colname, title, x_label, y_label, color_by_col, hue_order=None, palette="colorblind",
                legend=True, legend_title=None, legend_location="lower right", legend_dict=None, sort=True, rotate_x_ticks=90):
    sns.set(rc={'figure.figsize': (22, 10)})
    if sort is True:
        data_df.sort_values(by=y_colname, inplace=True, ascending=False)
    ax = sns.barplot(x=x_colname, y=y_colname, hue=color_by_col, data=data_df, palette=palette,
                     hue_order=hue_order)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(x_label, fontsize=19)
    ax.set_ylabel(y_label, fontsize=19)
    ax.tick_params(labelsize=14)
    if legend is True:
        lgnd = ax.legend(title=legend_title, fontsize=14, loc=legend_location)
        if legend_dict is not None:
            for key, item in legend_dict.items():
                lgnd.get_texts()[key].set_text(item)
        lgnd.get_title().set_fontsize(15)

    plt.xticks(rotation=rotate_x_ticks)
    plt.savefig("{}.png".format(title), bbox_inches='tight')
    # plt.show()
    plt.gcf().clear()


def vis_boxplot(data_df, x_colname, y_colname, title, x_label, y_label, color_by_col, hue_order=None, palette="colorblind",
                legend=True, legend_title=None, legend_location="lower right", legend_dict=None, sort=True, rotate_x_ticks=90,
                x_dict=None):
    sns.set(rc={'figure.figsize': (22, 10)})
    if sort is True:
        data_df.sort_values(by=y_colname, inplace=True, ascending=False)
    if x_dict:
        x_axis_order = x_dict.values()
        data_df[x_colname] = data_df[x_colname].map(x_dict).copy()
    else:
        x_axis_order = None

    ax = sns.boxplot(x=x_colname, y=y_colname, hue=color_by_col, data=data_df, palette=palette, hue_order=hue_order, order=x_axis_order)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(x_label, fontsize=19)
    ax.set_ylabel(y_label, fontsize=19)
    ax.tick_params(labelsize=14)
    if legend is True:
        lgnd = ax.legend(title=legend_title, fontsize=14, loc=legend_location)
        if legend_dict is not None:
            for key, item in legend_dict.items():
                lgnd.get_texts()[key].set_text(item)
        lgnd.get_title().set_fontsize(15)

    plt.xticks(rotation=rotate_x_ticks)
    plt.savefig("{}.png".format(title), bbox_inches='tight')
    # plt.show()
    plt.gcf().clear()


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

    ax = sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", annot_kws={"size": 15})
    ax.figure.axes[-1].set_ylabel('Percent Overlap', size=14)
    for t in ax.texts:
        t.set_text("{0:.0%}".format(float(t.get_text())))
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.6, 0.8, 1])
    cbar.set_ticklabels(["60%", "80%", "100%"])

    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

