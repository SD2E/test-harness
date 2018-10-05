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
    csvs = ["performances_114k-CNN-stabilityscore.csv"]
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

if __name__ == '__main__':
    main()
