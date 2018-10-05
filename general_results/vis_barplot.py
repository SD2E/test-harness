import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [f for f in files if '.csv' in f]
performance_files = [f for f in files if 'performances' in f]
feature_files = [f for f in files if 'features' in f]

frames = []
for f in performance_files:
    df = pd.read_csv(f, comment='#')
    model_used = f.split('-')[1]
    df['model_used'] = model_used
    frames.append(df)

performance_df = pd.concat(frames, axis=0)
print(performance_df.sort_values("R Squared"))
print()

performance_df.loc[performance_df['R Squared'] < -0.5, "R Squared"] = -0.5
print(performance_df.sort_values("R Squared"))

generalizability_df = performance_df.loc[performance_df['Data Set Description'].isin(["2", "4", "6"])]
agg_df = performance_df.loc[performance_df['Data Set Description'].isin(["1", "3", "5", "7"])]

g_dict = {2: "4 Topologies in Train\n11 Topologies in Test", 4: "11 Topologies in Train\n21 Topologies in Test",
          6: "21 Topologies in Train\n21 Topologies in Test"}
generalizability_df["Data Set Description"] = generalizability_df["Data Set Description"].map(g_dict).copy()
a_dict = {1: "Rocklin Data", 3: "SD2 Round 1\n+ Previous Data", 5: "SD2 Round 2", 7: "SD2 Round 3"}
agg_df["Data Set Description"] = agg_df["Data Set Description"].map(a_dict).copy()

palette3 = {'linreg': "#FFE4B5", 'RFR': '#96CDCD', 'CNN': '#B0C4DE'}

ax = sns.barplot(x="Data Set Description", y="R Squared", hue="model_used", data=generalizability_df, palette=palette3,
                 hue_order=['linreg', 'RFR', 'CNN'])
ax.set_title("Iterative Prediction of Stability Score in New Datasets", fontsize=22)
ax.set_xlabel("Complexity (# Topologies)", fontsize=19)
ax.set_ylabel("Accuracy (R^2)", fontsize=19)
ax.tick_params(labelsize=14)
legend = ax.legend(title="Model", fontsize=14, loc="lower right")
legend.get_texts()[0].set_text('Baseline Linear Regression')
legend.get_texts()[1].set_text('Random Forest Regression')
legend.get_texts()[2].set_text('Sequence Based CNN')
legend.get_title().set_fontsize(15)

plt.show()
