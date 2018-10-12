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
# print(performance_df.sort_values("R Squared").head(20))
# print()
performance_df.loc[performance_df['R Squared'] < -1.3, "R Squared"] = -1.3
# print(performance_df.sort_values("R Squared").head(20))

# Leave one group out plots:
ss_df = performance_df.loc[performance_df['Column Predicted'] == "stabilityscore"].copy()
ss_cal_df = performance_df.loc[performance_df['Column Predicted'] == "stabilityscore_calibrated"].copy()
ss_cal2_df = performance_df.loc[performance_df['Column Predicted'] == "stabilityscore_calibrated_v2"].copy()
ss_cnn_df = performance_df.loc[performance_df['Column Predicted'] == "stabilityscore_cnn"].copy()
ss_cnn_cal_df = performance_df.loc[performance_df['Column Predicted'] == "stabilityscore_cnn_calibrated"].copy()

sd2_impact_df = ss_df.loc[ss_df["Data Set Description"].isin(["16k", "114k"])]

x_dict = {"16k": "Rocklin Dataset (16k)\n4 Topologies Total", "81k": "SD2 Round 1 (81k)\n11 Topologies Total",
          "105k": "SD2 Round 2 (105k)\n21 Topologies Total", "114k": "SD2 Round 3 (114k)\n21 Topologies Total"}
ss_df["Data Set Description"] = ss_df["Data Set Description"].map(x_dict).copy()

sd2_impact_dict = {"16k": "Rocklin Dataset (16k)\n4 Topologies Total",
                   "114k": "SD2 Round 3 (114k)\n21 Topologies Total"}
sd2_impact_df["Data Set Description"] = sd2_impact_df["Data Set Description"].map(sd2_impact_dict).copy()

palette3 = {'linreg': "#FFE4B5", 'RFR': '#96CDCD', 'CNN': '#B0C4DE'}

# ax = sns.boxplot(x="Data Set Description", y="R Squared", hue="model_used", data=ss_df, palette=palette3,
#                   order=x_dict.values(), hue_order=['linreg', 'RFR', 'CNN'])
#
# ax.set_title("Leave-One-Group-Out Predictions of Stability Score as Dataset Complexity Increases", fontsize=22)
# ax.set_xlabel("Complexity (# Topologies)", fontsize=19)
# ax.set_ylabel("Accuracy (R^2)", fontsize=19)
# ax.tick_params(labelsize=14)
# legend = ax.legend(title="Model", fontsize=14, loc="lower right")
# legend.get_texts()[0].set_text('Baseline Linear Regression')
# legend.get_texts()[1].set_text('Random Forest Regression')
# legend.get_texts()[2].set_text('Sequence Based CNN')
# legend.get_title().set_fontsize(15)





# ax = sns.boxplot(x="Data Set Description", y="R Squared", hue="model_used", data=ss_cal_df, palette=palette3,
#                   order=['16k', '81k', '105k', '114k'], hue_order=['linreg', 'RFR', 'CNN'])
# ax.set_title("stabilityscore_calibrated")
#
# ax = sns.boxplot(x="Data Set Description", y="R Squared", hue="model_used", data=ss_cal2_df, palette=palette3,
#                   order=['16k', '81k', '105k', '114k'], hue_order=['linreg', 'RFR', 'CNN'])
# ax.set_title("stabilityscore_calibrated_v2")
#
# ax = sns.boxplot(x="Data Set Description", y="R Squared", hue="model_used", data=ss_cnn_df, palette=palette3,
#                   order=['16k', '81k', '105k', '114k'], hue_order=['linreg', 'RFR', 'CNN'])
# ax.set_title("stabilityscore_cnn")
#
# ax = sns.boxplot(x="Data Set Description", y="R Squared", hue="model_used", data=ss_cnn_cal_df, palette=palette3,
#                   order=['16k', '81k', '105k', '114k'], hue_order=['linreg', 'RFR', 'CNN'])
# ax.set_title("stabilityscore_cnn_calibrated")


# Everything together:
# catplot = sns.catplot(x="Data Set Description", y="R Squared", hue="model_used", data=performance_df, palette=palette3,
#                       order=['16k', '81k', '105k', '114k'], hue_order=['linreg', 'RFR', 'CNN'], col="Column Predicted",
#                       kind="box")


## Calibration/USM comparisons:
# df_16k = performance_df.loc[performance_df['Data Set Description'] == "16k"].copy()
# df_81k = performance_df.loc[performance_df['Data Set Description'] == "81k"].copy()
# df_105k = performance_df.loc[performance_df['Data Set Description'] == "105k"].copy()
# df_114k = performance_df.loc[performance_df['Data Set Description'] == "114k"].copy()

# ax = sns.boxplot(x="model_used", y="R Squared", hue="Column Predicted", data=df_16k, palette="Set3",
#                  order=['linreg', 'RFR', 'CNN'], hue_order=['stabilityscore', 'stabilityscore_calibrated',
#                                                             'stabilityscore_calibrated_v2', 'stabilityscore_cnn',
#                                                             'stabilityscore_cnn_calibrated'])
# ax.set_title("Calibration and USM Comparisons for 16k Dataset")

# ax = sns.boxplot(x="model_used", y="R Squared", hue="Column Predicted", data=df_81k, palette="Set3",
#                  order=['linreg', 'RFR', 'CNN'], hue_order=['stabilityscore', 'stabilityscore_calibrated',
#                                                             'stabilityscore_calibrated_v2', 'stabilityscore_cnn',
#                                                             'stabilityscore_cnn_calibrated'])
# ax.set_title("Calibration and USM Comparisons for 81k Dataset")
#
# ax = sns.boxplot(x="model_used", y="R Squared", hue="Column Predicted", data=df_105k, palette="Set3",
#                  order=['linreg', 'RFR', 'CNN'], hue_order=['stabilityscore', 'stabilityscore_calibrated',
#                                                             'stabilityscore_calibrated_v2', 'stabilityscore_cnn',
#                                                             'stabilityscore_cnn_calibrated'])
# ax.set_title("Calibration and USM Comparisons for 105k Dataset")
#
# ax = sns.boxplot(x="model_used", y="R Squared", hue="Column Predicted", data=df_114k, palette="Set3",
#                  order=['linreg', 'RFR', 'CNN'], hue_order=['stabilityscore', 'stabilityscore_calibrated',
#                                                             'stabilityscore_calibrated_v2', 'stabilityscore_cnn',
#                                                             'stabilityscore_cnn_calibrated'])
# ax.set_title("Calibration and USM Comparisons for 114k Dataset")
#
#
#
# ax.set_title("Leave-One-Group-Out Predictions of Different Types of Stability Scores for 3 Models", fontsize=22)
# ax.set_xlabel("Complexity (# Topologies)", fontsize=19)
# ax.set_ylabel("Accuracy (R^2)", fontsize=19)
# ax.tick_params(labelsize=14)
# ax.set_xticklabels(['Baseline Linear Regression', 'Random Forest Regression', 'Sequence Based CNN'])
# legend = ax.legend(title="Column Predicted", fontsize=14, loc="lower right")
# legend.get_title().set_fontsize(15)
# legend.get_texts()[3].set_text('stabilityscore_USM')
# legend.get_texts()[4].set_text('stabilityscore_USM_calibrated')
#
#
# plt.show()
