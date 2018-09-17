import pandas as pd
import os

newest_ss_folder = '/Users/he/PycharmProjects/SD2/versioned-datasets/data/protein-design/experimental_stability_scores'
filenames = ['Eva1.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'Eva2.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'Inna.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'Longxing.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'topology_mining_and_Longxing_chip_1.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'topology_mining_and_Longxing_chip_2.Rocklin_calibrated_experimental_stability_scores.v4.csv',
             'topology_mining_and_Longxing_chip_3.Rocklin_calibrated_experimental_stability_scores.v4.csv']
frames = []
for f in filenames:
    df = pd.read_csv(os.path.join(newest_ss_folder, f), comment='#')
    frames.append(df)

all_recent_data = pd.concat(frames, ignore_index=True)
# print(all_recent_data)
rocklin = pd.read_csv(os.path.join(newest_ss_folder, 'Rocklin.experimental_stability_scores.v2.csv'), comment='#')

print(set(all_recent_data.columns.values).difference(rocklin.columns.values))
print(set(rocklin.columns.values).difference(all_recent_data.columns.values))

# all_recent_data = pd.concat([all_recent_data, rocklin])
# print(all_recent_data)


rosetta_folder = '/Users/he/PycharmProjects/SD2/prot-stab-data-norm/data/structural_metrics'
filenames = ['Eva1.structural_metrics.csv',
             'Eva2.structural_metrics.csv',
             'Inna.structural_metrics.csv',
             'Longxing.structural_metrics.csv',
             'topology_mining_and_Longxing_chip_1.structural_metrics.csv',
             'topology_mining_and_Longxing_chip_2.structural_metrics.csv',
             'topology_mining_and_Longxing_chip_3.structural_metrics.csv']
frames = []
for f in filenames:
    df = pd.read_csv(os.path.join(rosetta_folder, f), comment='#')
    print(df.shape)
    print(df)
    print('\n\n')
    frames.append(df)

all_recent_data = pd.concat(frames, ignore_index=True)
print(all_recent_data)
rocklin = pd.read_csv(os.path.join(rosetta_folder, 'Rocklin.structural_metrics.csv'), comment='#')
print('\n\n\n')
# print(rocklin)

print(set(all_recent_data.columns.values).difference(rocklin.columns.values))
print(set(rocklin.columns.values).difference(all_recent_data.columns.values))
