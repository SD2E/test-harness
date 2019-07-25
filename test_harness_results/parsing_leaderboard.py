import pandas as pd

index_dict = {'kernel':(2,str),'samples':(0,int),'iteration':(1,int),'object':(3,str)}

def parse_description(description,key):
    subset = index_dict[key][1](description.split(';')[index_dict[key][0]].split(' ')[0])
    return subset



df = pd.read_html('test_harness_results/custom_classification_leaderboard.html')[0]

test = '100 samples;0 iteration;mic kernel;'
test2 = '500 samples;1 iteration;mic kernel;antibiotic object'
entity = 'object'
print(df.columns)
for key in index_dict.keys():
    print(key)
    df[key]=[parse_description(item,key) for item in df['Data and Split Description']]

df.to_csv('test_harness_results/leaderboard_df.csv')
