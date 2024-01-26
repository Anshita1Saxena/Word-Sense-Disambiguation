import pandas as pd
colnames = ['id', 'context']
# time_df = pd.read_csv('time.csv', delimiter="|", names=colnames)
# level_df = pd.read_csv('level_data_again.csv', delimiter="\t", names=colnames)
# sense_df = pd.read_csv('sense_df.csv', delimiter="|", names=colnames)
life_df = pd.read_csv('life_df.csv', delimiter="|", names=colnames)
colnames2 = ['id', 'sense']
# time_sense_df = pd.read_csv('time_sense.csv', delimiter="|", names=colnames2)
# level_sense_df = pd.read_csv('level_sense.csv', names=colnames2)
# sense_sense_df = pd.read_csv('sense_keys.csv', delimiter="|", names=colnames)
life_sense_df = pd.read_csv('life_keys.csv', delimiter="|", names=colnames)
# print(level_df.columns)
# print(level_sense_df.columns)
df = pd.merge(life_df, life_sense_df, on="id", how='inner')
df.drop(columns='id', inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv('final_life.csv', index=False)
