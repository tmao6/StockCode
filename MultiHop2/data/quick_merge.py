import pandas as pd

left_data = pd.read_csv('FirstFourVOX.csv')
right_data = pd.read_csv('LastThreeVOX.csv')

data = left_data.merge(right_data, how='inner', sort=False)
data = data.rename(columns={'Unnamed: 0':''})

export_csv = data.to_csv('joined.csv', index=False, header=True)
