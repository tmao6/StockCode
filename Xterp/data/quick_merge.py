import pandas as pd
import os

stock_data = pd.read_csv('words/VOX.csv')

for file in os.listdir('words'):
     filename = os.fsdecode(file)
     if not filename.endswith("X.csv"):
         print(os.path.join('words', filename))
         new_data = pd.read_csv(os.path.join('words', filename))
         stock_data = stock_data.merge(new_data, how='outer', sort=True)
         print(stock_data)


stock_data = stock_data.interpolate(limit_direction='both')

export_csv = stock_data.to_csv('joined.csv', index=False, header=True)

