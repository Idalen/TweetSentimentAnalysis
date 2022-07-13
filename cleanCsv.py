# This cleans the csv from repeated rows between the two

import pandas as pd

# Read the csv file
df1 = pd.read_csv('./tweets_stocks-full_agreement.csv')
df2 = pd.read_csv('./tweets_stocks.csv')

clean_data = df2[~df2['tweet_id'].isin(df1['tweet_id'])]

clean_data.to_csv('tweets_stock_clean.csv', index=False)

df3 = pd.read_csv('./tweets_stock_clean.csv')
print(df3)