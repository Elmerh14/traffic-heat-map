import pandas as pd

# There is like 92 thousand crashes in total so we are going to increment 

df = pd.read_excel("CleanData.xlsx")
df_sample = df.head(2000)  # or use df.sample(n) for random selection
df_sample.to_csv("heatmap_input.csv", index=False)
