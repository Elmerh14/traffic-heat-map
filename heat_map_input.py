import pandas as pd

df = pd.read_excel("CleanData.xlsx")
df_sample = df.head(50)  # or use df.sample(50) for random selection
df_sample.to_csv("heatmap_input.csv", index=False)
