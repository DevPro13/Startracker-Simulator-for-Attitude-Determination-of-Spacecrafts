import pandas as pd

data = pd.read_csv("hipparcos.csv")
df = pd.DataFrame(data)
df.head()
