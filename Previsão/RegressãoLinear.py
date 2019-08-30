#%%
import pandas as pd
from sklearn.datasets import load_boston

#%%
df = load_boston()
print(df.keys())
# previsores = df.data
# classificadores = df.target

#%%
# df.feature_names
print(df.DESCR)