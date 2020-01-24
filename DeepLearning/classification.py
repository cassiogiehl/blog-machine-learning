#%%
import pandas as pd


#%%
previsores = pd.read_csv('DeepLearning\entradas-breast.csv')
classe = pd.read_csv('DeepLearning\saidas-breast.csv')


#%%
from sklearn.model_selection import train_test_split
X_treino, x_teste, y_treino, y_teste = train_test_split()