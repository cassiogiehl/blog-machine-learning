#%%
from sklearn.datasets import load_boston

#%%
df = load_boston()
print(df.DESCR)

#%%
previsores = df.data
alvo = df.target
alvo[0]

#%%
from sklearn.model_selection import train_test_split

# Separando dados de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size=0.3)

#%%
from sklearn.linear_model import LinearRegression

regressao_linear = LinearRegression()
regressao_linear.fit(X_treino, y_treino)
y_pred = regressao_linear.predict(X_teste)
print(y_pred)

#%%
from matplotlib import pyplot as plt

plt.scatter(x=y_teste, y=y_pred)
plt.show()
