from sklearn.datasets import load_iris

# Importando a base de dados Íris
df = load_iris()
# Definindo atributos que vão ser utilizados como previsores
previsores = df.data
# Definindo os atributos de classe
classificadores = df.target

# Importando biblioteca que separa os dados de treino e teste
from sklearn.model_selection import train_test_split

# Definindo variáveis de treino e teste. Definindo tamanho do teste e seleção randômica de atributos.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    previsores, classificadores, test_size=0.30, random_state=42)

# Importando biblioteca que fará a predição dos atributos
from sklearn.naive_bayes import GaussianNB

# Instanciando o objeto da classe na variável
clf = GaussianNB()

# Aqui são passados os dados de treinamento (amostra de 70%) para que o algoritmo possa fazer a 
# aprendizagem e perceber os padrões no dataset
clf = clf.fit(X_treino, y_treino)

# Após o treinamento, colocamos em prática os dados de teste (amostra de 30%) 
# para que o algoritmo nos retorne os resultados que previu
print(clf.predict(X_teste))

# Além de treinar o modelo, podemos validar sua acurácia

# Importando biblioteca que fará a verificação da acurácia
from sklearn.model_selection import cross_val_score

# Distribuindo os dados em 5 partes iguais
# Cada parte será usada como teste, uma por vez, fazendo o treinamento das restantes
acur = cross_val_score(clf, previsores, classificadores, cv=5)
print("Acuracidade de cada item: {}".format(acur))

# Calculando a média da acurácia do modelo
mean_acur = sum(acur) / len(acur)
print("Média de acuracidade: {}".format(mean_acur))