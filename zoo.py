import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

dados, meta = arff.loadarff('./zoo.arff')
dados_df = pd.DataFrame(dados)

print(dados_df['type'].isna().sum())
dados_df = dados_df.dropna(subset=['type'])
print(dados_df['type'].isna().sum())

codificador = LabelEncoder()
alvo = codificador.fit_transform(dados_df['type'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

colunas_categoricas = ['animal', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']

for coluna in colunas_categoricas:
    dados_df[coluna] = dados_df[coluna].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x)
    dados_df[coluna] = codificador.fit_transform(dados_df[coluna])

caracteristicas = dados_df.drop('type', axis=1)

arvore_decisao = DecisionTreeClassifier(criterion='entropy').fit(caracteristicas, alvo)

def plot_matriz_confusao():
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics.ConfusionMatrixDisplay.from_estimator(arvore_decisao, caracteristicas, alvo, display_labels=['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate'], values_format='d', ax=ax)
    plt.show(block=False)
    plt.pause(0.1)

def plot_arvore_decisao():
    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(arvore_decisao, feature_names=caracteristicas.columns, class_names=['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate'], filled=True, rounded=True)
    plt.show(block=False)
    plt.pause(0.1)
plot_matriz_confusao()
plot_arvore_decisao()

plt.show()
