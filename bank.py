import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

dados, meta = arff.loadarff('./bank.arff')
dados_df = pd.DataFrame(dados)

print(dados_df['subscribed'].isna().sum())
dados_df = dados_df.dropna(subset=['subscribed'])
print(dados_df['subscribed'].isna().sum())

codificador = LabelEncoder()
alvo = codificador.fit_transform(dados_df['subscribed'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

colunas_categoricas = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in colunas_categoricas:
    dados_df[col] = dados_df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x)
    dados_df[col] = codificador.fit_transform(dados_df[col])

caracteristicas = dados_df.drop('subscribed', axis=1)

arvore = DecisionTreeClassifier(criterion='entropy').fit(caracteristicas, alvo)

def plot_matriz_confusao():
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics.ConfusionMatrixDisplay.from_estimator(arvore, caracteristicas, alvo, display_labels=['no', 'yes'], values_format='d', ax=ax)
    plt.show(block=False)
    plt.pause(0.1)

def plot_arvore_decisao():
    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(arvore, feature_names=caracteristicas.columns, class_names=['no', 'yes'], filled=True, rounded=True)
    plt.show(block=False)
    plt.pause(0.1)

plot_matriz_confusao()
plot_arvore_decisao()

plt.show()
