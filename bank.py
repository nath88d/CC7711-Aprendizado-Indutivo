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
encoder = LabelEncoder()
target = encoder.fit_transform(dados_df['subscribed'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_columns:
    dados_df[col] = dados_df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x)
    dados_df[col] = encoder.fit_transform(dados_df[col])

features = dados_df.drop('subscribed', axis=1)
arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics.ConfusionMatrixDisplay.from_estimator(arvore, features, target, display_labels=['no', 'yes'], values_format='d', ax=ax)
    plt.show(block=False)  # Não bloqueia o código
    plt.pause(0.1)  # Pausa o tempo necessário para a janela ser exibida

def plot_decision_tree():
    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(arvore, feature_names=features.columns, class_names=['no', 'yes'], filled=True, rounded=True)
    plt.show(block=False)  # Não bloqueia o código
    plt.pause(0.1)  # Pausa para a janela ser exibida

plot_confusion_matrix()
plot_decision_tree()

plt.show()
