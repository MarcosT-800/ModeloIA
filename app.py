from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Supondo que você tenha um DataFrame com dados de clientes
dados_clientes = pd.DataFrame({
    'ano_veiculo': [2020, 2019, 2018],
    'valor_seguro': [1500.00, 1200.00, 800.00],
    'apolice_ativa': [1, 1, 0],  # 1 para ativa, 0 para inativa
    'sinistro_passado': [0, 1, 0],  # 1 para sim, 0 para não
    # Adicione mais campos conforme necessário
})

# Separando características (X) e o alvo (y)
X = dados_clientes.drop('sinistro_passado', axis=1)
y = dados_clientes['sinistro_passado']

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o modelo
print(classification_report(y_test, y_pred))
