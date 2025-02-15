
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt

#%%
#Usando a função read_csv para ler os aquivos de teste e de treinamento
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% Exploração inicial
#Utilisa a funcão .head() para explorar o DataFrame
train.head()

#Utiliza função info() para observar o quantas observações tem em cada variavel
print(train.info())

#Ver se tem mt Nan
print(train.isnull().sum())
print('___________________')
print(test.isnull().sum())

#%%Tratar os null

# Substituir valores NaN ao invés de excluir todas as linhas
#Tratando Treino
train.fillna({
    'Age': train['Age'].median(),
    'CryoSleep': False,
    'VIP': False,
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0
}, inplace=True)

#tratando teste
test.fillna({
    'Age': test['Age'].median(),
    'CryoSleep': False,
    'VIP': False,
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0
}, inplace=True)


#%% Categorizando a idade na base de treino

kbin_idade = KBinsDiscretizer(n_bins= 5, encode= 'ordinal', strategy= 'quantile')
train.loc[:, 'cat_idade'] = kbin_idade.fit_transform(train[['Age']])


#Covertendo Booleanos para int
train['CryoSleep'] = train['CryoSleep'].astype(int)
train['VIP'] = train['VIP'].astype(int)

#excluindo colunas que não ajudam
#train.drop(columns = ['VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService'], inplace= True)

#%% Definino as variaveis para treinar a arvore
X_train = train[['cat_idade', 'CryoSleep', 'VIP']]
Y_train = train['Transported'].astype(int)


#%%Iniciando a arvore

# Criando o modelo de Random Forest
floresta = RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42)

# Treinando o modelo com os dados de treino
floresta.fit(X_train, Y_train)

#%% Importancia de cada variavel (feature) para o modelo

#Separando as features que serão analizadas
importancia = floresta.feature_importances_
features= X_train.columns

#Plotando grafico para observar quais são mais importantes
plt.figure(figsize=(10, 5)) #Tamanho da figura
plt.bar(features, importancia, color= 'teal') #Grafico de barras, quais variaveis usar, cor.
plt.xlabel('Variaveis') #Legenda do eixo X
plt.ylabel('Importancia') #Legenda do eixo Y
plt.title('Importancia das variavies na Random Forest') #Titulo do grafico
plt.show()

#%% Preparando os dados de teste

#Categorizando a idade na base de teste
kbin_idade_teste = KBinsDiscretizer(n_bins= 5, encode= 'ordinal', strategy= 'quantile')
test.loc[:, 'cat_idade'] = kbin_idade_teste.fit_transform(test[['Age']])

#Somando todos os gastos dos passageiros 'Essa variavel acabei não utilizando, mas futuramente vou avalia, por isso deichei aqui'
test["total_gasto"] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']


#Covertendo Booleanos para int
test['CryoSleep'] = test['CryoSleep'].astype(int)
test['VIP'] = test['VIP'].astype(int)

#%% Fazendo a previsão

#definindo as variaveis
X_teste = test[['cat_idade', 'CryoSleep', 'VIP']]

#Executando a previsão
Y_pred = floresta.predict(X_teste)

#%% Adicionando PassengerId ao DataFrame de resultados
resultado = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # Identificador do passageiro
    'Transported': pd.Series(Y_pred).replace({0: False, 1: True})  # Resultado da previsão
})

# Exibir as primeiras linhas para conferir
print(resultado.head())

# Salvar como CSV para análise posterior
resultado.to_csv('SpaceShip_RandomForest.csv', index=False)

