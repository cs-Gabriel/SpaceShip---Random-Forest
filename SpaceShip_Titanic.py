# A biblioteca pantas é importante para a amanipulação de DataFrames
import pandas as pd

# A principal Biblioteca para utilizarmos o metodo do Random Forest
from sklearn.ensemble import RandomForestClassifier

# Importante para facilitar a categorização de variaveis
from sklearn.preprocessing import KBinsDiscretizer

# Biblioteca essencial para a vizualização de dados
import matplotlib.pyplot as plt

# Biblioteca essencial para a vizualização de dados
import seaborn as sns

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

#%% Aprofundando a analize exploratoria

#Gerando um CroosTab para buscar ligação entre as variaveis e a resposta (Targuet)

pd.crosstab(train['CryoSleep'], train['Transported']).plot(kind= 'bar',
                                                           figsize=(10,5),
                                                           color= ['salmon', 'lightblue'])
plt.title('Frequencia de Transpotados em Sono induzido')
plt.xlabel('0 = Não estava em sono induzido, 1 = Estava em sono induzido')
plt.ylabel('Amostragem')

# Asim é possivel notar que existe uma relação grande entre estar induzio e ter sido transpotado, evidenciando essa variavel como uma importante para nosso modelo

#%% Vamos fazer o mesmo grafico mas para outra variavel

pd.crosstab(train['VIP'], train['Transported']).plot(kind= 'bar',
                                                           figsize=(10,5),
                                                           color= ['salmon', 'lightblue'])
plt.title('Frequencia de Transpotados VIP')
plt.xlabel('0 = Não VIP, 1 = VIP')
plt.ylabel('Amostragem')

# Como a quantidade de VIP é muito baixa, não é possivel afirmar uma relação dessa variavel com nossa Targuet

#%% Vamos fazer o mesmo grafico mas para outra variavel

pd.crosstab(train['cat_idade'], train['Transported']).plot(kind= 'bar',
                                                           figsize=(10,5),
                                                           color= ['salmon', 'lightblue'])
plt.title('Frequencia de Transpotados VIP')
plt.xlabel('Faixa de idade')
plt.ylabel('Amostragem')

'''
É possivel ver que na primerira divizão tem mais transportados e na ultima divizão o numero de não tranpostado supera o de transportado,
mostrando uma tendencia, assim essa variavel tem relaçao com a Targuet'''

#%% Separando as variaveis para fazer um mapa de calor buscando ligação entre elas
variaves = train[['Age','CryoSleep','VIP','RoomService','FoodCourt','ShoppingMall','Spa', 'VRDeck']]

#Criando o mapa de calor, não foi possivel encontra nenhuma relação muito relevante.
corr_matrix = variaves.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix,
            annot=True,
            linewidths= 0.5,
            fmt='.2f',
            cmap='YlGnBu')

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

