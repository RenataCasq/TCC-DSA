# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:17:29 2025

@author: Renata Alves
"""
#%%
# Intalação de Pacotes

!pip install pandas
!pip install numpy
!pip install seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests
!pip install xgboost
!pip install openpyxl

#%%
# Importação dos Pacotes 

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import os
from datetime import datetime

#%%
# Seleção o diretório com arquivos

os.chdir(r'C:\Users\Renata Alves\TCC')

#%%
# Importação dos Dados

vendas22 = pd.read_excel('vendas2022.xlsx', parse_dates=['Data Emissão'])
vendas23 = pd.read_excel('vendas2023.xlsx', parse_dates=['Data Emissão'])
produtos22 = pd.read_excel('produtos2022.xlsx')
produtos23 = pd.read_excel('produtos2023.xlsx')
produtos24 = pd.read_excel('produtos2024.xlsx')
vendas24real = pd.read_excel('vendas2024.xlsx', parse_dates=['Data Emissão'])

#%%
# Informações gerais sobre o DataFrame
    #Sobre VENDAS
print(vendas22.info())
print(vendas23.info())
print(vendas24real.info())
    #Sobre PRODUTOS
print(produtos22.info())
print(produtos23.info())
print(produtos24.info())

#%%
# Lista das colunas que quero transformar Variáveis de Vendas em string
colunas_para_str = ['Representante', 'Cliente', 'Marca', 'Pedido', 'Variante', 'Produto', 'Cor']

# Aplicar a conversão dos tipos das variáveis de Vendas
vendas22[colunas_para_str] = vendas22[colunas_para_str].astype(str)
vendas23[colunas_para_str] = vendas23[colunas_para_str].astype(str)
vendas24real[colunas_para_str] = vendas24real[colunas_para_str].astype(str)

#Ajuste nos tipos de Variáveis em Produtos
produtos22['Referência'] = produtos22['Referência'].astype(str)
produtos23['Referência'] = produtos23['Referência'].astype(str)
produtos24['Referência'] = produtos24['Referência'].astype(str)

#%%
# Estatísticas descritiva das variáveis
    #Sobre VENDAS
print(vendas22.describe())
print(vendas23.describe())

    #Sobre PRODUTOS
print(produtos22.describe())
print(produtos23.describe())
print(produtos24.describe())


#%%
# ANÁLISE EXPLORATÓRIA DOS DADOS

#%%
#Comparação do volume de vendas entre 2022 e 2023 (Qtde Peças e R$)

# Agrupar vendas por ano
vendas_ano_22 = vendas22[['Quantidade Pedida Por Variante', 'Valor Pedido por Variante']].sum()
vendas_ano_23 = vendas23[['Quantidade Pedida Por Variante', 'Valor Pedido por Variante']].sum()

# Definir estilo visual
sns.set(style='whitegrid')

# Dados das Vendas
df_vendas = pd.DataFrame({
    'Ano': ['2022', '2023'],
    'Qtd Peças': [vendas_ano_22[0], vendas_ano_23[0]],
    'Valor R$': [vendas_ano_22[1], vendas_ano_23[1]]})

# Criar figura
fig, ax1 = plt.subplots(figsize=(8,6))

# Quantidade de Peças - nas barras
cor_barras = '#4C72B0'
ax1.bar(df_vendas['Ano'], df_vendas['Qtd Peças'], color=cor_barras, width=0.4)
ax1.set_xlabel('Ano', fontsize=12)
ax1.set_ylabel('Quantidade de Peças', color=cor_barras, fontsize=12)
ax1.tick_params(axis='y', labelcolor=cor_barras)

for i, v in enumerate(df_vendas['Qtd Peças']):
    ax1.text(i, v + v*0.02, f'{int(v):,}', ha='center', color=cor_barras, fontsize=10, fontweight='bold')

# Valor em Reais - nas linhas
ax2 = ax1.twinx()
cor_linha = '#C44E52'
ax2.plot(df_vendas['Ano'], df_vendas['Valor R$'], color=cor_linha, marker='o', linewidth=2)
ax2.set_ylabel('Valor em R$', color=cor_linha, fontsize=12)
ax2.tick_params(axis='y', labelcolor=cor_linha)

for i, v in enumerate(df_vendas['Valor R$']):
    ax2.text(i, v + v*0.02, f'R$ {int(v):,}', ha='center', color=cor_linha, fontsize=10, fontweight='bold')

# 🔧 Título e grid
plt.title('Comparação de Vendas – Quantidade de Peças (Barras) e Valor em R$ (Linha)', fontsize=13, weight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%%
# Comparação da quantidade de referências disponíveis nos anos 2022, 2023 e 2024

qtd_ref_22 = produtos22['Referência'].nunique()
qtd_ref_23 = produtos23['Referência'].nunique()
qtd_ref_24 = produtos24['Referência'].nunique()

df_ref = pd.DataFrame({
    'Ano': ['2022', '2023', '2024'],
    'Qtd de Refs': [qtd_ref_22, qtd_ref_23, qtd_ref_24]})

ax = sns.barplot(data=df_ref, x='Ano', y='Qtd de Refs', color='#4C72B0')

# Título
plt.title('Quantidade de Referências Disponíveis para Vendas por Ano')

# 🔥 Adicionar os rótulos de valor em cima de cada barra
for i, v in enumerate(df_ref['Qtd de Refs']):
    ax.text(i, v + v * 0.02,  # posição: eixo x = i, eixo y = levemente acima do topo da barra
            str(v),            # o valor que será mostrado
            ha='center',       # alinhamento horizontal
            fontsize=10,       # tamanho da fonte
            fontweight='bold', # negrito
            color=cor_barras)

plt.tight_layout()
plt.show()

#%%
# Análise gráfica comparativa das vendas mensais entre 2022 e 2023 

# Criar coluna 'Mes' (número do mês) para ambos os anos
vendas22['Mes'] = vendas22['Data Emissão'].dt.month
vendas23['Mes'] = vendas23['Data Emissão'].dt.month

# Agrupar vendas mensais (independente do ano)
vendas_mes_22 = vendas22.groupby('Mes')['Quantidade Pedida Por Variante'].sum().reset_index()
vendas_mes_23 = vendas23.groupby('Mes')['Quantidade Pedida Por Variante'].sum().reset_index()

# Mudança do número do mês para nome
meses_nome = {
    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}

vendas_mes_22['MesNome'] = vendas_mes_22['Mes'].map(meses_nome)
vendas_mes_23['MesNome'] = vendas_mes_23['Mes'].map(meses_nome)

# Construção do gráfico
plt.figure(figsize=(9,6))
sns.set(style='whitegrid')

plt.plot(vendas_mes_22['MesNome'], vendas_mes_22['Quantidade Pedida Por Variante'], 
         marker='o', color='#4C72B0', linewidth=2, label='2022')

plt.plot(vendas_mes_23['MesNome'], vendas_mes_23['Quantidade Pedida Por Variante'], 
         marker='o', color='#C44E52', linewidth=2, label='2023')

for x, y in zip(vendas_mes_22['MesNome'], vendas_mes_22['Quantidade Pedida Por Variante']):
    plt.text(x, y + y*0.02, f'{int(y):,}', ha='center', fontsize=9, color='#4C72B0')

for x, y in zip(vendas_mes_23['MesNome'], vendas_mes_23['Quantidade Pedida Por Variante']):
    plt.text(x, y + y*0.02, f'{int(y):,}', ha='center', fontsize=9, color='#C44E52')

plt.title('Comparativo da Evolução das Vendas Mensais – 2022 x 2023 (Por Mês)', fontsize=14, weight='bold')
plt.ylabel('Quantidade Pedida', fontsize=11)
plt.xlabel('Mês', fontsize=11)

plt.show()

#%% 
# Agrupando as duas tabelas por ano (2022 e 2023)

# Renomear a coluna 'Produto' para 'Referência' na tabela de vendas
vendas22 = vendas22.rename(columns={'Produto': 'Referência'})
vendas23 = vendas23.rename(columns={'Produto': 'Referência'})

# Merge das tabelas de vendas com produtos
vendas_prod22 = vendas22.merge(produtos22, on='Referência', how='left')
vendas_prod23 = vendas23.merge(produtos23, on='Referência', how='left')


#%%
# Gráfico do Comportamento das Vendas Anuais – Separado por Categoria (2022 e 2023)

# Criar a tabela de vendas por categoria e ano
vendas_cat_22 = vendas_prod22.groupby('Categoria')['Quantidade Pedida Por Variante'].sum().reset_index()
vendas_cat_22['Ano'] = '2022'

vendas_cat_23 = vendas_prod23.groupby('Categoria')['Quantidade Pedida Por Variante'].sum().reset_index()
vendas_cat_23['Ano'] = '2023'

df_cat = pd.concat([vendas_cat_22, vendas_cat_23])

# Pivotar para organizar como tabela
tabela = df_cat.pivot(index='Categoria', columns='Ano', values='Quantidade Pedida Por Variante').fillna(0).astype(int)

# Calcular diferença percentual de 2022 para 2023
tabela['Diferença (%)'] = ((tabela['2023'] - tabela['2022']) / tabela['2022'].replace(0, 1)) * 100
tabela['Diferença (%)'] = tabela['Diferença (%)'].round(1)

# Ordenar pela quantidade de 2023 (opcional)
tabela = tabela.sort_values(by='Diferença (%)', ascending=False)

# Criando tabela 
tabela = tabela.reset_index()
tabela.columns = ['Categoria', '2022', '2023', 'Diferença (%)']

fig, ax = plt.subplots(figsize=(8, len(tabela) * 0.5))
ax.axis('off')  # Remove os eixos

tabela_plot = ax.table(cellText=tabela.values,
                        colLabels=tabela.columns,
                        loc='center',
                        cellLoc='center')

tabela_plot.auto_set_font_size(False)
tabela_plot.set_fontsize(10)
tabela_plot.scale(1.2, 1.2)

plt.title('Vendas por Categoria – 2022 x 2023 + Diferença (%)', fontsize=14, weight='bold')
plt.tight_layout()

plt.show()

#%%
# Analise do comportamento de vendas por mes por região do país nos anos de 2022 e 2023

# Dicionário de mapeamento dos estados para regiões do Brasil
mapa_regioes = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'}

# Criar coluna 'Região' em vendas2022 e vendas2023
vendas22['Região'] = vendas22['Estado'].map(mapa_regioes)
vendas23['Região'] = vendas23['Estado'].map(mapa_regioes)

# Obter a lista de regiões existentes
regioes = sorted(vendas22['Região'].dropna().unique())

# Loop para gerar um gráfico para cada região
for regiao in regioes:
    # Filtrar dados por região
    vendas22_reg = vendas22[vendas22['Região'] == regiao]
    vendas23_reg = vendas23[vendas23['Região'] == regiao]

    # Criar coluna 'Mes' (número do mês)
    vendas22_reg['Mes'] = vendas22_reg['Data Emissão'].dt.month
    vendas23_reg['Mes'] = vendas23_reg['Data Emissão'].dt.month

    # Agrupar vendas mensais (independente do ano)
    vendas_mes_22 = vendas22_reg.groupby('Mes')['Quantidade Pedida Por Variante'].sum().reset_index()
    vendas_mes_23 = vendas23_reg.groupby('Mes')['Quantidade Pedida Por Variante'].sum().reset_index()

    meses_nome = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }

    vendas_mes_22['MesNome'] = vendas_mes_22['Mes'].map(meses_nome)
    vendas_mes_23['MesNome'] = vendas_mes_23['Mes'].map(meses_nome)

    # Criando Gráfico
    plt.figure(figsize=(9,6))
    sns.set(style='whitegrid')
    
    plt.plot(vendas_mes_22['MesNome'], vendas_mes_22['Quantidade Pedida Por Variante'], 
             marker='o', color='#4C72B0', linewidth=2, label='2022')

    plt.plot(vendas_mes_23['MesNome'], vendas_mes_23['Quantidade Pedida Por Variante'], 
             marker='o', color='#C44E52', linewidth=2, label='2023')

    for x, y in zip(vendas_mes_22['MesNome'], vendas_mes_22['Quantidade Pedida Por Variante']):
        plt.text(x, y + y*0.02, f'{int(y):,}', ha='center', fontsize=9, color='#4C72B0')

    for x, y in zip(vendas_mes_23['MesNome'], vendas_mes_23['Quantidade Pedida Por Variante']):
        plt.text(x, y + y*0.02, f'{int(y):,}', ha='center', fontsize=9, color='#C44E52')

    plt.title(f'Comparativo do Comportamento das Vendas Mensais – Região {regiao}', fontsize=14, weight='bold')
    plt.ylabel('Quantidade Pedida', fontsize=11)
    plt.xlabel('Mês', fontsize=11)

    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Ano', fontsize=10, title_fontsize=11)
    plt.tight_layout()

    plt.show()
 
#%%
# Análises de Correlação das Variáveis (2022 e 2023 juntas)

# Concatenar as tabelas de vendas de 2022 e 2023
vendas_geral = pd.concat([vendas_prod22, vendas_prod23])

# Definir quais variáveis serão incluídas na análise de correlação
# Variáveis categóricas que serão transformadas em dummies (one-hot)
variaveis_categoricas = ['Categoria', 'Modelagem', 'Linha', 'Grupo MP', 'Marca']

# Variáveis numéricas diretamente relacionadas ao volume e valor
variaveis_numericas = ['Quantidade Pedida Por Variante', 'Valor Pedido por Variante', 'Valor Unitário da Variante']

# Selecionar apenas as colunas relevantes
dados_corr = vendas_geral[variaveis_numericas + variaveis_categoricas]

# Aplicar One Hot Encoding nas variáveis categóricas
dados_corr_encoded = pd.get_dummies(dados_corr, drop_first=False)

# Calcular a matriz de correlação
corr = dados_corr_encoded.corr()

# Gerar o mapa de calor da correlação
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)

plt.title('Mapa de Correlação – Dados Gerais (2022 + 2023)', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#%%
# Análises de Correlação das Variáveis - correlações fortes

# Gerar a matriz de correlação
corr = dados_corr_encoded.corr()

# Filtrar apenas correlações fortes (acima de 0.5 ou abaixo de -0.5)
corr_filtrada = corr[(corr >= 0.5) | (corr <= -0.5)]

plt.figure(figsize=(12,10))
sns.heatmap(corr_filtrada, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Mapa de Correlação – Apenas Correlações Fortes (2022 + 2023)', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#%%

# Função para gerar o mapa de correlação para uma variável categórica
def mapa_correlacao_categoria(variavel_categorica, dados):
    # Selecionar colunas que contêm a variável categórica + variáveis numéricas
    colunas_categoricas = [col for col in dados.columns if variavel_categorica + '_' in col]
    colunas_numericas = ['Quantidade Pedida Por Variante', 'Valor Pedido por Variante', 'Valor Unitário da Variante']
    
    colunas_interesse = colunas_categoricas + colunas_numericas
    
    # Calcular matriz de correlação
    corr = dados[colunas_interesse].corr()

    # Selecionar só correlação das categorias com as variáveis numéricas
    corr_interesse = corr.loc[colunas_categoricas, colunas_numericas]

    # Plotar
    plt.figure(figsize=(10, max(6, len(colunas_categoricas) * 0.5)))  # Ajusta altura conforme nº de categorias
    sns.heatmap(corr_interesse, annot=False, cmap='coolwarm', linewidths=0.5, cbar=True)

    plt.title(f'Mapa de Correlação – {variavel_categorica} vs Variáveis Numéricas', fontsize=14, weight='bold')
    plt.tight_layout()
    
    plt.show()


# Executar para todas as variáveis categóricas
variaveis_categoricas = ['Categoria', 'Modelagem', 'Linha', 'Grupo MP', 'Marca']

for variavel in variaveis_categoricas:
    mapa_correlacao_categoria(variavel, dados_corr_encoded)

#%%
# APLICAÇÃO DO XGBoost

#%%
# Dados de treino e teste: vendas + produtos de 2022 e 2023
# Agrupando dados dos dois anos
vendas_prod_22_23 = pd.concat([vendas_prod22, vendas_prod23], ignore_index=True)

# Verificar a junção
print(vendas_prod_22_23.info())
print(vendas_prod_22_23.head())

#%%
# Preparar os dados de treino (2022 + 2023)

# Agrupar vendas para obter a quantidade total vendida por Referência
vendas_treino = vendas_prod_22_23.groupby('Referência').agg({
    'Quantidade Pedida Por Variante': 'sum'}).reset_index()

# Juntar com as características dos produtos de 2024
base_treino = vendas_treino.merge(produtos24.drop_duplicates(subset='Referência'), 
                                  on='Referência', how='left')

#%%
# Preparar os dados de previsão (produtos de 2024)

base_previsao = produtos24.drop_duplicates(subset='Referência')

#%%
# Definir variáveis preditoras (X) e variável alvo (y)

# Variáveis preditoras (features)
variaveis_categoricas = ['Categoria', 'Modelagem', 'Linha', 'Grupo MP']

# Aplicar One Hot Encoding
X = pd.get_dummies(base_treino[variaveis_categoricas + ['Referência']], drop_first=False)

#  Variável alvo
y = base_treino['Quantidade Pedida Por Variante']

# Aplicar Encoding na base de previsão
X_previsao = pd.get_dummies(base_previsao[variaveis_categoricas + ['Referência']], drop_first=False)
X_previsao = X_previsao.reindex(columns=X.columns, fill_value=0) #Alinhar colunas das bases

#%%
from xgboost.callback import EarlyStopping

#%%
# Treinar o modelo XGBoost

# Separar dados para treino e validação interna (80/20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Criar o modelo
modelo = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse')

# Treinar o modelo
modelo.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True)

#%%
# Avaliar o modelo na validação interna
y_pred_valid = modelo.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
r2 = r2_score(y_valid, y_pred_valid)

print(f'Desempenho na validação interna:')
print(f'→ RMSE: {rmse:.2f}')
print(f'→ R²: {r2:.2%}')

#%%
# Gerar previsão de vendas para os produtos de 2024
y_pred_2024 = modelo.predict(X_previsao)

# Montar tabela com os resultados
previsao_2024 = base_previsao[['Referência']].copy()
previsao_2024['Previsão Qtde Venda'] = y_pred_2024.round(0).astype(int)

print(previsao_2024)

