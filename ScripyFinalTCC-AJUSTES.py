# -*- coding: utf-8 -*-
"""Created on Sun Feb 15 13:53:14 2026
@author: Renata Alves"""

#%%
# Versão Final - Script considerando Anos de 2021 a 2024 para estudo e 2025 para criação da 
# previsão com o médoto de Machine Learning XGBoost e, no fim, comparação entre o obtido 
# pelo método escolhido, o previsto inicial pela empresa e o real obtido com as vendas.

#%% PARTE 1 - IMPORTAÇÃO E AJUSTE DOS DADOS INICIAIS

#%% Intalação de Pacotes

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

#%% Importação dos Pacotes 

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from matplotlib import cm
import plotly.graph_objects as go # gráficos 3D
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # estatísticas (médias e r²)
from sklearn.metrics import roc_auc_score, roc_curve # estatistica com curva roc
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import os
from datetime import datetime

#%% Seleção o diretório com arquivos

os.chdir(r'C:\Users\Renata Alves\TCC')

#%% Importação dos Dados

#2021
produtos2021 = pd.read_csv('produtos2021.csv', sep=';', encoding='latin1')
vendas2021 = pd.read_csv('vendas2021.csv', sep=';', encoding='latin1')

#2022
produtos2022 = pd.read_csv('produtos2022.csv', sep=';', encoding='latin1')
vendas2022 = pd.read_csv('vendas2022.csv', sep=';', encoding='latin1')

#2023
produtos2023 = pd.read_csv('produtos2023.csv', sep=';', encoding='latin1')
vendas2023 = pd.read_csv('vendas2023.csv', sep=';', encoding='latin1')

#2024
produtos2024 = pd.read_csv('produtos2024.csv', sep=';', encoding='latin1')
vendas2024 = pd.read_csv('vendas2024.csv', sep=';', encoding='latin1')

#2025
produtos2025 = pd.read_csv('produtos2025.csv', sep=';', encoding='latin1')
produtosEcor2025 = pd.read_csv('produtosEcor2025.csv', sep=';', encoding='latin1')

#%% Transformação dos tipos de variáveis

#Conversão dos tipos das variáveis de Vendas
vendas_str = ['Representante','Cliente','Pedido','Marca','Referencia','Cor']
vendas2021[vendas_str] = vendas2021[vendas_str].astype(str)
vendas2022[vendas_str] = vendas2022[vendas_str].astype(str)
vendas2023[vendas_str] = vendas2023[vendas_str].astype(str)
vendas2024[vendas_str] = vendas2024[vendas_str].astype(str)

#Conversão dos tipos de Variáveis de Produtos
produtos2021['Referencia'] = produtos2021['Referencia'].astype(str)
produtos2022['Referencia'] = produtos2022['Referencia'].astype(str)
produtos2023['Referencia'] = produtos2023['Referencia'].astype(str)
produtos2024['Referencia'] = produtos2024['Referencia'].astype(str)
produtos2025['Referencia'] = produtos2025['Referencia'].astype(str)
produtosEcor2025['Referencia'] = produtosEcor2025['Referencia'].astype(str)
produtosEcor2025['Cor'] = produtosEcor2025['Cor'].astype(str)
produtosEcor2025['Marca'] = produtosEcor2025['Marca'].astype(str)

#%% PARTE 2 - ANÁLISE EXPLORATÓRIA DOS DADOS

#%% Simplificação do Banco de Dados - unificando tabelas

#Ano nas vendas
vendas2021["Ano"] = 2021
vendas2022["Ano"] = 2022
vendas2023["Ano"] = 2023
vendas2024["Ano"] = 2024

#Ano nos produtos
produtos2021["Ano"] = 2021
produtos2022["Ano"] = 2022
produtos2023["Ano"] = 2023
produtos2024["Ano"] = 2024
produtos2025["Ano"] = 2025

#Junção de todos os anos 
vendas_total = pd.concat([vendas2021, vendas2022, vendas2023, vendas2024], ignore_index=True)
produtos_total = pd.concat([produtos2021, produtos2022, produtos2023, produtos2024, produtos2025], ignore_index=True)

#Inclusão da região do estado nos pedidos
mapa_regioes = {
    "GO":"Centro-Oeste","DF":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
    "SP":"Sudeste","RJ":"Sudeste","MG":"Sudeste","ES":"Sudeste",
    "PR":"Sul","SC":"Sul","RS":"Sul",
    "BA":"Nordeste","PE":"Nordeste","CE":"Nordeste","RN":"Nordeste","PB":"Nordeste","AL":"Nordeste","SE":"Nordeste","MA":"Nordeste","PI":"Nordeste",
    "AM":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","AC":"Norte","AP":"Norte","TO":"Norte"}

vendas_total["Regiao"] = vendas_total["Estado"].map(mapa_regioes).fillna("Não identificado")

# Padronizar referência nas vendas
vendas_total["Referencia"] = (vendas_total["Referencia"].astype(str).str.strip())

# Padronizar referência nos produtos
produtos_total["Referencia"] = (produtos_total["Referencia"].astype(str).str.strip())

#Junção de produtos em vendas
vendas_completo = vendas_total.merge(
    produtos_total[["Ano", "Referencia", "Categoria", "Modelagem", "Grupo MP", "Grade"]],
    on=["Ano", "Referencia"],
    how="left")
vendas_completo['Ano'] = vendas_completo['Ano'].astype(str)

#Construção da tabela em ordem dos anos
ordem_anos = ["2021", "2022", "2023", "2024", "2025"]

vendas_completo["Ano"] = pd.Categorical(
    vendas_completo["Ano"].astype(str),
    categories=ordem_anos,
    ordered=True)
vendas_completo = vendas_completo.sort_values("Ano")

#%% Padronização da configuração dos gráficos e tabelas 

plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "normal", # Legendas dos eixos NÃO grafadas em negrito
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.grid": False,           # Sem linhas de grade
    "axes.facecolor": "none",     # Sem preenchimento nos gráficos
    "figure.facecolor": "none",
    "axes.linewidth": 1.5,        # Linhas dos eixos na largura 1,5 pt
    "axes.edgecolor": "black",    # Linhas dos eixos na cor preta
    "axes.spines.top": False,     # Sem borda superior
    "axes.spines.right": False,   # Sem borda direita
})

OUT_DIR = "Figuras_TCCDSA_RenataACasqueiro"
os.makedirs(OUT_DIR, exist_ok=True)

def azul_palette(n):
    cmap = cm.get_cmap("Blues")
    vals = np.linspace(0.35, 0.85, n)
    return [cmap(v) for v in vals]

AZUL_ESCURO = cm.get_cmap("Blues")(0.80)
AZUL_MEDIO  = cm.get_cmap("Blues")(0.60)
AZUL_CLARO  = cm.get_cmap("Blues")(0.40)

def format_milhar_br(x):
    try:
        return f"{int(round(x)):,}".replace(",", ".")
    except:
        return str(x)

def salvar_fig(nome_arquivo):
    plt.savefig(os.path.join(OUT_DIR, nome_arquivo), dpi=300, bbox_inches="tight")

def add_labels_barras(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, format_milhar_br(v), ha="center", va="bottom", fontsize=9)

#%% Análise do volume de vendas (em peças) dos anos de 2021 a 2024 -> Volume Total de Vendas (Peças) de Alto Verão por Ano (2021 a 2024)
volume_ano = (vendas_completo.groupby("Ano")["Quantidade Faturada"].sum())
volume_ano = volume_ano[volume_ano > 0]

fig, ax = plt.subplots(figsize=(8,5))
cores = azul_palette(len(volume_ano))
ax.bar(volume_ano.index, volume_ano.values, color=cores)

ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade Faturada (Peças)")
add_labels_barras(ax, volume_ano.values)

plt.tight_layout()
salvar_fig("01_volume_vendas_por_ano.png")
plt.show()

#%% Análise da quantidade de referências disponíveis nos anos 2021 a 2025 -> Quantidade de Referências em Alto Verão por Ano (2021 a 2025)
refs_ano = produtos_total.groupby("Ano")["Referencia"].nunique().sort_index()

fig, ax = plt.subplots(figsize=(8,5))
cores = azul_palette(len(refs_ano))
ax.bar(refs_ano.index.astype(str), refs_ano.values, color=cores)

ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade de Referências")
add_labels_barras(ax, refs_ano.values)

plt.tight_layout()
salvar_fig("02_qtd_referencias_por_ano.png")
plt.show()

#%% Análise do volume de vendas por marca nos anos de 2021 a 2024

vendas_marca = (vendas_completo.groupby(["Ano","Marca"])["Quantidade Faturada"].sum().reset_index())
pivot_marca = vendas_marca.pivot(index="Ano",columns="Marca",values="Quantidade Faturada").fillna(0)
pivot_marca = pivot_marca.loc[pivot_marca.index != "2025"]

fig, ax = plt.subplots(figsize=(9,5))
cores = azul_palette(pivot_marca.shape[1])
pivot_marca.plot(kind="bar", ax=ax, color=cores)

ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
ax.tick_params(axis='x', rotation=0)

for container in ax.containers:
    for bar in container:
    altura = bar.get_height()
        if altura > 0:
            ax.text(bar.get_x() + bar.get_width()/2, altura, f'{int(altura):,}'.replace(',', '.'),
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
salvar_fig("03_vendas_por_marca.png")
plt.show()

#%% Análise da quantidade de representantes ativos nos anos de 2021 a 2024 -> Quantidade de Representantes Ativos em Alto Verão por Ano (2021 a 2024)

rep_ativos = vendas_completo.groupby("Ano")["Representante"].nunique().sort_index()
rep_ativos = rep_ativos[rep_ativos > 0]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(rep_ativos.index.astype(str), rep_ativos.values, color=azul_palette(len(rep_ativos)))

ax.set_xlabel("Ano")
ax.set_ylabel("Representantes Ativos")
add_labels_barras(ax, rep_ativos.values)

plt.tight_layout()
salvar_fig("04_representantes_ativos.png")
plt.show()

#%% Análise da quantidade de clientes atendidos nos anos de 2021 a 2024 -> Quantidade de Clientes Atendidos em Alto Verão por Ano (2021 a 2024)

clientes_ano = vendas_completo.groupby("Ano")["Cliente"].nunique().sort_index()
clientes_ano = clientes_ano[clientes_ano > 0]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(clientes_ano.index.astype(str), clientes_ano.values, color=azul_palette(len(clientes_ano)))

ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade de Clientes Atendidos")
add_labels_barras(ax, clientes_ano.values)

plt.tight_layout()
salvar_fig("05_clientes_por_ano.png")
plt.show()

#%% Análise do ticket médio (em peças) por cliente por ano entre 2021 e 2024 -> Ticket Médio de Peças por Cliente em Alto Verão por Ano (2021 a 2024)

ticket_medio = (vendas_completo.groupby("Ano").apply(lambda x: x["Quantidade Faturada"].sum() / max(x["Cliente"].nunique(), 1)).sort_index())
ticket_medio = ticket_medio[ticket_medio > 0]

fig, ax = plt.subplots(figsize=(8,5))
anos = ticket_medio.index.astype(int)
ax.plot(ticket_medio.index.astype(int), ticket_medio.values, marker="o", color=AZUL_ESCURO, linewidth=2)

ax.set_xticks(anos)
ax.set_xticklabels(anos)
ax.set_xlabel("Ano")
ax.set_ylabel("Peças por Cliente")

for x, y in zip(ticket_medio.index.astype(int), ticket_medio.values):
    ax.text(x, y, format_milhar_br(y), ha="right", va="bottom", fontsize=9)

plt.tight_layout()
salvar_fig("06_ticket_medio_por_cliente.png")
plt.show()

#%% Análise do volume de vendas (em peças) por categoria nos anos de 2021 a 2024 -> Volume de Vendas de Alto Verão por Ano – Painel por Categoria (2021 a 2024)

df = vendas_completo.copy()
df["Categoria"] = df["Categoria"].astype(str).str.strip()
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
anos = [2021, 2022, 2023, 2024]
df = df[df["Ano_num"].isin(anos)]

vendas_cat = (df.groupby(["Categoria", "Ano_num"])["Quantidade Faturada"].sum().reset_index())
pivot_cat = (vendas_cat.pivot(index="Categoria", columns="Ano_num", values="Quantidade Faturada").fillna(0))
pivot_cat = pivot_cat.reindex(columns=anos, fill_value=0)

pivot_cat["Total_Periodo"] = pivot_cat.sum(axis=1)
pivot_cat = pivot_cat.sort_values(by="Total_Periodo", ascending=False)
pivot_cat = pivot_cat.drop(columns=["Total_Periodo"])

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
axes = axes.ravel()
x = np.arange(len(anos))
cores = azul_palette(len(anos))

for i, categoria in enumerate(pivot_cat.index[:9]):
    ax = axes[i]
    valores = pivot_cat.loc[categoria, anos].values

    ax.bar(x, valores, color=cores, width=0.7)
    ax.set_title(categoria, fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos], fontsize=9)

for j in range(i + 1, 9):
    axes[j].axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar_fig("07_painel_vendas_por_categoria_3x3.png")
plt.show()

#%% Análise da quantidade de referências por categoria nos anos de 2021 a 2025 (tabela e painel de gráficos)  -> Quantidade de Referências de Alto Verão por Ano por Categoria (2021 a 2025)

ref_cat = (produtos_total.groupby(["Ano","Categoria"])["Referencia"].nunique().reset_index(name="Qtd_Ref"))
tabela_ref_cat = ref_cat.pivot(index="Categoria", columns="Ano", values="Qtd_Ref").fillna(0).astype(int)
anos = [2021, 2022, 2023, 2024, 2025]
anos_existentes = [a for a in anos if a in tabela_ref_cat.columns]
df_cat = tabela_ref_cat[anos_existentes].copy()

df_cat["Total_Periodo"] = df_cat.sum(axis=1)
df_cat = df_cat.sort_values(by="Total_Periodo", ascending=False)
df_cat = df_cat.drop(columns=["Total_Periodo"])

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
axes = axes.ravel()
x = np.arange(len(anos_existentes))
for i, categoria in enumerate(df_cat.index):
    ax = axes[i]
    valores = df_cat.loc[categoria].values

    ax.bar(x, valores, color=cores, width=0.7)
    ax.set_title(str(categoria), fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(anos_existentes, fontsize=9)

    for xi, v in zip(x, valores):
        ax.text(xi, v, f"{int(v)}", ha="center", va="bottom", fontsize=11)
        
for j in range(i + 1, 9):
    axes[j].axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar_fig("08_painel_referencias_por_categoria.png")
plt.show()

#%% Análise do Volume de Vendas de Alto Verão por Modelagem no Ano - Painel por top 5 modelagens (2021 a 2024) -> Volume de Vendas de Alto Verão por Modelagem | Top 5 por Ano

dados_vendas = vendas_completo.copy()
dados_vendas["Ano_num"] = dados_vendas["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
anos_validos = [2021, 2022, 2023, 2024]
dados_vendas = dados_vendas[dados_vendas["Ano_num"].isin(anos_validos)].copy()

vendas_modelagem_ano = (dados_vendas.groupby(["Ano_num","Modelagem"])["Quantidade Faturada"].sum().reset_index())
fig, axes = plt.subplots(2,2, figsize=(13,8))
axes = axes.ravel()

for i, ano in enumerate(anos_validos):
    ax = axes[i]
    top_modelagens = (vendas_modelagem_ano[vendas_modelagem_ano["Ano_num"] == ano].sort_values("Quantidade Faturada", ascending=False)
        .head(5)
        .sort_values("Quantidade Faturada", ascending=True))
    
    ax.barh(top_modelagens["Modelagem"], top_modelagens["Quantidade Faturada"], color=azul_palette(5)[-2])
    ax.set_title(f"{ano}", weight="bold")
    ax.set_xlabel("Quantidade Faturada")

plt.tight_layout(rect=[0,0,1,0.95])
salvar_fig("09_top5_modelagem_painel.png")
plt.show()

#%% Análise do Volume de Vendas de Alto Verão por Grupo de Matéria Prima no Ano - Painel por top 5 grupos (2021 a 2024)

vendas_gp_ano = (dados_vendas.groupby(["Ano_num","Grupo MP"])["Quantidade Faturada"].sum().reset_index())

fig, axes = plt.subplots(2,2, figsize=(13,8))
axes = axes.ravel()

for i, ano in enumerate(anos_validos):
    ax = axes[i]
    top_gp = (vendas_gp_ano[vendas_gp_ano["Ano_num"] == ano].sort_values("Quantidade Faturada", ascending=False)
        .head(5)
        .sort_values("Quantidade Faturada", ascending=True))

    ax.barh(top_gp["Grupo MP"], top_gp["Quantidade Faturada"], color=azul_palette(5)[-1])
    ax.set_title(f"{ano}", weight="bold")
    ax.set_xlabel("Quantidade Faturada")
        
plt.tight_layout(rect=[0,0,1,0.95])
salvar_fig("10_top5_grupoMP_painel.png")
plt.show()

#%% Análise do volume de vendas por grade nos anos de 2021 a 2024 -> Volume de Vendas de Alto Verão por Grade e Ano (2021 a 2024)

dados_vendas = vendas_completo.copy()
dados_vendas["Ano_num"] = dados_vendas["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
anos_validos = [2021, 2022, 2023, 2024]
dados_vendas = dados_vendas[dados_vendas["Ano_num"].isin(anos_validos)].copy()

for coluna in ["Grade", "Modelagem", "Grupo MP"]:
    if coluna in dados_vendas.columns:
        dados_vendas[coluna] = dados_vendas[coluna].astype(str).str.strip()
    
vendas_grade_ano = (dados_vendas.groupby(["Ano_num", "Grade"])["Quantidade Faturada"].sum().reset_index())

tabela_grade_ano = (vendas_grade_ano.pivot(index="Grade", columns="Ano_num", values="Quantidade Faturada").fillna(0).reindex(columns=anos_validos, fill_value=0))
tabela_grade_ano = tabela_grade_ano.loc[tabela_grade_ano.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(tabela_grade_ano.index))
width = 0.18
cores = azul_palette(len(anos_validos))

for i, ano in enumerate(anos_validos):
    ax.bar(x + i*width, tabela_grade_ano[ano].values, width=width, color=cores[i], label=str(ano))

ax.set_xlabel("Grade")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(tabela_grade_ano.index, ha="center")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

plt.tight_layout()
salvar_fig("11_vendas_por_grade.png")
plt.show()

#%% Análise do comportamento de vendas por região do país nos anos de 2021 a 2024 -> Volume de Vendas de Alto Verão por Região por Ano (2021 a 2024)

vendas_reg = (vendas_completo.groupby(["Ano","Regiao"])["Quantidade Faturada"].sum().reset_index())
pivot_reg = vendas_reg.pivot(index="Regiao", columns="Ano", values="Quantidade Faturada").fillna(0)
pivot_reg.columns = pivot_reg.columns.astype(int)
pivot_reg = pivot_reg.loc[pivot_reg.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10,5))
anos = [2021, 2022, 2023, 2024]
x = np.arange(len(pivot_reg.index))
width = 0.18
cores = azul_palette(len(anos))

for i, ano in enumerate(anos):
    ax.bar(x + i*width, pivot_reg[ano].values, width=width, color=cores[i], label=str(ano))

ax.set_xlabel("Região")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(pivot_reg.index, ha="center")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

plt.tight_layout()
salvar_fig("12_vendas_por_regiao.png")
plt.show()

#%% Análise da quantidade de clientes atendidos por região nos anos de 2021 a 2024 -> Clientes Atendidos em Alto Verão por Ano – Painel por Região (2021 a 2024)
anos = [2021, 2022, 2023, 2024]
df = vendas_completo.copy()
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

clientes_reg = (df[df["Ano_num"].isin(anos)].groupby(["Regiao", "Ano_num"])["Cliente"].nunique().reset_index(name="Qtd_Clientes"))
pivot_cli_reg = (clientes_reg.pivot(index="Regiao", columns="Ano_num", values="Qtd_Clientes").fillna(0).reindex(columns=anos, fill_value=0))
pivot_cli_reg = pivot_cli_reg.loc[pivot_cli_reg.sum(axis=1).sort_values(ascending=False).index]
regioes = pivot_cli_reg.index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.ravel()
cores = azul_palette(len(anos))
x = np.arange(len(anos))

for i, regiao in enumerate(regioes):
    ax = axes[i]
    valores = pivot_cli_reg.loc[regiao, anos].values
    ax.bar(x, valores, color=cores, width=0.7)
    ax.set_title(regiao, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos], fontsize=11)
            
axes[5].axis("off")
fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar_fig("13_painel_clientes_por_regiao.png")
plt.show()

#%% Análise do % de Participação do volume de vendas das Categorias por região a cada ano -> Heatmap da Participação % das Categorias nas Vendas por Região (2021 a 2024 juntos)

df = vendas_completo.copy()
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
anos = [2021, 2022, 2023, 2024]
df = df[df["Ano_num"].isin(anos)].copy()

df["Regiao"] = df["Regiao"].astype(str).str.strip().str.upper()
df["Categoria"] = df["Categoria"].astype(str).str.strip().str.upper()

ordem_categorias = sorted(df["Categoria"].dropna().unique())
ordem_regioes = ["CENTRO-OESTE", "NORDESTE", "NORTE", "SUDESTE", "SUL"]

total_regiao_ano = df.groupby(["Ano_num", "Regiao"])["Quantidade Faturada"].sum()
vendas_cat_regiao = df.groupby(["Categoria", "Ano_num", "Regiao"])["Quantidade Faturada"].sum()
top_categorias = df.groupby("Categoria")["Quantidade Faturada"].sum().nlargest(9).index.tolist()

participacao = (vendas_cat_regiao / total_regiao_ano * 100).fillna(0).reset_index(name="Part_Perc")
vmax_global = float(np.ceil(participacao["Part_Perc"].max()))

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True) 
axes = axes.flatten()

for i, categoria in enumerate(top_categorias):
    ax = axes[i]
    df_cat = participacao[participacao["Categoria"] == categoria]
    pivot = df_cat.pivot(index="Regiao", columns="Ano_num", values="Part_Perc").fillna(0)
    pivot = pivot.reindex(index=ordem_regioes, columns=anos, fill_value=0)
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=vmax_global, aspect="auto")
    
    ax.set_title(categoria, weight="bold", fontsize=12)
    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels(anos, fontsize=10)
    ax.set_yticks(range(len(ordem_regioes)))
    if i % 3 == 0:
        ax.set_yticklabels(ordem_regioes, fontsize=10)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            valor = pivot.iloc[row, col]
            cor_texto = "white" if valor > (vmax_global/2) else "black"
            if valor > 0:
                ax.text(col, row, f"{valor:.1f}%", ha="center", va="center", fontsize=11, color=cor_texto)
            else:
                ax.text(col, row, "-", ha="center", va="center", fontsize=11, color="black")

for j in range(len(top_categorias), 9):
    axes[j].axis("off")

cbar = fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, pad=0.25)
cbar.outline.set_visible(False)

plt.subplots_adjust(hspace=0.35, wspace=0.05, top=0.92, bottom=0.28, left=0.08, right=0.95)

salvar_fig("14_heatmap_evolucao_categoria_painel.png")
plt.show()

#%% PARTE 3 - RODANDO O XGBOOST

#%% Padronização e União dos Dados

pd.set_option("display.max_columns", 200) # Permite visualizar mais colunas

def padronizar_texto(df, cols): # remover espaços em branco e deixar tudo em maiúsculo
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df

for dfp in [produtos2021, produtos2022, produtos2023, produtos2024, produtos2025]:
    padronizar_texto(dfp, ["Referencia", "Grade", "Categoria", "Modelagem", "Grupo MP", "Marca"]) # padronização nas bases de Produtos
    if "Ano" in dfp.columns:
        dfp["Ano"] = dfp["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
        
for dfv in [vendas2021, vendas2022, vendas2023, vendas2024]:
    padronizar_texto(dfv, ["Representante", "Cliente", "Estado", "Pedido", "Marca", "Referencia", "Cor"]) # padronização nas bases de Vendas
    dfv["Ano"] = dfv["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    dfv["Quantidade Faturada"] = pd.to_numeric(dfv["Quantidade Faturada"], errors="coerce").fillna(0)

padronizar_texto(produtosEcor2025, ["Referencia", "Cor", "Grade", "Categoria", "Modelagem", "Grupo MP", "Marca"]) # padronização da base de previsão de 2025

def add_regiao(df_vendas):
    if "Regiao" not in df_vendas.columns and "Estado" in df_vendas.columns: # adicionar a Região baseada no Estado
        df_vendas["Regiao"] = df_vendas["Estado"].map(mapa_regioes).fillna("DESCONHECIDA")
    return df_vendas

for dfv in [vendas2021, vendas2022, vendas2023, vendas2024]:
    add_regiao(dfv)

# União dos dados de vendas e produtos de todos os anos em um dataframde só
vendas_hist = pd.concat([vendas2021, vendas2022, vendas2023, vendas2024], ignore_index=True)
produtos_hist = pd.concat([produtos2021, produtos2022, produtos2023, produtos2024], ignore_index=True)

#Inclusão da Marca na tabela de Produtos
produtos_hist["Marca"] = produtos_hist["Referencia"].astype(str).str[2]
produtosEcor2025["Marca"] = produtosEcor2025["Referencia"].astype(str).str[2]

# Criação da variável "Variante" (Referência_Cor)
vendas_hist["Variante"] = vendas_hist["Referencia"].astype(str) + "_" + vendas_hist["Cor"].astype(str)

# Unificação da tabelo de Vendas com os Produtos para trazer todas as características 
vendas_prod = vendas_hist.merge(
    produtos_hist[["Ano","Referencia","Grade","Categoria","Modelagem","Grupo MP","Marca"]],
    on=["Ano","Referencia"],
    how="left",
    suffixes=('_venda', '_produto'))

vendas_prod["Marca"] = vendas_prod["Marca_produto"].fillna(vendas_prod["Marca_venda"])
vendas_prod = vendas_prod.drop(columns=["Marca_venda", "Marca_produto"])

#%% Criação de um Base Macro (a nível de Referencia) para modelagem hierárquica

# Criação da 'base_macro': Vendas por Referência por Ano para modelo aprender a prever esse volume total
base_macro = (
    vendas_prod.groupby(["Ano", "Referencia", "Marca", "Categoria", "Grupo MP", "Modelagem", "Grade"])
    .agg(volume_total_ref=("Quantidade Faturada", "sum"))
    .reset_index())

# Criação da 'lag1_volume': que é o volume que a Referência vendeu no ano anterior
base_macro = base_macro.sort_values(["Referencia", "Ano"]).reset_index(drop=True)
base_macro["lag1_volume"] = base_macro.groupby("Referencia")["volume_total_ref"].shift(1)

# Criação 'Tipo_Referencia': que é classificação entre "Recorrente" (tem histórico no ano anterior) ou "Novo" (lançamento, sem histórico).
base_macro["Tipo_Referencia"] = np.where(base_macro["lag1_volume"].notna(), "Recorrente", "Novo")

#%% Criação da Lógica de Cold Start para crontruir um fallback para os produtos novos

# Base filtrada para olhar apenas o comportamento do que foi "Novo" em cada ano
base_apenas_novos = base_macro[base_macro["Tipo_Referencia"] == "Novo"].copy()

# Criação de tabelas de apoio que calculam as médias de venda do ano passado agrupadas em diferentes níveis de detalhe 
# Nível 1 (Mais específico): Marca + Categoria + Grupo MP + Modelagem
media_lvl1 = base_apenas_novos.groupby(["Ano", "Marca", "Categoria", "Grupo MP", "Modelagem"])["volume_total_ref"].mean().reset_index(name="media_lvl1")
media_lvl1["Ano_Aplicacao"] = media_lvl1["Ano"] + 1
# Nível 2: Marca + Categoria + Grupo MP
media_lvl2 = base_apenas_novos.groupby(["Ano", "Marca", "Categoria", "Grupo MP"])["volume_total_ref"].mean().reset_index(name="media_lvl2")
media_lvl2["Ano_Aplicacao"] = media_lvl2["Ano"] + 1
# Nível 3: Marca + Categoria
media_lvl3 = base_apenas_novos.groupby(["Ano", "Marca", "Categoria"])["volume_total_ref"].mean().reset_index(name="media_lvl3")
media_lvl3["Ano_Aplicacao"] = media_lvl3["Ano"] + 1
# Nível 4 (Mais genérico - Garantia): Marca
media_lvl4 = base_apenas_novos.groupby(["Ano", "Marca"])["volume_total_ref"].mean().reset_index(name="media_lvl4")
media_lvl4["Ano_Aplicacao"] = media_lvl4["Ano"] + 1

# Trazendo as médias para a 'base_macro' (para o treino do modelo)
base_macro = base_macro.merge(media_lvl1.drop(columns="Ano"), left_on=["Ano", "Marca", "Categoria", "Grupo MP", "Modelagem"], right_on=["Ano_Aplicacao", "Marca", "Categoria", "Grupo MP", "Modelagem"], how="left").drop(columns="Ano_Aplicacao")
base_macro = base_macro.merge(media_lvl2.drop(columns="Ano"), left_on=["Ano", "Marca", "Categoria", "Grupo MP"], right_on=["Ano_Aplicacao", "Marca", "Categoria", "Grupo MP"], how="left").drop(columns="Ano_Aplicacao")
base_macro = base_macro.merge(media_lvl3.drop(columns="Ano"), left_on=["Ano", "Marca", "Categoria"], right_on=["Ano_Aplicacao", "Marca", "Categoria"], how="left").drop(columns="Ano_Aplicacao")
base_macro = base_macro.merge(media_lvl4.drop(columns="Ano"), left_on=["Ano", "Marca"], right_on=["Ano_Aplicacao", "Marca"], how="left").drop(columns="Ano_Aplicacao")

# Criação da variável 'estimativa_inicial': que é o histórico da venda real anterior (lag1) ou a média lvl1, 2, 3 ou 4 (a que tiver informação)
base_macro["estimativa_inicial"] = base_macro["lag1_volume"]
base_macro["estimativa_inicial"] = base_macro["estimativa_inicial"].fillna(base_macro["media_lvl1"])
base_macro["estimativa_inicial"] = base_macro["estimativa_inicial"].fillna(base_macro["media_lvl2"])
base_macro["estimativa_inicial"] = base_macro["estimativa_inicial"].fillna(base_macro["media_lvl3"])
base_macro["estimativa_inicial"] = base_macro["estimativa_inicial"].fillna(base_macro["media_lvl4"])
base_macro["estimativa_inicial"] = base_macro["estimativa_inicial"].fillna(base_macro["volume_total_ref"].median())

#%% Validação Cruzada Temporal (Time Series Split)

# Criação das variáveis 'features_cat' e 'features_num': para separar infos string e int
features_cat = ["Marca", "Categoria", "Grupo MP", "Modelagem"]
features_num = ["estimativa_inicial"]

# Criação da variável 'base_modelo_completa': One Hot Enconding com as categoricas (0 ou 1) para rodar XGboost
base_modelo_completa = pd.get_dummies(base_macro, columns=features_cat, drop_first=False)

# Criação da variável 'cols_treino': para considerar tds as caracteristicas (pós dummizar) como X (features)
cols_treino = [col for col in base_modelo_completa.columns if col.startswith(tuple(features_cat))] + features_num
print("\n--- INICIANDO VALIDAÇÃO CRUZADA TEMPORAL ---")

# Criação da variável 'ondas': para modelo treinar um ano com informação do anterior
ondas = [
    {"treino": [2021], "teste": 2022},
    {"treino": [2021, 2022], "teste": 2023},
    {"treino": [2021, 2022, 2023], "teste": 2024}]

# Criação da variável 'resultados_cv': para calcular RMSE de cada onde e termos a média no final
resultados_cv = []
for onda in ondas:
    df_train_onda = base_modelo_completa[base_modelo_completa["Ano"].isin(onda["treino"])]  # filtram os anos corretos da onda atual
    df_test_onda = base_modelo_completa[base_modelo_completa["Ano"] == onda["teste"]]
    
    X_train_onda = df_train_onda[cols_treino] #treino
    y_train_onda = df_train_onda["volume_total_ref"]
    
    X_test_onda = df_test_onda[cols_treino] #teste
    y_test_onda = df_test_onda["volume_total_ref"]

# Criação da variável 'modelo_cv': é o modelo que vai fazer as ondas de treino do XGboost
    modelo_cv = xgb.XGBRegressor( 
        objective="reg:squarederror",
        n_estimators=300,      
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)

    modelo_cv.fit(X_train_onda, y_train_onda)
    
    pred_onda = modelo_cv.predict(X_test_onda) # previsão feita pelo modelo naquela onda
    rmse_onda = np.sqrt(mean_squared_error(y_test_onda, pred_onda))
    r2_onda = r2_score(y_test_onda, pred_onda)
    
    print(f"Onda Teste {onda['teste']} -> RMSE: {rmse_onda:.2f} | R²: {r2_onda:.2%}")
    resultados_cv.append(rmse_onda)

print(f"-> RMSE Médio da Validação Cruzada: {np.mean(resultados_cv):.2f}")

#%% Treinamento Final do Modelo (Com todo o histórico)

# Criação da variável 'df_train_final': todos os anos juntos para treinar o modelo final
df_train_final = base_modelo_completa[base_modelo_completa["Ano"].isin([2021, 2022, 2023, 2024])]
X_train_final = df_train_final[cols_treino]
y_train_final = df_train_final["volume_total_ref"]

# Criação da variável 'modelo_final': é o modelo que vai gerar a previsão de 2025 -> Usa mais árvores (n_estimators=500)
modelo_final = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,     
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

modelo_final.fit(X_train_final, y_train_final)
print("\nModelo Final XGBoost treinado com sucesso com dados de 2021 a 2024!")

#%% Geração da Previsão de Vendas de 2025 com Modelo Hierárquico (infos de Macro/Referencia para Micro/Variante)

print("\n--- INICIANDO PREVISÃO 2025 ---")

# Criação da variável 'base_prev_macro': é a base com infos das refs de 2025 e suas caracteristicas
base_prev_macro = produtosEcor2025[["Referencia", "Marca", "Categoria", "Grupo MP", "Modelagem"]].drop_duplicates().copy()
base_prev_macro["Ano"] = 2025

# Criação da variável 'historico_2024': são as infos de 2024 para servir de lag para 2025
historico_2024 = base_macro[base_macro["Ano"] == 2024][["Referencia", "volume_total_ref"]]

# Trazendo o lag1 (de 2024) para as referências de 2025
base_prev_macro = base_prev_macro.merge(historico_2024, on="Referencia", how="left")
base_prev_macro = base_prev_macro.rename(columns={"volume_total_ref": "lag1_volume"})

# Classificando como Recorrente (tinha em 2024) ou Novo (lançamento em 2025)
base_prev_macro["Tipo_Referencia"] = np.where(base_prev_macro["lag1_volume"].notna(), "Recorrente", "Novo")

# Aplicação da lógica Cold Start para AV 25 -> Usando as médias dos LANÇAMENTOS de 2024
media_lvl1_2024 = media_lvl1[media_lvl1["Ano"] == 2024].copy()
media_lvl2_2024 = media_lvl2[media_lvl2["Ano"] == 2024].copy()
media_lvl3_2024 = media_lvl3[media_lvl3["Ano"] == 2024].copy()
media_lvl4_2024 = media_lvl4[media_lvl4["Ano"] == 2024].copy()

base_prev_macro = base_prev_macro.merge(media_lvl1_2024.drop(columns=["Ano", "Ano_Aplicacao"]), on=["Marca", "Categoria", "Grupo MP", "Modelagem"], how="left")
base_prev_macro = base_prev_macro.merge(media_lvl2_2024.drop(columns=["Ano", "Ano_Aplicacao"]), on=["Marca", "Categoria", "Grupo MP"], how="left")
base_prev_macro = base_prev_macro.merge(media_lvl3_2024.drop(columns=["Ano", "Ano_Aplicacao"]), on=["Marca", "Categoria"], how="left")
base_prev_macro = base_prev_macro.merge(media_lvl4_2024.drop(columns=["Ano", "Ano_Aplicacao"]), on=["Marca"], how="left")

# Preenchendo os "buracos" para os produtos de 2025 
base_prev_macro["estimativa_inicial"] = base_prev_macro["lag1_volume"]
base_prev_macro["estimativa_inicial"] = base_prev_macro["estimativa_inicial"].fillna(base_prev_macro["media_lvl1"])
base_prev_macro["estimativa_inicial"] = base_prev_macro["estimativa_inicial"].fillna(base_prev_macro["media_lvl2"])
base_prev_macro["estimativa_inicial"] = base_prev_macro["estimativa_inicial"].fillna(base_prev_macro["media_lvl3"])
base_prev_macro["estimativa_inicial"] = base_prev_macro["estimativa_inicial"].fillna(base_prev_macro["media_lvl4"])
base_prev_macro["estimativa_inicial"] = base_prev_macro["estimativa_inicial"].fillna(base_macro["volume_total_ref"].median())

# Aplicando One Hot Enconding nas infos para o modelo (Dummificando)
X_prev_raw = base_prev_macro[features_cat + features_num]
X_prev_dummies = pd.get_dummies(X_prev_raw, columns=features_cat, drop_first=False)
X_prev = X_prev_dummies.reindex(columns=cols_treino, fill_value=0) # Garante que tem as mesmas colunas do treino

# 1º Modelo prevê volume por Referência
base_prev_macro["Previsao_Referencia"] = modelo_final.predict(X_prev)
base_prev_macro["Previsao_Referencia"] = base_prev_macro["Previsao_Referencia"].clip(lower=0) # Não permite previsão negativa

# 2º Modelo prevê volume por variante (rateio inteligente do volume da ref nas cores)
# Definindo peso de cada cor dentro de cada Categoria
peso_historico_cor = vendas_prod.groupby(["Categoria", "Cor"])["Quantidade Faturada"].sum().reset_index(name="Vol_Cor")
vol_categoria = vendas_prod.groupby("Categoria")["Quantidade Faturada"].sum().reset_index(name="Vol_Cat")
peso_historico_cor = peso_historico_cor.merge(vol_categoria, on="Categoria")
peso_historico_cor["Peso_Cor"] = peso_historico_cor["Vol_Cor"] / peso_historico_cor["Vol_Cat"]

base_variantes_2025 = produtosEcor2025.copy() # Levando as previsões macro e os pesos para a base de Variantes de 2025
base_variantes_2025["Variante"] = base_variantes_2025["Referencia"].astype(str) + "_" + base_variantes_2025["Cor"].astype(str)
base_variantes_2025 = base_variantes_2025.merge(base_prev_macro[["Referencia", "Previsao_Referencia"]], on="Referencia", how="left")
base_variantes_2025 = base_variantes_2025.merge(peso_historico_cor[["Categoria", "Cor", "Peso_Cor"]], on=["Categoria", "Cor"], how="left")
base_variantes_2025["Peso_Cor"] = base_variantes_2025["Peso_Cor"].fillna(0.01) # Se for uma cor super nova sem peso, considerar peso minúsculo de fallback

# Normalizando os pesos (A soma dos pesos das cores de uma referência tem que dar 100%)
soma_pesos_ref = base_variantes_2025.groupby("Referencia")["Peso_Cor"].sum().reset_index(name="Soma_Pesos")
base_variantes_2025 = base_variantes_2025.merge(soma_pesos_ref, on="Referencia")
base_variantes_2025["Peso_Final"] = base_variantes_2025["Peso_Cor"] / base_variantes_2025["Soma_Pesos"]

# 3º Calculo final da Previsão por Variante de AV 25
base_variantes_2025["Previsao_XGBoost"] = np.round(base_variantes_2025["Previsao_Referencia"] * base_variantes_2025["Peso_Final"], 0).astype(int)

previsao_final_2025 = base_variantes_2025[["Variante", "Referencia", "Cor", "Marca", "Categoria", "Modelagem", "Previsao_XGBoost"]].copy()
previsao_final_2025 = previsao_final_2025.sort_values("Previsao_XGBoost", ascending=False)

previsao_final_2025.to_csv(os.path.join(OUT_DIR, "previsao_AV_2025_por_variante.csv"), sep=";", index=False, encoding="utf-8-sig")
print("\nPrevisão de 2025 gerada e exportada com sucesso!")


#%% PARTE 4 - COMPARAÇÃO ENTRE PREVISÕES E VENDAS REAIS

#%% Avaliação e Comparações com o Resultado Obtido

print("\n--- INICIANDO COMPARAÇÕES COM DADOS REAIS DE 2025 ---")

vendas2025 = pd.read_csv('vendas2025.csv', sep=';', encoding='latin1')
previsaoinicial2025 = pd.read_csv('previsaoinicial2025.csv', sep=';', encoding='latin1')

padronizar_texto(vendas2025, ["Representante", "Cliente", "Estado", "Pedido", "Marca", "Referencia", "Cor"])
vendas2025["Quantidade Faturada"] = pd.to_numeric(vendas2025["Quantidade Faturada"], errors="coerce").fillna(0)
vendas2025["Variante"] = vendas2025["Referencia"] + "_" + vendas2025["Cor"]

padronizar_texto(previsaoinicial2025, ["Referencia", "Cor"])
previsaoinicial2025["Previsao Inicial"] = pd.to_numeric(previsaoinicial2025["Previsao Inicial"], errors="coerce").fillna(0)
previsaoinicial2025["Variante"] = previsaoinicial2025["Referencia"] + "_" + previsaoinicial2025["Cor"]

# Consolidando a Base de Comparação
vendas_real_2025 = vendas2025.groupby("Variante", as_index=False)["Quantidade Faturada"].sum().rename(columns={"Quantidade Faturada": "Qtd_Real"})
prev_empresa_2025 = previsaoinicial2025.groupby("Variante", as_index=False)["Previsao Inicial"].sum().rename(columns={"Previsao Inicial": "Qtd_Prevista_Empresa"})

comparacao_2025 = previsao_final_2025.merge(prev_empresa_2025, on="Variante", how="left").merge(vendas_real_2025, on="Variante", how="left")
comparacao_2025["Qtd_Prevista_Empresa"] = comparacao_2025["Qtd_Prevista_Empresa"].fillna(0)
comparacao_2025["Qtd_Real"] = comparacao_2025["Qtd_Real"].fillna(0)

#%% Cálculo das Métricas

# Erro Absoluto 
comparacao_2025["erro_empresa"] = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Qtd_Prevista_Empresa"])
comparacao_2025["erro_xgboost"] = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Previsao_XGBoost"])

# MAPE (Média de erros percentuais)
comparacao_2025["erro_pct_empresa"] = comparacao_2025["erro_empresa"] / comparacao_2025["Qtd_Real"].replace(0, 1)
comparacao_2025["erro_pct_xgboost"] = comparacao_2025["erro_xgboost"] / comparacao_2025["Qtd_Real"].replace(0, 1)
mape_empresa = comparacao_2025["erro_pct_empresa"].mean()
mape_xgboost = comparacao_2025["erro_pct_xgboost"].mean()

# WAPE (Erro ponderado pelo volume total)
volume_total_real = comparacao_2025["Qtd_Real"].sum()
wape_empresa = comparacao_2025["erro_empresa"].sum() / volume_total_real
wape_xgboost = comparacao_2025["erro_xgboost"].sum() / volume_total_real

# Resultados da comparações
print("\n--- RESULTADOS GERAIS DE AVALIAÇÃO - AV 2025 ---")
print(f"Volume Total Real da Coleção: {int(volume_total_real):,}".replace(",", "."))

print("\n[Volume Total Previsto]")
print(f"Empresa: {int(volume_empresa):,}".replace(",", "."))
print(f"XGBoost: {int(volume_xgboost):,}".replace(",", "."))

print("\n[Diferença vs Real (%)]")
print(f"Empresa: {(volume_empresa - volume_total_real) / volume_total_real:.2%}")
print(f"XGBoost: {(volume_xgboost - volume_total_real) / volume_total_real:.2%}")

print("\n[MAPE Médio]")
print(f"Empresa: {mape_empresa:.2%}")
print(f"XGBoost: {mape_xgboost:.2%}")

print("\n[WAPE]")
print(f"Empresa: {wape_empresa:.2%}")
print(f"XGBoost: {wape_xgboost:.2%}")

print("\n[Assertividade WAPE (1 - WAPE)]")
print(f"Empresa: {1 - wape_empresa:.2%}")
print(f"XGBoost: {1 - wape_xgboost:.2%}")
print("-" * 50 + "\n")

#%% Construção de Gráficos e Tabelas das Comparações Finais

# Gráfico de Comparação de Volume por Marca
def autolabel(rects, ax): # colocando rotulo nas barras
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}'.replace(",", "."),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

comp_marca = comparacao_2025.groupby("Marca")[["Qtd_Real", "Qtd_Prevista_Empresa", "Previsao_XGBoost"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
marcas = comp_marca["Marca"].astype(str)
x = np.arange(len(marcas))
width = 0.25

cores_grafico = azul_palette(3)
rects1 = ax.bar(x - width, comp_marca["Qtd_Real"], width=width, color=cores_grafico[0], label="Real")
rects2 = ax.bar(x, comp_marca["Qtd_Prevista_Empresa"], width=width, color=cores_grafico[1], label="Prev. Empresa")
rects3 = ax.bar(x + width, comp_marca["Previsao_XGBoost"], width=width, color=cores_grafico[2], label="Prev. XGBoost")

ax.set_ylabel("Quantidade Faturada (Peças)")
ax.set_xticks(x)
ax.set_xticklabels(marcas, ha="center")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

autolabel(rects1, ax)
autolabel(rects2, ax)
autolabel(rects3, ax)

plt.tight_layout()
salvar_fig("15_comparacao_volume_marca.png")
plt.show()

# Tabela de Comparação de Volume e Diferença Percentual por Categoria

# Agrupando dados por Categoria
comp_cat = comparacao_2025.groupby("Categoria")[["Qtd_Real", "Qtd_Prevista_Empresa", "Previsao_XGBoost"]].sum().reset_index()
comp_cat = comp_cat.sort_values(by="Qtd_Real", ascending=False)

tabela_cat = pd.DataFrame()
tabela_cat["Categoria"] = comp_cat["Categoria"]
tabela_cat["Volume Real"] = comp_cat["Qtd_Real"].apply(lambda x: f"{int(x):,}").str.replace(",", ".")

tabela_cat["Prev. Empresa"] = comp_cat["Qtd_Prevista_Empresa"].apply(lambda x: f"{int(x):,}").str.replace(",", ".")
tabela_cat["Dif. Empresa (%)"] = ((comp_cat["Qtd_Prevista_Empresa"] - comp_cat["Qtd_Real"]) / comp_cat["Qtd_Real"].replace(0, 1)).apply(lambda x: f"{x:+.2%}")
tabela_cat["Prev. XGBoost"] = comp_cat["Previsao_XGBoost"].apply(lambda x: f"{int(x):,}").str.replace(",", ".")
tabela_cat["Dif. XGBoost (%)"] = ((comp_cat["Previsao_XGBoost"] - comp_cat["Qtd_Real"]) / comp_cat["Qtd_Real"].replace(0, 1)).apply(lambda x: f"{x:+.2%}")

# Gerando a imagem da tabela
fig, ax = plt.subplots(figsize=(11, len(tabela_cat) * 0.4 + 1.5))
fig.patch.set_facecolor('white')
ax.axis("off")

tabela_plot = ax.table(cellText=tabela_cat.values, colLabels=tabela_cat.columns, loc="center", cellLoc="center")
tabela_plot.auto_set_font_size(False)
tabela_plot.set_fontsize(12)
tabela_plot.scale(1, 1.8)

for (row, col), cell in tabela_plot.get_celld().items():
    if row > 0:
        if col in [3, 5]:
            valor_texto = cell.get_text().get_text()
            if valor_texto.startswith('+'):
                cell.set_facecolor("#D4EFDF")
            elif valor_texto.startswith('-'):
                cell.set_facecolor("#FADBD8")

plt.tight_layout()
salvar_fig("16_tabela_volume_categoria.png")
plt.show()

#%% FIM