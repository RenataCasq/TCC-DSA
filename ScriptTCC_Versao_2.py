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
from scipy.stats import pearsonr # correlações de Pearson
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

#%% Informações gerais sobre o DataFrame para conferir tipos de variáveis

    #Sobre VENDAS
print(vendas2021.info())
print(vendas2022.info())
print(vendas2023.info())
print(vendas2024.info())
    #Sobre PRODUTOS
print(produtos2021.info())
print(produtos2022.info())
print(produtos2023.info())
print(produtos2024.info())
print(produtos2025.info())
    #Sobre VARIANTE
print(produtosEcor2025.info())

#%% Estatísticas descritiva das variáveis

    #Sobre VENDAS
print(vendas2021.describe())
print(vendas2022.describe())
print(vendas2023.describe())
print(vendas2024.describe())
    #Sobre PRODUTOS
print(produtos2021.describe())
print(produtos2022.describe())
print(produtos2023.describe())
print(produtos2024.describe())
print(produtos2025.describe())
    #Sobre VARIANTE
print(produtosEcor2025.describe())

#%% PARTE 2 - ANÁLISE EXPLORATÓRIA DOS DADOS

#%% Padronização da configuração dos gráficos e tabelas

#Padrão de fontes
plt.style.use("default")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False})

#Pasta para salvar imagens
OUT_DIR = "figuras_eda"
os.makedirs(OUT_DIR, exist_ok=True)

#Padrão de cores (azul claro -> azul escuro)
def azul_palette(n):
    cmap = cm.get_cmap("Blues")
    vals = np.linspace(0.35, 0.85, n)
    return [cmap(v) for v in vals]

AZUL_ESCURO = cm.get_cmap("Blues")(0.80)
AZUL_MEDIO  = cm.get_cmap("Blues")(0.60)
AZUL_CLARO  = cm.get_cmap("Blues")(0.40)

#Padrão do separador dos números
def format_milhar_br(x):
    try:
        return f"{int(round(x)):,}".replace(",", ".")
    except:
        return str(x)

#Padrão do qualidade de imagens
def salvar_fig(nome_arquivo):
    plt.savefig(os.path.join(OUT_DIR, nome_arquivo), dpi=300, bbox_inches="tight")

#Padrão de informação nos rótulos
def add_labels_barras(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, format_milhar_br(v), ha="center", va="bottom", fontsize=9)

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

#%% Análise do volume de vendas (em peças) dos anos de 2021 a 2024 
volume_ano = (vendas_completo.groupby("Ano")["Quantidade Faturada"].sum())
volume_ano = volume_ano[volume_ano > 0]

fig, ax = plt.subplots(figsize=(8,5))
cores = azul_palette(len(volume_ano))
ax.bar(volume_ano.index, volume_ano.values, color=cores)

ax.set_title("Volume Total de Vendas (Peças) de Alto Verão por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
add_labels_barras(ax, volume_ano.values)

plt.tight_layout()
salvar_fig("01_volume_vendas_por_ano.png")
plt.show()

#%% Análise da quantidade de referências disponíveis nos anos 2021 a 2025
refs_ano = produtos_total.groupby("Ano")["Referencia"].nunique().sort_index()

fig, ax = plt.subplots(figsize=(8,5))
cores = azul_palette(len(refs_ano))
ax.bar(refs_ano.index.astype(str), refs_ano.values, color=cores)

ax.set_title("Quantidade de Referências em Alto Verão por Ano (2021 a 2025)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade de Referências")
ax.grid(axis="y", linestyle="--", alpha=0.5)
add_labels_barras(ax, refs_ano.values)

plt.tight_layout()
salvar_fig("02_qtd_referencias_por_ano.png")
plt.show()

#%% Análise da quantidade de referências por categoria nos anos de 2021 a 2025 (tabela e painel de gráficos)
# → muda padrão de venda

ref_cat = (produtos_total.groupby(["Ano","Categoria"])["Referencia"].nunique().reset_index(name="Qtd_Ref"))

#Tabela
tabela_ref_cat = ref_cat.pivot(index="Categoria", columns="Ano", values="Qtd_Ref").fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(9,6))
ax.axis('off')

tabela_plot = ax.table(
    cellText=tabela_ref_cat.values,
    colLabels=tabela_ref_cat.columns,
    rowLabels=tabela_ref_cat.index,
    loc='upper left',
    cellLoc='center')
tabela_plot.auto_set_font_size(False)
tabela_plot.set_fontsize(10)
tabela_plot.scale(0.9, 1.4)

for (row, col), cell in tabela_plot.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#D6EAF8")

plt.title("Quantidade de Referências de Alto Verão por Ano – Painel por Categoria (2021 a 2024)", pad=20, loc='center', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_tabela_ref_por_categoria.png"), dpi=300, bbox_inches='tight')
plt.show()

#Gráficos
anos = [2021, 2022, 2023, 2024, 2025]
anos_existentes = [a for a in anos if a in tabela_ref_cat.columns]

df_cat = tabela_ref_cat[anos_existentes].copy()

def paleta_azul_gradiente(n):
    cmap = cm.get_cmap("Blues")
    # faixa de tons: 0.35 (claro) a 0.85 (escuro)
    vals = np.linspace(0.35, 0.85, n)
    return [cmap(v) for v in vals]
cores = paleta_azul_gradiente(len(anos_existentes))

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
axes = axes.ravel()
x = np.arange(len(anos_existentes))
for i, categoria in enumerate(df_cat.index):
    ax = axes[i]
    valores = df_cat.loc[categoria].values

    ax.bar(x, valores, color=cores, width=0.7)

    ax.set_title(str(categoria), fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos_existentes], fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for xi, v in zip(x, valores):
        ax.text(xi, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Quantidade de Referências de Alto Verão por Ano por Categoria (2021 a 2025)", fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("painel_categorias_3x3.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análise do volume de vendas por marca nos anos de 2021 a 2024
vendas_marca = (vendas_completo.groupby(["Ano","Marca"])["Quantidade Faturada"].sum().reset_index())

pivot_marca = vendas_marca.pivot(index="Ano",columns="Marca",values="Quantidade Faturada").fillna(0)
pivot_marca = pivot_marca.loc[pivot_marca.index != "2025"]

fig, ax = plt.subplots(figsize=(9,5))
cores = azul_palette(pivot_marca.shape[1])
pivot_marca.plot(kind="bar", ax=ax, color=cores)

ax.set_title("Volume de Vendas de Alto Verão por Marca e Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(title="Marca", bbox_to_anchor=(1.02,1), loc="upper left")

for container in ax.containers:
    for bar in container:
        altura = bar.get_height()
        if altura > 0:
            ax.text(bar.get_x() + bar.get_width()/2, altura, f'{int(altura):,}'.replace(',', '.'),
            ha='center',
            va='bottom',
            rotation=20,
            fontsize=9)

plt.tight_layout()
salvar_fig("04_vendas_por_marca.png")
plt.show()

#%% Análise do volume de vendas (em peças) por categoria nos anos de 2021 a 2024
def paleta_azul_gradiente(n):
    cmap = cm.get_cmap("Blues")
    vals = np.linspace(0.35, 0.85, n)  # claro -> escuro
    return [cmap(v) for v in vals]

df = vendas_completo.copy()
df["Categoria"] = df["Categoria"].astype(str).str.strip()
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

anos = [2021, 2022, 2023, 2024]

df = df[df["Ano_num"].isin(anos)]

vendas_cat = (df.groupby(["Categoria", "Ano_num"])["Quantidade Faturada"]
      .sum()
      .reset_index())

pivot_cat = (vendas_cat.pivot(index="Categoria", columns="Ano_num", values="Quantidade Faturada")
             .fillna(0))
pivot_cat = pivot_cat.reindex(columns=anos, fill_value=0)

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
axes = axes.ravel()

x = np.arange(len(anos))
cores = paleta_azul_gradiente(len(anos))

for i, categoria in enumerate(pivot_cat.index[:9]):
    ax = axes[i]
    valores = pivot_cat.loc[categoria, anos].values

    ax.bar(x, valores, color=cores, width=0.7)
    ax.set_title(categoria, fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for xi, v in zip(x, valores):
        if v > 0:
            ax.text(
                xi, v,
                f"{int(v):,}".replace(",", "."),
                ha="center", va="bottom",
                fontsize=8, rotation=60)
            
for j in range(i + 1, 9):
    axes[j].axis("off")

fig.suptitle("Volume de Vendas de Alto Verão por Ano – Painel por Categoria (2021 a 2024)", fontsize=14, weight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("05_painel_vendas_por_categoria_3x3.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análise do comportamento de vendas por região do país nos anos de 2021 a 2024
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

ax.set_title("Volume de Vendas de Alto Verão por Região por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Região")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(pivot_reg.index, rotation=20, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="Ano", bbox_to_anchor=(1.02,1), loc="upper left")

for container in ax.containers:
    for bar in container:
        altura = bar.get_height()
        if altura > 0:
            ax.text(bar.get_x() + bar.get_width()/2,altura,f'{int(altura):,}'.replace(',', '.'),
                ha='center',
                va='bottom',
                rotation=22,
                fontsize=8)

plt.tight_layout()
salvar_fig("06_vendas_por_regiao.png")
plt.show()

#%% Análise da quantidade de representantes ativos nos anos de 2021 a 2024
# → mostra capacidade comercial

rep_ativos = vendas_completo.groupby("Ano")["Representante"].nunique().sort_index()
rep_ativos = rep_ativos[rep_ativos > 0]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(rep_ativos.index.astype(str), rep_ativos.values, color=azul_palette(len(rep_ativos)))

ax.set_title("Quantidade de Representantes Ativos em Alto Verão por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Representantes Ativos")
ax.grid(axis="y", linestyle="--", alpha=0.5)
add_labels_barras(ax, rep_ativos.values)

plt.tight_layout()
salvar_fig("07_representantes_ativos.png")
plt.show()

#%% Análise da quantidade de clientes atendidos nos anos de 2021 a 2024
# → mostra expansão de mercado

clientes_ano = vendas_completo.groupby("Ano")["Cliente"].nunique().sort_index()
clientes_ano = clientes_ano[clientes_ano > 0]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(clientes_ano.index.astype(str), clientes_ano.values, color=azul_palette(len(clientes_ano)))

ax.set_title("Quantidade de Clientes Atendidos em Alto Verão por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade de Clientes Atendidos")
ax.grid(axis="y", linestyle="--", alpha=0.5)
add_labels_barras(ax, clientes_ano.values)

plt.tight_layout()
salvar_fig("08_clientes_por_ano.png")
plt.show()

#%% Análise da quantidade de clientes atendidos por região nos anos de 2021 a 2024
anos = [2021, 2022, 2023, 2024]
df = vendas_completo.copy()
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

clientes_reg = (df[df["Ano_num"].isin(anos)].groupby(["Regiao", "Ano_num"])["Cliente"].nunique().reset_index(name="Qtd_Clientes"))
pivot_cli_reg = (clientes_reg.pivot(index="Regiao", columns="Ano_num", values="Qtd_Clientes").fillna(0).reindex(columns=anos, fill_value=0))

pivot_cli_reg = pivot_cli_reg.loc[pivot_cli_reg.sum(axis=1).sort_values(ascending=False).index]
regioes = pivot_cli_reg.index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.ravel()

cores = paleta_azul_gradiente(len(anos))
x = np.arange(len(anos))

for i, regiao in enumerate(regioes):
    ax = axes[i]
    valores = pivot_cli_reg.loc[regiao, anos].values
    ax.bar(x, valores, color=cores, width=0.7)
    ax.set_title(regiao, fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for xi, v in zip(x, valores):
        if v > 0:
            ax.text(xi, v, f"{int(v):,}".replace(",", "."),
                ha="center", va="bottom",
                fontsize=8,
                rotation=30)
            
axes[5].axis("off")

fig.suptitle("Clientes Atendidos em Alto Verão por Ano – Painel por Região (2021 a 2024)",fontsize=15,weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar_fig("09_painel_clientes_por_regiao.png")
plt.show()

#%% Análise do ticket médio (em peças) por cliente por ano entre 2021 e 2024
# → mostra comportamento do cliente

ticket_medio = (vendas_completo.groupby("Ano").apply(lambda x: x["Quantidade Faturada"].sum() / max(x["Cliente"].nunique(), 1)).sort_index())
ticket_medio = ticket_medio[ticket_medio > 0]

fig, ax = plt.subplots(figsize=(8,5))
anos = ticket_medio.index.astype(int)
ax.plot(ticket_medio.index.astype(int), ticket_medio.values, marker="o", color=AZUL_ESCURO, linewidth=2)

ax.set_xticks(anos)
ax.set_xticklabels(anos)
ax.set_title("Ticket Médio de Peças por Cliente em Alto Verão por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Peças por Cliente")
ax.grid(linestyle="--", alpha=0.5)

for x, y in zip(ticket_medio.index.astype(int), ticket_medio.values):
    ax.text(x, y, format_milhar_br(y), ha="right", va="bottom", fontsize=9)

plt.tight_layout()
salvar_fig("10_ticket_medio_por_cliente.png")
plt.show()

#%% Análise da venda média por referência por ano entre 2021 e 2024
# → mostra saturação de coleção

venda_media_ref = (vendas_completo.groupby("Ano").apply(lambda x: x["Quantidade Faturada"].sum() / max(x["Referencia"].nunique(), 1)).sort_index())
venda_media_ref = venda_media_ref[venda_media_ref > 0]

fig, ax = plt.subplots(figsize=(8,5))   
anos = venda_media_ref.index.astype(int)
ax.plot(venda_media_ref.index.astype(int), venda_media_ref.values, marker="o", color=AZUL_MEDIO, linewidth=2)

ax.set_xticks(anos)
ax.set_xticklabels(anos)
ax.set_title("Venda Média de Peças por Referência em Alto Verão por Ano (2021 a 2024)", weight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Peças por Referência")
ax.grid(linestyle="--", alpha=0.5)

for x, y in zip(venda_media_ref.index.astype(int), venda_media_ref.values):
    ax.text(x, y, format_milhar_br(y), ha="right", va="bottom", fontsize=9)

plt.tight_layout()
salvar_fig("11_venda_media_por_referencia.png")
plt.show()

#%% Análise do volume de vendas por grade nos anos de 2021 a 2024

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
cores = paleta_azul_gradiente(len(anos_validos))

for i, ano in enumerate(anos_validos):
    ax.bar(x + i*width, tabela_grade_ano[ano].values, width=width, color=cores[i], label=str(ano))

ax.set_title("Volume de Vendas de Alto Verão por Grade e Ano (2021 a 2024)", weight="bold")
ax.set_xlabel("Grade")
ax.set_ylabel("Quantidade Faturada (Peças)")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(tabela_grade_ano.index, rotation=25, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.35)

ax.legend(title="Ano", bbox_to_anchor=(1.02,1), loc="upper left")
for container in ax.containers:
    for bar in container:
        altura = bar.get_height()
        if altura > 0:
            ax.text(bar.get_x() + bar.get_width()/2, altura, f'{int(altura):,}'.replace(',', '.'),
            ha='center',
            va='bottom',
            rotation=30,
            fontsize=9)

plt.tight_layout()
salvar_fig("12_vendas_por_grade.png")
plt.show()

#%% Análise do Volume de Vendas de Alto Verão por Modelagem no Ano - Painel por top 5 modelagens (2021 a 2024)

vendas_modelagem_ano = (dados_vendas.groupby(["Ano_num","Modelagem"])["Quantidade Faturada"].sum().reset_index())

fig, axes = plt.subplots(2,2, figsize=(13,8))
axes = axes.ravel()

for i, ano in enumerate(anos_validos):
    ax = axes[i]

    top_modelagens = (vendas_modelagem_ano[vendas_modelagem_ano["Ano_num"] == ano].sort_values("Quantidade Faturada", ascending=False)
        .head(5)
        .sort_values("Quantidade Faturada", ascending=True))
    
    ax.barh(top_modelagens["Modelagem"],
            top_modelagens["Quantidade Faturada"],
            color=paleta_azul_gradiente(5)[-2])

    ax.set_title(f"Top 5 Modelagens – {ano}", weight="bold")
    ax.set_xlabel("Quantidade Faturada")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for y, v in enumerate(top_modelagens["Quantidade Faturada"].values):
        ax.text(v, y, f"  {format_milhar_br(v)}", va="center", fontsize=9)

for container in ax.containers:
    for bar in container:
        altura = bar.get_height()
        if altura > 0:
            ax.text(bar.get_x() + bar.get_width()/2, altura, f'{int(altura):,}'.replace(',', '.'),
            ha='center',
            va='bottom',
            rotation=30,
            fontsize=9)
            
plt.suptitle("Volume de Vendas de Alto Verão por Modelagem | Top 5 por Ano", weight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
salvar_fig("13_top5_modelagem_painel.png")
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

    ax.barh(top_gp["Grupo MP"],
            top_gp["Quantidade Faturada"],
            color=paleta_azul_gradiente(5)[-1])

    ax.set_title(f"Top 5 Grupo MP – {ano}", weight="bold")
    ax.set_xlabel("Quantidade Faturada")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for y, v in enumerate(top_gp["Quantidade Faturada"].values):
        ax.text(v, y, f"  {format_milhar_br(v)}", va="center", fontsize=9)
        
    for container in ax.containers:
        for bar in container:
            altura = bar.get_height()
            if altura > 0:
                ax.text(bar.get_x() + bar.get_width()/2, altura, f'{int(altura):,}'.replace(',', '.'),
                ha='center',
                va='bottom',
                rotation=30,
                fontsize=9)

plt.suptitle("Volume de Vendas de Alto Verão por Grupo de Matéria Prima | Top 5 por Ano", weight="bold",fontsize=18)
plt.tight_layout(rect=[0,0,1,0.95])
salvar_fig("14_top5_grupoMP_painel.png")
plt.show()

#%% Correlações
#%% HEATMAP 1 — CORRELAÇÃO "DENTRO DO ANO"
# Agregando por REGIÃO em cada ano

dados_hm = vendas_completo.copy()
dados_hm["Ano_num"] = dados_hm["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

anos_validos = [2021, 2022, 2023, 2024]
dados_hm = dados_hm[dados_hm["Ano_num"].isin(anos_validos)].copy()

for col in ["Regiao", "Categoria"]:
    if col in dados_hm.columns:
        dados_hm[col] = dados_hm[col].astype(str).str.strip().str.upper()

def montar_base_indicadores_por_ano(df_ano):   
    base = (
        df_ano.groupby("Regiao")
        .agg(
            Qtde_Referencias=("Referencia", "nunique"),
            Qtde_Representantes=("Representante", "nunique"),
            Qtde_Clientes=("Cliente", "nunique"),
            Qtde_Pedidos=("Pedido", "nunique"),
            Quantidade_Vendida=("Quantidade Faturada", "sum"),
            Qtde_Categorias=("Categoria", "nunique"),)
        .reset_index())

    base["Ticket_Medio_por_Cliente"] = (
        base["Quantidade_Vendida"] / base["Qtde_Clientes"].replace(0, np.nan))

    base["Venda_Media_por_Referencia"] = (
        base["Quantidade_Vendida"] / base["Qtde_Referencias"].replace(0, np.nan))

    base = base.replace([np.inf, -np.inf], np.nan).fillna(0)
    return base

def plot_heatmap_corr(base, titulo, salvar_como=None): # Gera um heatmap de correlação (Pearson) apenas para colunas numéricas
    num = base.select_dtypes(include=[np.number])
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(titulo, weight="bold")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
            
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches="tight")
    plt.show()

#Rodar os 4 heatmaps: 2021, 2022, 2023, 2024
for ano in anos_validos:
    df_ano = dados_hm[dados_hm["Ano_num"] == ano].copy()
    base_indicadores = montar_base_indicadores_por_ano(df_ano)

    plot_heatmap_corr(base_indicadores,
        titulo=f"Correlação dos Indicadores (por Região) – {ano}",
        salvar_como=f"corr_indicadores_{ano}.png")
    
#%% HEATMAP 2 - POR ANO – AGREGADO POR CATEGORIA
# Cada linha = uma categoria no ano

dados_hm = vendas_completo.copy()
dados_hm["Ano_num"] = dados_hm["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

anos_validos = [2021, 2022, 2023, 2024]
dados_hm = dados_hm[dados_hm["Ano_num"].isin(anos_validos)].copy()

for col in ["Categoria"]:
    if col in dados_hm.columns:
        dados_hm[col] = dados_hm[col].astype(str).str.strip().str.upper()

# Montar base de indicadores por categoria no ano
def montar_base_por_categoria(df_ano):

    base = (
        df_ano.groupby("Categoria")
        .agg(
            Qtde_Referencias=("Referencia", "nunique"),
            Qtde_Representantes=("Representante", "nunique"),
            Qtde_Clientes=("Cliente", "nunique"),
            Qtde_Pedidos=("Pedido", "nunique"),
            Quantidade_Vendida=("Quantidade Faturada", "sum"),)
        .reset_index())

    base["Ticket_Medio_por_Cliente"] = (
        base["Quantidade_Vendida"] /
        base["Qtde_Clientes"].replace(0, np.nan))

    base["Venda_Media_por_Referencia"] = (
        base["Quantidade_Vendida"] /
        base["Qtde_Referencias"].replace(0, np.nan))

    base = base.replace([np.inf, -np.inf], np.nan).fillna(0)
    return base

# Plotar heatmap
def plot_heatmap_corr(base, titulo, salvar=None):

    num = base.select_dtypes(include=[np.number])
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(titulo, weight="bold")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if salvar:
        pasta = os.path.dirname(salvar)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        plt.savefig(salvar, dpi=300, bbox_inches="tight")

    plt.show()

# GERAR OS 4 HEATMAPS
for ano in anos_validos:

    df_ano = dados_hm[dados_hm["Ano_num"] == ano].copy()
    base = montar_base_por_categoria(df_ano)

    plot_heatmap_corr(
        base,
        titulo=f"Correlação entre Indicadores – {ano}",
        salvar=f"outputs/corr_indicadores_categoria_{ano}.png")
    
#%% HEATMAP 3 – PARTICIPAÇÃO (%) DE CATEGORIAS POR REGIÃO

dados_cat = vendas_completo.copy()

# Padronizar texto
dados_cat["Regiao"] = dados_cat["Regiao"].astype(str).str.strip().str.upper()
dados_cat["Categoria"] = dados_cat["Categoria"].astype(str).str.strip().str.upper()

# Volume vendido por Região e Categoria
base_reg_cat = (
    dados_cat.groupby(["Regiao", "Categoria"])["Quantidade Faturada"]
    .sum()
    .reset_index())

# Pivot: linhas = Região | colunas = Categoria
pivot_reg_cat = (
    base_reg_cat.pivot(index="Regiao", columns="Categoria", values="Quantidade Faturada")
    .fillna(0))

# Converter para PARTICIPAÇÃO (%) dentro de cada região
participacao = pivot_reg_cat.div(pivot_reg_cat.sum(axis=1), axis=0) * 100

# Plotar heatmap
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(participacao.values, cmap="Blues")

ax.set_xticks(range(len(participacao.columns)))
ax.set_yticks(range(len(participacao.index)))

ax.set_xticklabels(participacao.columns, rotation=45, ha="right")
ax.set_yticklabels(participacao.index)

ax.set_title("Participação (%) das Categorias nas Vendas por Região", weight="bold")

# Mostrar valores nas células
for i in range(participacao.shape[0]):
    for j in range(participacao.shape[1]):
        valor = participacao.iloc[i, j]
        ax.text(j, i, f"{valor:.1f}%", ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("heatmap_categoria_regiao_participacao.png", dpi=300, bbox_inches="tight")
plt.show()

#%% HEATMAP 4 - PARTICIPAÇÃO (%) – CATEGORIA × REGIÃO POR ANO

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

dados = vendas_completo.copy()

# Garantir que Ano esteja como número (2021, 2022, 2023, 2024)
dados["Ano_num"] = dados["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

anos_validos = [2021, 2022, 2023, 2024]
dados = dados[dados["Ano_num"].isin(anos_validos)].copy()

# Padronizar textos (evita duplicações por espaços / maiúsculas)
dados["Regiao"] = dados["Regiao"].astype(str).str.strip().str.upper()
dados["Categoria"] = dados["Categoria"].astype(str).str.strip().str.upper()

# (Opcional) Fixar a ordem das categorias para ficar igual em todos os anos
ordem_categorias = sorted(dados["Categoria"].dropna().unique())

# (Opcional) Fixar a ordem das regiões
ordem_regioes = ["CENTRO-OESTE", "NORDESTE", "NORTE", "SUDESTE", "SUL"]

# Função para montar e plotar o heatmap de 1 ano

def heatmap_cat_regiao_por_ano(df_ano, ano, salvar=True):
    # 1) Soma do volume vendido por Região e Categoria
    base = (
        df_ano.groupby(["Regiao", "Categoria"])["Quantidade Faturada"]
        .sum()
        .reset_index())

    # 2) Pivot: linhas = Região | colunas = Categoria
    pivot = (
        base.pivot(index="Regiao", columns="Categoria", values="Quantidade Faturada")
        .fillna(0))

    # Garantir mesma ordem (categorias e regiões) em todos os anos
    pivot = pivot.reindex(index=ordem_regioes, columns=ordem_categorias, fill_value=0)

    # 3) Converter em participação (%) dentro de cada região
    participacao = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    participacao = participacao.fillna(0)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(participacao.values, cmap="Blues", vmin=0, vmax=participacao.values.max() if participacao.values.max() > 0 else 1)

    ax.set_xticks(range(len(participacao.columns)))
    ax.set_yticks(range(len(participacao.index)))

    ax.set_xticklabels(participacao.columns, rotation=45, ha="right")
    ax.set_yticklabels(participacao.index)

    ax.set_title(f"Participação (%) das Categorias nas Vendas por Região – {ano}", weight="bold")

    # Valores nas células (com 1 casa decimal)
    for i in range(participacao.shape[0]):
        for j in range(participacao.shape[1]):
            valor = participacao.iloc[i, j]
            ax.text(j, i, f"{valor:.1f}%", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # 5) Salvar figura
    if salvar:
        caminho = os.path.join(OUT_DIR, f"heatmap_cat_regiao_{ano}.png")
        plt.savefig(caminho, dpi=300, bbox_inches="tight")

    plt.show()

# Gerar heatmap para cada ano (2021–2024)

for ano in anos_validos:
    df_ano = dados[dados["Ano_num"] == ano].copy()
    heatmap_cat_regiao_por_ano(df_ano, ano, salvar=True)
    
#%% HEATMAP 5 - (Categoria × Região) em PARTICIPAÇÃO (%) por ANO
# Painel 2×2 (2021–2024)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = vendas_completo.copy()

# Garantir Ano como número (2021, 2022, 2023, 2024)
df["Ano_num"] = df["Ano"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
anos = [2021, 2022, 2023, 2024]
df = df[df["Ano_num"].isin(anos)].copy()

# Padronizar textos
df["Regiao"] = df["Regiao"].astype(str).str.strip().str.upper()
df["Categoria"] = df["Categoria"].astype(str).str.strip().str.upper()

# Fixar ordem de categorias e regiões (padrão visual consistente)
ordem_categorias = sorted(df["Categoria"].dropna().unique())
ordem_regioes = ["CENTRO-OESTE", "NORDESTE", "NORTE", "SUDESTE", "SUL"]

# Montar matriz de participação (%) Região × Categoria para um ano
def matriz_participacao_por_ano(df_ano):
    base = (
        df_ano.groupby(["Regiao", "Categoria"])["Quantidade Faturada"]
        .sum()
        .reset_index())

    pivot = (
        base.pivot(index="Regiao", columns="Categoria", values="Quantidade Faturada")
        .fillna(0))

    # garantir mesma ordem em todos os anos
    pivot = pivot.reindex(index=ordem_regioes, columns=ordem_categorias, fill_value=0)

    # participação (%) dentro de cada região
    part = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    part = part.fillna(0)

    return part

# Pré-calcular as matrizes de todos os anos
matrizes = {}
for ano in anos:
    matrizes[ano] = matriz_participacao_por_ano(df[df["Ano_num"] == ano].copy())

# Definir vmax GLOBAL (mesma escala de cor para todos) -> pega o maior % observado em qualquer ano/célula
vmax_global = max(m.values.max() for m in matrizes.values())
vmax_global = float(np.ceil(vmax_global))

# Plotar painel 2×2
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
axes = axes.flatten()

for ax, ano in zip(axes, anos):
    part = matrizes[ano]

    im = ax.imshow(part.values, cmap="Blues", vmin=0, vmax=vmax_global)

    ax.set_title(f"{ano}", weight="bold")

    ax.set_xticks(range(len(part.columns)))
    ax.set_yticks(range(len(part.index)))

    ax.set_xticklabels(part.columns, rotation=45, ha="right")
    ax.set_yticklabels(part.index)

    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            valor = part.iloc[i, j]
            ax.text(j, i, f"{valor:.1f}%", ha="center", va="center", fontsize=9)

fig.suptitle("Participação (%) das Categorias nas Vendas por Região – Alto Verão (2021–2024)",
             fontsize=14, weight="bold")

cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("Participação (%) dentro da Região", rotation=90)

caminho = os.path.join(OUT_DIR, "heatmap_cat_regiao_painel_2021_2024.png")
plt.savefig(caminho, dpi=300, bbox_inches="tight")
plt.show()

#%% PARTE 3 - RODANDO O XGBOOST

#%% Padronização dos dados e união das tabelas
pd.set_option("display.max_columns", 200)

# Padronização de Nomes de Colunas e Tipos
def padronizar_texto(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df

cols_prod = ["Referencia", "Grade", "Categoria", "Modelagem", "Grupo MP", "Ano"]
cols_vend = ["Representante", "Cliente", "Estado", "Pedido", "Marca", "Referencia", "Cor", "Quantidade Faturada", "Ano"]

for dfp in [produtos2021, produtos2022, produtos2023, produtos2024, produtos2025]:
    padronizar_texto(dfp, ["Referencia", "Grade", "Categoria", "Modelagem", "Grupo MP"])
    if "Ano" in dfp.columns:
        dfp["Ano"] = dfp["Ano"].astype(str).str.extract(r"(\d{4})")[0]
        
for dfv in [vendas2021, vendas2022, vendas2023, vendas2024]:
    padronizar_texto(dfv, ["Representante", "Cliente", "Estado", "Pedido", "Marca", "Referencia", "Cor"])
    dfv["Ano"] = dfv["Ano"].astype(str).str.extract(r"(\d{4})")[0]
    dfv["Quantidade Faturada"] = pd.to_numeric(dfv["Quantidade Faturada"], errors="coerce").fillna(0)

padronizar_texto(produtosEcor2025, ["Referencia", "Cor", "Grade", "Categoria", "Modelagem", "Grupo MP", "Marca"])

# Adição de Regioes nas tabelas de Vendas
def add_regiao(df_vendas):
    if "Regiao" not in df_vendas.columns and "Estado" in df_vendas.columns:
        df_vendas["Regiao"] = df_vendas["Estado"].map(mapa_regioes).fillna("DESCONHECIDA")
    return df_vendas

for dfv in [vendas2021, vendas2022, vendas2023, vendas2024]:
    add_regiao(dfv)

# Concatenar 2021 a 2024 e criar chave do produto (prod)-> Referencia + Cor 
vendas_hist = pd.concat([vendas2021, vendas2022, vendas2023, vendas2024], ignore_index=True)
produtos_hist = pd.concat([produtos2021, produtos2022, produtos2023, produtos2024], ignore_index=True)

vendas_hist["SKU"] = vendas_hist["Referencia"].astype(str) + "_" + vendas_hist["Cor"].astype(str)

# Ordenação por Ano numérico
vendas_hist["Ano"] = vendas_hist["Ano"].astype(int)
produtos_hist["Ano"] = produtos_hist["Ano"].astype(int)

# Agrupar vendas com produtos para unificar caracteristicas
vendas_prod = vendas_hist.merge(
    produtos_hist[["Ano","Referencia","Grade","Categoria","Modelagem","Grupo MP"]],
    on=["Ano","Referencia"],
    how="left")

# Verificação produtos ficaram sem categoria/modelagem...
print("Linhas com características faltando após merge:")
print(vendas_prod[["Grade","Categoria","Modelagem","Grupo MP"]].isna().mean().sort_values(ascending=False))

#%% Criação de Base Spervisionada -> a nível de produto por ano

# X = características do produto + indicadores comerciais agregados
# y = quantidade total vendida daquele SKU naquele ano

base_ano_sku = (
    vendas_prod.groupby(["Ano","SKU","Referencia","Cor","Marca","Grade","Categoria","Modelagem","Grupo MP"])
    .agg(
        y_qtd=("Quantidade Faturada", "sum"),
        pedidos=("Pedido", "nunique"),
        clientes=("Cliente", "nunique"),
        reps=("Representante", "nunique"),
        estados=("Estado", "nunique"),
        regioes=("Regiao", "nunique"),)
    .reset_index())

print("\nBase SKU-Ano pronta:")
print(base_ano_sku.head())
print(base_ano_sku.info())

#%% Definição dos tipo de produtos pela venda em cada ano (para modelo poder generalizar a forma de análise por tipo)

# - Para produtos RECORRENTES: Lags por SKU = venda do ano anterior
# - Para produtos NOVOS: Médias por grupo (Categoria/Modelagem/Grupo MP/Marca/Grade) 

# Lag 1 (venda do ano anterior do mesmo SKU)
base_ano_sku = base_ano_sku.sort_values(["SKU", "Ano"]).reset_index(drop=True) #ordenando
base_ano_sku["lag1_y"] = base_ano_sku.groupby("SKU")["y_qtd"].shift(1)

# Média histórica do SKU até o ano anterior
base_ano_sku["mean_hist_sku"] = (
    base_ano_sku.groupby("SKU")["y_qtd"]
    .transform(lambda s: s.shift(1).expanding().mean()))

# Médias por grupo (estatísticas para os SKUs novos)
grupo_cols = ["Categoria", "Modelagem", "Grupo MP", "Marca", "Grade"]

# Média do grupo por ano anterior (considera do anterior para o seguinte)
base_ano_sku["mean_grupo_ano"] = (
    base_ano_sku.groupby(grupo_cols + ["Ano"])["y_qtd"].transform("mean"))

base_ano_sku["mean_grupo_prev"] = (
    base_ano_sku.groupby(grupo_cols)["mean_grupo_ano"].shift(1))

# Preenchimentos, quando não há histórico
# - lag1_y e mean_hist_sku: preenche com 0 se não tinha histórico
# - mean_grupo_prev: preenche com média global do ano anterior
base_ano_sku["lag1_y"] = base_ano_sku["lag1_y"].fillna(0)
base_ano_sku["mean_hist_sku"] = base_ano_sku["mean_hist_sku"].fillna(0)

base_ano_sku["mean_grupo_prev"] = base_ano_sku["mean_grupo_prev"].fillna(base_ano_sku["y_qtd"].mean())
base_ano_sku.drop(columns=["mean_grupo_ano"], inplace=True)

#%% Difinição da Base de Treino e Validação
# Melhor que 80/20 aleatório: validação no último ano (2024)
# Treino com 2021 a 2023
# Validação com 2024

train_df = base_ano_sku[base_ano_sku["Ano"].isin([2021,2022,2023])].copy()
valid_df = base_ano_sku[base_ano_sku["Ano"] == 2024].copy()

# Features categóricas e numéricas
cat_features = ["Categoria","Modelagem","Grupo MP","Marca","Grade","Cor"]
num_features = ["pedidos","clientes","reps","estados","regioes","lag1_y","mean_hist_sku","mean_grupo_prev"]

# X e y
X_train_raw = train_df[cat_features + num_features].copy()
y_train = train_df["y_qtd"].copy()

X_valid_raw = valid_df[cat_features + num_features].copy()
y_valid = valid_df["y_qtd"].copy()

# One Hot Encoding (para dummiezar variáveis categóricas)
X_train = pd.get_dummies(X_train_raw, drop_first=False)
X_valid = pd.get_dummies(X_valid_raw, drop_first=False)
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0) # Alinhar colunas

#%% Treino do XGBOOST (Regressão) e Validação

# Treino com infos de 2021 a 2023
modelo = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,      
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,       
    eval_metric="rmse")

modelo.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False)

# Avaliação com 2024
pred_valid = modelo.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
r2 = r2_score(y_valid, pred_valid)

print("\nDesempenho na validação (Ano 2024):")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2%}")

#%% Preparação da Base de Pevisão para 2025

# Criação do SKU
produtosEcor2025["SKU"] = produtosEcor2025["Referencia"].astype(str) + "_" + produtosEcor2025["Cor"].astype(str)

# Para 2025, pedidos/clientes/reps/estados/regioes não existem -> colocar 0
base_prev_2025 = produtosEcor2025.copy()
base_prev_2025["pedidos"] = 0
base_prev_2025["clientes"] = 0
base_prev_2025["reps"] = 0
base_prev_2025["estados"] = 0
base_prev_2025["regioes"] = 0

# Para produtos recorrentes, buscar lag1_y e mean_hist_sku com base no histórico 
# Venda total de 2024 por SKU
sku_2024 = base_ano_sku[base_ano_sku["Ano"] == 2024][["SKU","y_qtd"]].copy()
sku_2024 = sku_2024.rename(columns={"y_qtd": "lag1_y_2025"})

# Média histórica por SKU (2021-2024)
sku_mean = base_ano_sku.groupby("SKU")["y_qtd"].mean().reset_index(name="mean_hist_sku_2025")

# Média por grupo (Categoria/Modelagem/Grupo MP/Marca/Grade) no último ano (2024)
grupo_2024 = (
    base_ano_sku[base_ano_sku["Ano"] == 2024]
    .groupby(grupo_cols)["y_qtd"].mean()
    .reset_index(name="mean_grupo_prev_2025"))

# Junção das feartures no bd de 2025
base_prev_2025 = base_prev_2025.merge(sku_2024, on="SKU", how="left")
base_prev_2025 = base_prev_2025.merge(sku_mean, on="SKU", how="left")
base_prev_2025 = base_prev_2025.merge(grupo_2024, on=grupo_cols, how="left")

# Preencher NaNs:
# - SKU novo: lag1 e mean_hist_sku ficam NaN -> vira 0
# - grupo não apareceu em 2024: mean_grupo_prev_2025 -> média global
base_prev_2025["lag1_y_2025"] = base_prev_2025["lag1_y_2025"].fillna(0)
base_prev_2025["mean_hist_sku_2025"] = base_prev_2025["mean_hist_sku_2025"].fillna(0)
base_prev_2025["mean_grupo_prev_2025"] = base_prev_2025["mean_grupo_prev_2025"].fillna(base_ano_sku["y_qtd"].mean())

# Renomear colunas igual a base de treino e criar base X_previsao com as mesmas features
base_prev_2025["lag1_y"] = base_prev_2025["lag1_y_2025"]
base_prev_2025["mean_hist_sku"] = base_prev_2025["mean_hist_sku_2025"]
base_prev_2025["mean_grupo_prev"] = base_prev_2025["mean_grupo_prev_2025"]

X_prev_raw = base_prev_2025[cat_features + num_features].copy()
X_prev = pd.get_dummies(X_prev_raw, drop_first=False)
X_prev = X_prev.reindex(columns=X_train.columns, fill_value=0)

#%% Geração da Previsão de 2025

pred_2025 = modelo.predict(X_prev)

previsao_2025 = base_prev_2025[["Referencia","Cor","Grade","Categoria","Modelagem","Grupo MP"]].copy()
previsao_2025["Previsao_Qtd_Faturada"] = np.round(pred_2025, 0).astype(int)

previsao_2025["Previsao_Qtd_Faturada"] = previsao_2025["Previsao_Qtd_Faturada"].clip(lower=0) # Evitar valores negativos (às vezes ML pode prever negativo)
previsao_2025 = previsao_2025.sort_values("Previsao_Qtd_Faturada", ascending=False) # ordenando decrecente

print("\nTabela final - previsão 2025 (top 20):")
print(previsao_2025.head(20))

# Exportar
previsao_2025.to_csv(os.path.join(OUT_DIR, "previsao_AV_2025_por_referencia_cor.csv"),
                     sep=";", index=False, encoding="utf-8-sig")

#%% PARTE 4 - AVALIAÇÃO E COMPARAÇÕES COM O PREVISÃO OBTIDA

#%% Importação de Dados de AV 2025

vendas2025 = pd.read_csv('vendas2025.csv', sep=';', encoding='latin1')
previsaoinicial2025 = pd.read_csv('previsaoinicial2025.csv', sep=';', encoding='latin1')

#Conversão dos tipos de Variáveis de Produtos
vendas2025[vendas_str] = vendas2025[vendas_str].astype(str)

previsaoinicial2025['Referencia'] = previsaoinicial2025['Referencia'].astype(str)
previsaoinicial2025['Cor'] = previsaoinicial2025['Cor'].astype(str)

#%% Preparação das Bases de Comparações

# Padronizar vendas reais de 2025
vendas2025["Referencia"] = vendas2025["Referencia"].astype(str).str.strip().str.upper()
vendas2025["Cor"] = vendas2025["Cor"].astype(str).str.strip().str.upper()
vendas2025["Quantidade Faturada"] = pd.to_numeric(vendas2025["Quantidade Faturada"], errors="coerce").fillna(0)

vendas2025["SKU"] = vendas2025["Referencia"] + "_" + vendas2025["Cor"] # Criar SKU

vendas_real_2025 = (
    vendas2025.groupby(["SKU", "Referencia", "Cor", "Marca"], as_index=False)
    .agg(Qtd_Real=("Quantidade Faturada", "sum"))) # Agregar as vendas reais por SKU

# Padronizar previsão inicial da empresa
previsaoinicial2025["Referencia"] = previsaoinicial2025["Referencia"].astype(str).str.strip().str.upper()
previsaoinicial2025["Cor"] = previsaoinicial2025["Cor"].astype(str).str.strip().str.upper()
previsaoinicial2025["Previsao Inicial"] = pd.to_numeric(previsaoinicial2025["Previsao Inicial"], errors="coerce").fillna(0)

previsaoinicial2025["SKU"] = previsaoinicial2025["Referencia"] + "_" + previsaoinicial2025["Cor"] # Criar SKU

prev_empresa_2025 = (
    previsaoinicial2025.groupby(["SKU", "Referencia", "Cor"], as_index=False)
    .agg(Qtd_Prevista_Empresa=("Previsao Inicial", "sum"))) # Agregar previsão inicial por SKU

# Padronizar previsão do modelo
previsao_2025["Referencia"] = previsao_2025["Referencia"].astype(str).str.strip().str.upper()
previsao_2025["Cor"] = previsao_2025["Cor"].astype(str).str.strip().str.upper()
previsao_2025["SKU"] = previsao_2025["Referencia"] + "_" + previsao_2025["Cor"]

#%% Consolidação da Bases de Dados para Comparações

comparacao_2025 = previsao_2025.merge(
    prev_empresa_2025[["SKU", "Qtd_Prevista_Empresa"]],
    on="SKU",
    how="left")

comparacao_2025 = comparacao_2025.merge(
    vendas_real_2025[["SKU", "Qtd_Real", "Marca"]],
    on="SKU",
    how="left")

# Preencher faltantes com zero
comparacao_2025["Qtd_Prevista_Empresa"] = comparacao_2025["Qtd_Prevista_Empresa"].fillna(0)
comparacao_2025["Qtd_Real"] = comparacao_2025["Qtd_Real"].fillna(0)

# Incluir marca a partir das vendas reais
if "Marca_x" in comparacao_2025.columns and "Marca_y" in comparacao_2025.columns:
    comparacao_2025["Marca"] = comparacao_2025["Marca_x"].fillna(comparacao_2025["Marca_y"])
elif "Marca_y" in comparacao_2025.columns:
    comparacao_2025["Marca"] = comparacao_2025["Marca_y"]

print("\nBase consolidada de comparação:")
print(comparacao_2025.head())

#%% Comparação do Volume Total de Vendas

total_real = comparacao_2025["Qtd_Real"].sum()
total_empresa = comparacao_2025["Qtd_Prevista_Empresa"].sum()
total_modelo = comparacao_2025["Previsao_Qtd_Faturada"].sum()

print("\nVolumes totais:")
print(f"Real 2025: {total_real:,.0f}".replace(",", "."))
print(f"Previsão Inicial Empresa: {total_empresa:,.0f}".replace(",", "."))
print(f"Previsão com XGBoost: {total_modelo:,.0f}".replace(",", "."))

plt.figure(figsize=(8,5))
labels = ["Real", "Prev. Empresa", "Prev. XGBoost"]
values = [total_real, total_empresa, total_modelo]

plt.bar(labels, values)
for i, v in enumerate(values):
    plt.text(i, v, f"{int(v):,}".replace(",", "."), ha="center", va="bottom", fontsize=10)

plt.title("Comparação do Volume Total de Vendas AV 2025", weight="bold")
plt.ylabel("Peças")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparacao_volume_total_2025.png"), dpi=300, bbox_inches="tight")
plt.show()

#%% Análise da Quantidade de SKUs com porevisão zerada pelo modelo do XGBoost
# -> Ao verificar volume total previsionado muito abaixo, fez-se essa analise

skus_zero = (comparacao_2025["Previsao_Qtd_Faturada"] == 0).sum()
total_skus = len(comparacao_2025)
pct_zero = skus_zero / total_skus * 100

print("\nSKUs com previsão zero no modelo:")
print(f"Quantidade: {skus_zero}")
print(f"Percentual: {pct_zero:.2f}%")

# Exportar lista dos SKUs zerados
skus_zerados = comparacao_2025[comparacao_2025["Previsao_Qtd_Faturada"] == 0].copy()
skus_zerados.to_csv(os.path.join(OUT_DIR, "skus_previsao_zero_modelo.csv"),
                    sep=";", index=False, encoding="utf-8-sig")

#%% Comparação do Volume de Venda por SKU

comp_categoria = (
    comparacao_2025.groupby("Categoria", as_index=False)
    .agg(
        Real=("Qtd_Real", "sum"),
        Prev_Empresa=("Qtd_Prevista_Empresa", "sum"),
        Prev_Modelo=("Previsao_Qtd_Faturada", "sum")))

print("\nComparação por categoria:")
print(comp_categoria)

comp_categoria.set_index("Categoria").plot(kind="bar", figsize=(11,6))
plt.title("Comparação do Volume de Vendas por Categoria - AV 2025", weight="bold")
plt.ylabel("Peças")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparacao_categoria_2025.png"), dpi=300, bbox_inches="tight")
plt.show()

#%% Assertividade Média por SKU (MAPE)

# Erro absoluto
comparacao_2025["erro_empresa"] = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Qtd_Prevista_Empresa"])
comparacao_2025["erro_modelo"] = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Previsao_Qtd_Faturada"])

# Erro percentual
# Para evitar divisão por zero, quando Qtd_Real = 0 usamos 1 no denominador
comparacao_2025["erro_pct_empresa"] = comparacao_2025["erro_empresa"] / comparacao_2025["Qtd_Real"].replace(0, 1)
comparacao_2025["erro_pct_modelo"] = comparacao_2025["erro_modelo"] / comparacao_2025["Qtd_Real"].replace(0, 1)

mape_empresa = comparacao_2025["erro_pct_empresa"].mean()
mape_modelo = comparacao_2025["erro_pct_modelo"].mean()

acertividade_empresa = 1 - mape_empresa
acertividade_modelo = 1 - mape_modelo

print("\nAcertividade média por SKU:")
print(f"Empresa: {acertividade_empresa:.2%}")
print(f"XGBoost: {acertividade_modelo:.2%}")

plt.figure(figsize=(7,5))
labels = ["Prev. Empresa", "Prev. XGBoost"]
values = [acertividade_empresa * 100, acertividade_modelo * 100]

plt.bar(labels, values)
for i, v in enumerate(values):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

plt.title("Acertividade Média por SKU", weight="bold")
plt.ylabel("Acertividade (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "acertividade_media_por_sku.png"), dpi=300, bbox_inches="tight")
plt.show()

#%% WAPE
# Soma dos erros absolutos
erro_total_empresa = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Qtd_Prevista_Empresa"]).sum()
erro_total_modelo = abs(comparacao_2025["Qtd_Real"] - comparacao_2025["Previsao_Qtd_Faturada"]).sum()

# Soma do volume real
volume_total_real = comparacao_2025["Qtd_Real"].sum()

# Cálculo do WAPE
wape_empresa = erro_total_empresa / volume_total_real
wape_modelo = erro_total_modelo / volume_total_real

print("\nWAPE (Erro Percentual Ponderado):")
print(f"Empresa: {wape_empresa:.2%}")
print(f"XGBoost: {wape_modelo:.2%}")

# Acertividade baseada no WAPE
acc_wape_empresa = 1 - wape_empresa
acc_wape_modelo = 1 - wape_modelo

print("\nAcertividade baseada no WAPE:")
print(f"Empresa: {acc_wape_empresa:.2%}")
print(f"XGBoost: {acc_wape_modelo:.2%}")

#%% Distribuição de Vendas (Real x Previstos) - Top 20 produtos

top20_real = comparacao_2025.sort_values("Qtd_Real", ascending=False).head(20).copy()

plt.figure(figsize=(12,7))
x = np.arange(len(top20_real))
width = 0.25

plt.bar(x - width, top20_real["Qtd_Real"], width=width, label="Real")
plt.bar(x, top20_real["Qtd_Prevista_Empresa"], width=width, label="Prev. Empresa")
plt.bar(x + width, top20_real["Previsao_Qtd_Faturada"], width=width, label="Prev. XGBoost")

plt.xticks(x, top20_real["SKU"], rotation=90)
plt.title("Top 20 Produtos (SKU) - Real x Previsões", weight="bold")
plt.ylabel("Peças")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top20_skus_real_vs_previstos.png"), dpi=300, bbox_inches="tight")
plt.show()

#%% GRÁFICO FINAL Comparação entre:
# - histórico real por ano
# - previsão empresa 2025
# - previsão xgboost 2025


historico_total = (
    base_ano_sku.groupby("Ano", as_index=False)["y_qtd"]
    .sum()
    .rename(columns={"y_qtd": "Volume"}))

fig, ax = plt.subplots(figsize=(10,6))

# Histórico real
ax.plot(historico_total["Ano"], historico_total["Volume"], marker="o", linewidth=2, label="Real Histórico")

# Pontos finais 2025
ax.scatter([2025], [total_empresa], s=100, label="Prev. Empresa 2025")
ax.scatter([2025], [total_modelo], s=100, label="Prev. XGBoost 2025")

# Rótulos
for x, y in zip(historico_total["Ano"], historico_total["Volume"]):
    ax.text(x, y, f"{int(y):,}".replace(",", "."), ha="center", va="bottom", fontsize=9)

ax.text(2025, total_empresa, f"{int(total_empresa):,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
ax.text(2025, total_modelo, f"{int(total_modelo):,}".replace(",", "."), ha="center", va="bottom", fontsize=9)

ax.set_title("Evolução Histórica e Comparação das Previsões para 2025", weight="bold")
ax.set_xlabel("Ano")
ax.set_ylabel("Volume Total de Vendas (Peças)")
ax.set_xticks([2021, 2022, 2023, 2024, 2025])
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "grafico_final_tcc_historico_vs_previsoes.png"), dpi=300, bbox_inches="tight")
plt.show()

#%%GRÁFICO COMPARATIVO DO WAPE

plt.figure(figsize=(6,4))

labels = ["Prev. Empresa", "Prev. XGBoost"]
values = [wape_empresa * 100, wape_modelo * 100]

plt.bar(labels, values)

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

plt.title("Erro Percentual Ponderado (WAPE)")
plt.ylabel("Erro (%)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wape_comparacao.png"), dpi=300)
plt.show()

#%% TABELA FINAL DE AVALIAÇÃO DO MODELO

# Volume total
volume_real = comparacao_2025["Qtd_Real"].sum()
volume_empresa = comparacao_2025["Qtd_Prevista_Empresa"].sum()
volume_modelo = comparacao_2025["Previsao_Qtd_Faturada"].sum()

# Diferença percentual do volume total
dif_empresa = (volume_empresa - volume_real) / volume_real
dif_modelo = (volume_modelo - volume_real) / volume_real

# SKUs com previsão zero
skus_zero_modelo = (comparacao_2025["Previsao_Qtd_Faturada"] == 0).sum()

# Total de SKUs
total_skus = len(comparacao_2025)

# MAPE médio por SKU
mape_empresa = comparacao_2025["erro_pct_empresa"].mean()
mape_modelo = comparacao_2025["erro_pct_modelo"].mean()

# Acertividade baseada no MAPE
acc_mape_empresa = 1 - mape_empresa
acc_mape_modelo = 1 - mape_modelo

# WAPE já calculado anteriormente
# wape_empresa
# wape_modelo

acc_wape_empresa = 1 - wape_empresa
acc_wape_modelo = 1 - wape_modelo

# Montar tabela resumo
tabela_resultados = pd.DataFrame({
    "Métrica":[
        "Volume total previsto",
        "Diferença vs real (%)",
        "MAPE médio por SKU",
        "Acertividade (1-MAPE)",
        "WAPE",
        "Acertividade (1-WAPE)",
        "SKUs com previsão zero",
        "Total de SKUs avaliados"],
    
    "Previsão Empresa":[
        volume_empresa,
        dif_empresa,
        mape_empresa,
        acc_mape_empresa,
        wape_empresa,
        acc_wape_empresa,
        np.nan,
        total_skus],
    
    "Modelo XGBoost":[
        volume_modelo,
        dif_modelo,
        mape_modelo,
        acc_mape_modelo,
        wape_modelo,
        acc_wape_modelo,
        skus_zero_modelo,
        total_skus]})

print("\nTabela resumo de avaliação do modelo:")
print(tabela_resultados)

# Exportar tabela
tabela_resultados.to_csv(
    os.path.join(OUT_DIR, "tabela_resumo_resultados_modelo.csv"),
    sep=";",
    index=False,
    encoding="utf-8-sig")

# Também exportar Excel (melhor para o TCC)
tabela_resultados.to_excel(
    os.path.join(OUT_DIR, "tabela_resumo_resultados_modelo.xlsx"),
    index=False)


#%% GERAR IMAGEM DA TABELA FINAL DE RESULTADOS

# Criar uma cópia para formatar melhor os valores na imagem
tabela_img = tabela_resultados.copy()

# Formatar números para ficar bonito na tabela
for col in ["Previsão Empresa", "Modelo XGBoost"]:
    tabela_img[col] = tabela_img.apply(
        lambda row: f"{row[col]:.2%}" if "Acertividade" in row["Métrica"] 
        or "MAPE" in row["Métrica"]
        or "WAPE" in row["Métrica"]
        or "Diferença vs real (%)" in row["Métrica"]
        else (f"{int(row[col]):,}".replace(",", ".") if pd.notnull(row[col]) else "-"),
        axis=1)

# Substituir NaN por traço
tabela_img = tabela_img.fillna("-")

# Criar figura
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axis("off")

# Criar tabela
tabela_plot = ax.table(
    cellText=tabela_img.values,
    colLabels=tabela_img.columns,
    loc="center",
    cellLoc="center")

# Ajustes visuais
tabela_plot.auto_set_font_size(False)
tabela_plot.set_fontsize(10)
tabela_plot.scale(1, 1.5)

# Estilizar cabeçalho
for (row, col), cell in tabela_plot.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="black")
        cell.set_facecolor("#D6EAF8")  # azul claro
    else:
        cell.set_facecolor("white")

plt.title("Tabela Resumo de Avaliação do Modelo", fontsize=13, weight="bold", pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tabela_resumo_resultados_modelo.png"),
            dpi=300, bbox_inches="tight")
plt.show()

#%% FIM
