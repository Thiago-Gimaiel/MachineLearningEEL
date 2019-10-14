#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importação das bibliotecas necessárias
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Neste exemplo irei usar os dados do Cancer de Mama
dados_cancer = load_breast_cancer()


# In[3]:


def retornaResultadosModeloKNN_Classificacao(random_state, quantidade, dados, respostas):
    #Divisão entre observações de teste e observações de treino
    X_train, X_test, y_train, y_test = train_test_split(dados, respostas, random_state = random_state)
    # Vetores de armazenamento dos resultados de teste e de treino
    quantidade_k = range(1,quantidade)
    res_teste = []
    res_treino = []
    
    # loop das classificações

    for i in quantidade_k:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)

        # Adicionando os valores dos resultados a seus respectivos vetores
        res_treino.append(knn.score(X_train, y_train))
        res_teste.append(knn.score(X_test, y_test))
        
    return quantidade_k, res_treino, res_teste


# In[4]:


# Analisando o dataset do dados_cancer
print(dados_cancer.keys())


# In[12]:


# Print de alguns resultados
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN_Classificacao(1, 20, dados_cancer['data'], dados_cancer['target'])
i = 2
print("Treino {} : {}".format(i, res_treino[i]))
print("Teste {} : {}".format(i, res_teste[i]))


# In[7]:


# Plot dos gráficos com os seguintes randoms 1, 5, 20, 550
dados = dados_cancer['data']
respostas = dados_cancer['target']

legendas = ["Treino", "Teste"]
quantidade = 20
rand = 1
f, axarr = plt.subplots(2,2)
plt.setp(axarr, xticks=np.arange(0,20, step=1))
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN_Classificacao(rand, quantidade, dados, respostas)
axarr[0,0].plot(res_treino)
axarr[0,0].plot(res_teste)
axarr[0,0].grid(True)
axarr[0,0].set_title("Rand 1")
axarr[0,0].legend(legendas)

# Rand 5
rand = 5
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN_Classificacao(rand, quantidade, dados, respostas)
axarr[0,1].plot(res_treino)
axarr[0,1].plot(res_teste)
axarr[0,1].grid(True)
axarr[0,1].set_title("Rand 5")
axarr[0,1].legend(legendas)

#Rand 20
rand = 20
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN_Classificacao(rand, quantidade, dados, respostas)
axarr[1,0].plot(res_treino)
axarr[1,0].plot(res_teste)
axarr[1,0].grid(True)
axarr[1,0].set_title("Rand 20")
axarr[1,0].legend(legendas)

#Rand 550
rand = 550
quantidade_k, res_treino, res_teste = retornaResultadosModeloKNN_Classificacao(rand, quantidade, dados, respostas)
axarr[1,1].plot(res_treino)
axarr[1,1].plot(res_teste)
axarr[1,1].grid(True)
axarr[1,1].set_title("Rand 550")
axarr[1,1].legend(legendas)

#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.rcParams["figure.figsize"] = [12,12]

plt.show()


# In[ ]:




