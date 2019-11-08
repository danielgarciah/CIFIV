# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:18:08 2019

@author: danie
"""

import pandas as pd
import matplotlib.pyplot as plt
#%%
data = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Parametros")#,index_col="Customer_number")
#%%
data = data.iloc[2:,:]
cuenta = data['A'].value_counts()
cuenta.sum()
plt.plot(cuenta)
plt.xticks(rotation=90)
#%%
work =data.loc[data.A=='Work__']
work_unicos = work.C.value_counts()

#%%
workB=work.B
WorkB_num=pd.value_counts(workB)
#Nos damos cuenta que algunas de las clasificaciones tienen dos resultados o que se usan dos veces

#%%
#leru = data.loc[data.A=="Leru"]
#leru_unicos = leru.C.value_counts()
#leruB=leru.B
#leruB_num=pd.value_counts(leruB)

#%%
#lc = data.loc[data.A=="LC"]

#%%
#life_scienc =data.loc[data.C=="Truven Lifesciences"] 
#Se observa de que categorias sale Truven lifesciences debido a que es el resultado que más se repite

#%%
repetidos = data.B.value_counts()
#%%


#%%







