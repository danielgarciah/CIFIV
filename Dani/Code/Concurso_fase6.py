# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:19:45 2019

@author: danie
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as sc
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%% Importar los datos
data_parametros = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Raw data 8XX")
parametros = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Parametros")#,index_col="Customer_number")

#%% Quitar columnas vacías
parametros = parametros.iloc[2:,:3]
data_parametros=data_parametros.iloc[:,:30]
data_parametros=data_parametros.drop(["Year","Customer_number","Segment","Quarter","Month","ORDERNUM","Maj","Minor","Iot_Name","Imt_Name","Country","Cost","SRC","LoB","Bmdiv","Ctrynum"],axis=1)

#%% Función replace text
def replace_text(x, to_replace,replacement):
    try:
        x= x.replace(to_replace,replacement)
    except:
        pass
    return x

#%% Limpiar base de datos
data_parametros["Segmentation Offering"] = data_parametros["Segmentation Offering"].apply(replace_text, args=('Truven simpler','Truven Simpler'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000M90','M90'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000M23','M23')) #HAY UN ERROR
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000M23','000000M23')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('260334M23','M23')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000010',''))
data_parametros["OCC_Desc"] = data_parametros["OCC_Desc"].apply(replace_text, args=('Oncology SaaS','Oncology SaaS Labor'))
data_parametros["OCC_Desc"] = data_parametros["OCC_Desc"].apply(replace_text, args=('Oncology SaaS Labor Labor','Oncology SaaS Labor'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('XM90','M90'))
#data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('NFOUNDM90','M90')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ROYCTMM92','M92'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ROYWFGM92','M92'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ROYWFOM92','M92'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000SRMM92','M92'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCAM90','M90'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCJM20','M20'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCJM90','M90')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCAM21','M21'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCJM21','M21'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD39T0M21','M21'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD5CP0M21','M21')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD7BK0M21','M21'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD9QJ0M21','M21'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCAM23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCAM93','M93'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCJM93','M93'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('ALLOCJM23','M23')) 
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OC4SO0M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OC44S0M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OC9OQ0M93','M93'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD7TJ0M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD7VS0M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OD9BR0M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('OE2H50M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('677ONSM91','M91'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('CLO999M95','M95'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('CLOSIMM90','M90'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('TOPM90','M90'))

#%%
data_parametros["Major_Minor"] = data_parametros["Major_Minor"].apply(str)

#%% REPORTE DE CALIDAD DE LOS DATOS
def DQR(data):
    
    #%% Lista de variables que se encuentran en la base de datos
    columns = pd.DataFrame(list(data.columns.values) , columns = ["nombres"], index = list(data.columns.values))
    
    #%% Lista de tipos de variables
    data_types = pd.DataFrame(data.dtypes, columns = ["tipo"])
    
    #%% Lista de datos presentes
    present_values = pd.DataFrame(data.count(), columns = ["Datos presentes"])
    
    #%% Lista de datos faltantes
    miss_values = pd.DataFrame(data.isnull().sum() , columns = ["Datos faltantes"])
    
    #%% Lista de valores Unicos
    unique_values = pd.DataFrame(columns = ["Valores Unicos"])
    for col in list(data.columns.values):
        unique_values.loc[col] = [data[col].nunique()]
        
    #%% Lista de valores minimos
    min_values = pd.DataFrame(columns = ["Valores mínimos"])
    for col in list(data.columns.values):
        try:
            min_values.loc[col] = [data[col].min()]
        except:
            pass
    
    #%% Lista de valores maximos
    max_values = pd.DataFrame(columns = ["Valores maximos"])
    for col in list(data.columns.values):
        try:
            max_values.loc[col] = [data[col].max()]
        except:
            pass
    
    #%% Reporte final
    data_quality_report = columns.join(data_types).join(present_values).join(miss_values).join(unique_values).join(min_values).join(max_values)
    return data_quality_report

#%% Usar mi funcion
data_quality_report_data = DQR(data_parametros)
data_quality_report_parametros = DQR(parametros)

#%%Separando columna de resultados
resultados = data_parametros['Segmentation Offering']
data_parametros = data_parametros.iloc[:,:-1]

#%% Limpiando base de datos
resultados = resultados.apply(replace_text,args=('Truven simpler','Truven Simpler'))
parametros["C"] = parametros["C"].apply(replace_text, args=('Truven simpler','Truven Simpler'))

#%% Vaciando columna de Offering
nan_tmp = np.zeros(620)
nan_tmp = nan_tmp.fill(np.nan)
data_parametros['Offering'] = nan_tmp
del nan_tmp

#%%Creando Diccionario
dic ={}
parametros = parametros.sort_values(by=['A'])
for i in parametros['A']:
    if i in dic:
        dic[i]=dict(zip(parametros['B'][parametros['A']==i],parametros['C'][parametros['A']==i]))
    else:
        dic[i]=dict(zip(parametros['B'][parametros['A']==i],parametros['C'][parametros['A']==i]))
dic.pop('Customer_Name')     
dic["SegCalc"]["M92"] = "Royalties"
dic["SegCalc"]["M95"] = "Oncology & Genomics"
#dic["SegCalc"]["M90"] = "Oncology & Genomics"   #Para checar errores
#dic["SegCalc"]["M91"] = "Oncology & Genomics"

#%% Nombres categorias y columnas
nombres_categorias = pd.DataFrame(resultados.unique())
nombres_columnas = pd.DataFrame(parametros.A.unique())

#%% Evaluando resultados
w = np.arange(0,len(data_parametros))
resultados_prueba = pd.DataFrame(data_parametros['Offering'])
resultados = pd.DataFrame(resultados)
resultados_prueba = resultados_prueba.set_index(w)
resultados = resultados.set_index(w)
del w

#%% Quitando columna que no tenemos
nombres_columnas = nombres_columnas.drop([2]) 

#%%Pasando indices a str para poder evaluar con .loc
res_indx3 = resultados.index
res_indx3 = res_indx3.astype(str)
resultados.index = res_indx3
res_indx4 = data_parametros.index
res_indx4 = res_indx4.astype(str)
data_parametros.index = res_indx4
res_indx = resultados_prueba.index
res_indx = res_indx.astype(str)
resultados_prueba.index = res_indx
data_parametros.index = resultados.index

#%% Utilizando diccionario en base de datos
for i in nombres_columnas[0]:
    X=pd.DataFrame(data_parametros[i].map(dic[i]).dropna())
    ind = X.index
    ind = ind.astype(str)
    X.index = ind
    X = X.rename(columns={i:'Offering'})
    idx = pd.Index(X.index)
    for j in range(len(resultados_prueba)):
        u=str(j)
        if idx.contains(u):
            resultados_prueba.loc[u,'Offering'] =X.loc[u,'Offering']
del u
del X
del i
del j

#%% Análisis del error
t=0
for w in range(len(resultados_prueba)):
    u=str(w)
    try:
        if resultados_prueba.loc[u,'Offering'] == resultados.loc[u,'Segmentation Offering']:
            t+=1
            
    except:
        pass
nans = int(resultados_prueba.isnull().sum())
erroneas = int(len(resultados)-nans-t)
del t

#%% Obteniendo indices de las erroneas para análisis
analisis = []
for w in range(len(resultados_prueba)):
    u=str(w)
    if resultados_prueba.loc[u,'Offering'] != resultados.loc[u,'Segmentation Offering']:
        analisis.append(u)
del u
del w
#%% Solución de errores
data_parametros_erroneas = data_parametros.loc[analisis,:]
data_parametros_erroneas["Offering"] = resultados_prueba["Offering"]
data_parametros_erroneas["Offering real"] = resultados["Segmentation Offering"]

#%% Agregamos nuestros resultados a base de datos
data_parametros["Offering"] = resultados_prueba["Offering"]

#%% Importamos de nuevo la base de datos
data = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Raw data 8XX")
data=data.iloc[:,:30]
data.index = resultados.index

#%% Le agregamos nuestra propuesta de resultado
data["Segmentation Offering"] = resultados_prueba["Offering"]

#%% Exportamos a un excel
data.to_excel("../Data/propuesta.xls")





