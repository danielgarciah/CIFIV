# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:28:37 2019

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
data = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Raw data 8XX")#,index_col="Customer_number")
parametros = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Parametros")#,index_col="Customer_number")
#%% Quitar columnas vacías
data=data.iloc[:,:30]
parametros = parametros.iloc[2:,:3]
data=data.drop(["Year","Segment","Quarter","Month","ORDERNUM","Major_Minor","Iot_Name","Imt_Name","OCC","FAMILY","Country","Cost","SRC"],axis=1)
#Se eliminan las columnas de Year, Segment, Quarter, Month, ORDERNUM, Major_Minor, Iot_Name, Imt_Name 
#debido a que para el análisis no son necesarios.

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
data_quality_report = DQR(data)

#%% Usar Customer_number como índice
#data.index=data.Customer_number
data=data.set_index("Customer_number")
#data.rename_axis("Customer_number",axis="index", inplace=True)

#%%Separando columna de resultados
resultados = data['Segmentation Offering']
data = data.iloc[:,:-1]

#%% Función replace text
def replace_text(x, to_replace,replacement):
    try:
        x= x.replace(to_replace,replacement)
    except:
        pass
    return x
#%% Limpiando base de datos
resultados = resultados.apply(replace_text,args=('Truven simpler','Truven Simpler'))
parametros["C"] = parametros["C"].apply(replace_text, args=('Truven simpler','Truven Simpler'))

#%% Analizando los resultados
num_sementation=pd.value_counts(resultados)

#%%Funcion de normalizacion
def normalizar(x):
    return (x-x.mean())/x.std()
#%%Normalizando
#data['Ctrynum']=normalizar(data['Ctrynum'])
data['Maj']=normalizar(data['Maj'])
data['Minor']=normalizar(data['Minor'])

#%%Categorizando 
def categorizar(X):
    val_unic = np.array(X.unique())
    w = np.round(np.linspace(0,1,len(X.unique())),3)
    directorio = dict(zip(val_unic,w))
    return [directorio[item] for item in X]

#%%Aplicando formula
data['Work__'] = categorizar(data['Work__'])
data["OCC_Desc"] = categorizar(data["OCC_Desc"])    
data["PRODID"] = categorizar(data["PRODID"])
data["LDIV"] = categorizar(data["LDIV"])
data["LoB"] = categorizar(data["LoB"]) 
data["Pillar"] = categorizar(data["Pillar"]) 
data["Bmdiv"] = categorizar(data["Bmdiv"]) 
data["SegCalc"] = categorizar(data["SegCalc"])
data["Cust__"] = categorizar(data["Cust__"])
data["Leru"] = categorizar(data["Leru"])

#%%
reporte = DQR(data)

#%%Leyendo hoja de datos nuevamente
data_parametros = pd.read_excel("../Data/M9 M2 AIW segmentation September WD4 - Oscar.xlsx",sheetname="Raw data 8XX")

#%% Limpiar base de datos
data_parametros["Segmentation Offering"] = data_parametros["Segmentation Offering"].apply(replace_text, args=('Truven simpler','Truven Simpler'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000M90','M90'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000M23','M23'))
data_parametros["SegCalc"] = data_parametros["SegCalc"].apply(replace_text, args=('000010',''))
data_parametros["OCC_Desc"] = data_parametros["OCC_Desc"].apply(replace_text, args=('Oncology SaaS','Oncology SaaS Labor'))

#%%Generando nuevo dataframe con las columnas relevantes
data_parametros=data_parametros.iloc[:,:29]
data_parametros=data_parametros.drop(["Year","Customer_number","Segment","Quarter","Month","ORDERNUM","Maj","Minor","Iot_Name","Imt_Name","Country","Cost","SRC","LoB","Bmdiv","Ctrynum"],axis=1)
#%%
reporte_parametros=DQR(data_parametros)

#%%Separando parametros repetidos de los parametros sin repetir
parametros_cuenta = pd.DataFrame(parametros['B'].value_counts())
parametros_r = parametros_cuenta[parametros_cuenta>=2].dropna()
parametros_r['B'] = parametros_r.index
parametros_u = parametros_cuenta[parametros_cuenta==1].dropna()
parametros_u['B'] = parametros_u.index

#%%Creando el DataFrame de categorias y parametros
nombres_categorias = pd.DataFrame(resultados.unique())
nombres_columnas = pd.DataFrame(parametros.A.unique())
parametros_u['A'] = parametros_u['B']
parametros_u['C'] = parametros_u['B']
data_parametros['Offering'] = data_parametros['Work__']

#%%Generando parametros unicos
for val in parametros_u['B']:
    parametros_u['A'][val]=parametros['A'][parametros['B']==val].item()
    parametros_u['C'][val]=parametros['C'][parametros['B']==val].item()
    
#%%Creando Diccionario
dic ={}
parametros = parametros.sort_values(by=['A'])
for i in parametros['A']:
    if i in dic:
        dic[i]=dict(zip(parametros['B'][parametros['A']==i],parametros['C'][parametros['A']==i]))
    else:
        dic[i]=dict(zip(parametros['B'][parametros['A']==i],parametros['C'][parametros['A']==i]))
        
#%%Iterando en el dataframe
'''x=pd.DataFrame()  
for i in data_parametros.iloc[:,:-1].columns:
    if i in dic:
        resultados_prueba = data_parametros.stack().map(dic).unstack()
    else:
        pass'''

#%%
CN = pd.DataFrame()
CUST = pd.DataFrame()
FAMILY = pd.DataFrame()
LC = pd.DataFrame()
LDIV = pd.DataFrame()
LERU = pd.DataFrame()
MM = pd.DataFrame()
OCC = pd.DataFrame()
OCC_D = pd.DataFrame()
PRODID = pd.DataFrame()
PILLAR = pd.DataFrame()
SC = pd.DataFrame()
WORK = pd.DataFrame()

#%%
CN= pd.DataFrame(data_parametros["Contract_Num"].map(dic["Contract_Num"])).dropna()
CUST = pd.DataFrame(data_parametros["Cust__"].map(dic["Cust__"])).dropna()
FAMILY = pd.DataFrame(data_parametros["FAMILY"].map(dic["FAMILY"])).dropna()
LC = pd.DataFrame(data_parametros["LC"].map(dic["LC"])).dropna()
LDIV = pd.DataFrame(data_parametros["LDIV"].map(dic["LDIV"])).dropna()
LERU = pd.DataFrame(data_parametros["Leru"].map(dic["Leru"])).dropna()
MM = pd.DataFrame(data_parametros["Major_Minor"].map(dic["Major_Minor"])).dropna()
OCC = pd.DataFrame(data_parametros["OCC"].map(dic["OCC"])).dropna()
OCC_D = pd.DataFrame(data_parametros["OCC_Desc"].map(dic["OCC_Desc"])).dropna()
PRODID = pd.DataFrame(data_parametros["PRODID"].map(dic["PRODID"])).dropna()
PILLAR = pd.DataFrame(data_parametros["Pillar"].map(dic["Pillar"])).dropna()
SC = pd.DataFrame(data_parametros["SegCalc"].map(dic["SegCalc"])).dropna()
WORK = pd.DataFrame(data_parametros["Work__"].map(dic["Work__"])).dropna()

#%%
CN=CN.rename(columns={"Contract_Num":"Offering"})
CUST=CUST.rename(columns={"Cust__":"Offering"})
FAMILY=FAMILY.rename(columns={"FAMILY":"Offering"})
LC=LC.rename(columns={"LC":"Offering"})
LDIV=LDIV.rename(columns={"LDIV":"Offering"})
LERU=LERU.rename(columns={"Leru":"Offering"})
MM=MM.rename(columns={"Major_Minor":"Offering"})
OCC=OCC.rename(columns={"OCC":"Offering"})
OCC_D=OCC_D.rename(columns={"OCC_Desc":"Offering"})
PRODID=PRODID.rename(columns={"PRODID":"Offering"})
PILLAR=PILLAR.rename(columns={"Pillar":"Offering"})
SC=SC.rename(columns={"SegCalc":"Offering"})
WORK=WORK.rename(columns={"Work__":"Offering"})

#%%
frames = [CN,CUST,FAMILY,LC,LDIV,LERU,MM,OCC,OCC_D,PRODID,PILLAR,SC,WORK]
resultados_prueba2 = pd.concat(frames)
resultados_prueba2 = resultados_prueba2.sort_index()

#%%
#for i in range(620):
#    tmp = resultados_prueba2.loc[i,:] 

#%% 
'''
k = 1
data_parametros_Off = data_parametros.iloc[:,-13:]
for i in data_parametros_Off.columns:
    for j in range(len(data_parametros)):
        #if data_parametros.iloc[j,i] != "nan":
            data_parametros["Offering"] = data_parametros[i]
            k += 1
        else:
            pass'''
'''for i in range(len(data_parametros_Off.columns)):
    for j in range(len(data_parametros)):
        if data_parametros_Off.iloc[j,i] != "nan":
            data_parametros["Offering"] = data_parametros_Off.iloc[j,i]
            k += 1
        else:
            pass
for j in range(len(data_parametros)):
    #if np.isnan(data_parametros_Off.iloc[j,1]):
    if data_parametros_Off[j,1] == float("nan"):
        pass
    else:
        data_parametros["Offering"] = data_parametros_Off.iloc[j,1]
        k += 1

'''
#%%Evaluando resultados
w = np.linspace(0,len(data_parametros),len(data_parametros),dtype=int)
w = np.round(w,0)
resultados_prueba = pd.DataFrame(data_parametros['Offering'])
resultados = pd.DataFrame(resultados)
resultados_prueba = resultados_prueba.set_index(w)
resultados = resultados.set_index(w)
checks = 0
resultados_p = resultados_prueba.dropna()
ar1 = np.array(resultados_p.index)

#%%Checando predicciones correctas, nans e incorrectas
for i in range(len(data_parametros)):
    if resultados_prueba2.iloc[i,0] == resultados.iloc[i,0]:
            checks +=1
nans = resultados_prueba2.isnull().sum().item()
wrongs = len(resultados_prueba2)-nans-checks 

#%%lista con los indices de los parametros erroneos
eval1 = resultados_prueba['Offering'][resultados_prueba['Offering']!=resultados['Segmentation Offering']]
eval2 = eval1.dropna()
eval3 = resultados['Segmentation Offering'][resultados_prueba['Offering']!=resultados['Segmentation Offering']]
eval3_nan = eval3.dropna()
err = eval2.value_counts()       #Error en nuestra predicción
err1 = eval3_nan.value_counts()  #Error en lo que debería de ser.






