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
num_segmentation=pd.value_counts(resultados)

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

#%% Vaciando columna de Offering
nan_tmp = np.zeros(620)
nan_tmp = nan_tmp.fill(np.nan)
data_parametros['Offering'] = nan_tmp

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
dic.pop('Customer_Name')     
dic["SegCalc"]["M92"] = "Royalties"
dic["SegCalc"]["M95"] = "Oncology & Genomics"
#dic["SegCalc"]["M90"] = "Oncology & Genomics"   #Para checar errores
#dic["SegCalc"]["M91"] = "Oncology & Genomics"


#%%Evaluando resultados
w = np.arange(0,len(data_parametros))
resultados_prueba = pd.DataFrame(data_parametros['Offering'])
resultados = pd.DataFrame(resultados)
resultados_prueba = resultados_prueba.set_index(w)
resultados = resultados.set_index(w)
checks = 0
resultados_p = resultados_prueba.dropna()
ar1 = np.array(resultados_p.index)

#%%
nombres_columnas = nombres_columnas.drop([9])

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
data.index = resultados.index
#%%
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

#%%analisis de error
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

#%%Obteniendo indices de las erroneas para analisis
analisis = []
for w in range(len(resultados_prueba)):
    u=str(w)
    if resultados_prueba.loc[u,'Offering'] != resultados.loc[u,'Segmentation Offering']:
        analisis.append(u)
        
#%%Obteniendo nombre de parametros de conflicto
param_conflicto = []
for w in range(len(resultados_prueba)):
    u=str(w)
    if resultados_prueba.loc[u,'Offering'] != resultados.loc[u,'Segmentation Offering']:
        param_conflicto.append(data_parametros.loc[u,'SegCalc'])
param_conflicto = pd.DataFrame(param_conflicto)

param_conflicto2 =param_conflicto[0].map(dic['SegCalc']).dropna()
indices_conflicto = param_conflicto2.index.astype(str)
param_conflicto = data_parametros.loc[indices_conflicto,'SegCalc']
param_conflicto = pd.DataFrame(param_conflicto)
param_conflicto_SegCalc=param_conflicto.drop_duplicates()
#%% Haciendo correlacion para dar prediccion
Truven_Simpler = data[resultados['Segmentation Offering']=='Truven Simpler']
Truven_Simpler = Truven_Simpler.drop(columns=['LC','Contract_Num'])
Onc_and_Gen = data[resultados['Segmentation Offering']=='Oncology & Genomics']
Onc_and_Gen = Onc_and_Gen.drop(columns=['LC','Contract_Num'])
Lifesciences = data[resultados['Segmentation Offering']=='Lifesciences']
Lifesciences = Lifesciences.drop(columns=['LC','Contract_Num'])
Truven_Lifesciences = data[resultados['Segmentation Offering']=='Truven Lifesciences']
Truven_Lifesciences = Truven_Lifesciences.drop(columns=['LC','Contract_Num'])
Eclinical = data[resultados['Segmentation Offering']=='Eclinical']
Eclinical = Eclinical.drop(columns=['LC','Contract_Num'])
Softlayer = data[resultados['Segmentation Offering']=='Softlayer']
Softlayer = Softlayer.drop(columns=['LC','Contract_Num'])
Royalties = data[resultados['Segmentation Offering']=='Royalties']
Royalties = Royalties.drop(columns=['LC','Contract_Num'])
Truven = data[resultados['Segmentation Offering']=='Truven']
Truven = Truven.drop(columns=['LC','Contract_Num'])
Explorys_Lifesciences = data[resultados['Segmentation Offering']=='Explorys Lifesciences']
Explorys_Lifesciences = Explorys_Lifesciences.drop(columns=['LC','Contract_Num'])
#%%Creando el dataframe con las medias
columnas = Royalties.columns
#%%
data_medias = pd.DataFrame(columns=columnas,index=['Truven_Simpler','Onc_and_Gen','Lifesciences','Truven_Lifesciences','Eclinical',
                                                   'Softlayer','Royalties','Truven','Explorys_Lifesciences'])
#%% Solución de errores
data_parametros_erroneas = data_parametros.loc[analisis,:]
data_parametros_erroneas["Offering"] = resultados_prueba["Offering"]
data_parametros_erroneas["Offering real"] = resultados["Segmentation Offering"]


    
    