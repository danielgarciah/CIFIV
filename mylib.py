# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:44:20 2019

@author: Ariadna
"""

class mylib:
    def dqr(data):
        import pandas as pd
        #%%lista de variables de la base de datos 
        columns = pd.DataFrame(list(data.columns.values),
                           columns = ['Nombres'],
                           index = list(data.columns.values))
    
        #%%lista de tipos de datos 
        data_types = pd.DataFrame(data.dtypes, columns=['Data_Types'])
    
        #%%lista de datos perdidos
        missing_values = pd.DataFrame(data.isnull().sum(), columns=['Missing'])
    
        #%%lista de valores unicos
        unique_values = pd.DataFrame(columns=['unique values'])
        for col in list(data.columns.values):
            unique_values.loc[col] = [data[col].unique()]
        
        #%% lista de los datos presentes
        present_values = pd.DataFrame(data.count(), columns= ['Present_Values'])
    
        #%% lista de valores minimos
        min_values = pd.DataFrame(columns = ['Min'])
        for col in list(data.columns.values):
            try: 
                min_values.loc[col]=[data[col].min()]
            except:
                pass
        
        #%%lista de valores maximos
        max_values = pd.DataFrame(columns = ['Max'])
        for col in list(data.columns.values):
            try: 
                max_values.loc[col]=[data[col].max()]
            except:
                pass
        
        return columns.join(data_types).join(missing_values).join(unique_values).join(present_values).join(min_values).join(max_values)
