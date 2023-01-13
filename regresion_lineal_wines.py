# Ejemplo propuesto para el curso de inteligencia de negocios.
# Dataset tomado de https://www.kaggle.com/piyushagni5/white-wine-quality
# Paper de respaldo: [Cortez et al., 2009]
# Ejemplo para la predicción de valores aproximados con regresión lineal.
# Preparado por: Yesid Ospitia Medina

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

#%% Cargar dataset ###################################
dataset="winequality-white.csv"
dataf= pd.read_csv(dataset)
data= dataf.values
x=data[:,:-1]
y=data[:,-1]

#%% Descripción del dataset
#For more information, read [Cortez et al., 2009]. 
# URL: https://www.kaggle.com/piyushagni5/white-wine-quality
#Input variables (based on physicochemical tests):
#1 - fixed acidity
#2 - volatile acidity
#3 - citric acid
#4 - residual sugar
#5 - chlorides
#6 - free sulfur dioxide
#7 - total sulfur dioxide
#8 - density
#9 - pH
#10 - sulphates
#11 - alcohol
#Output variable (based on sensory data):
#12 - quality (score between 0 and 10)

n,d=x.shape  # Dimensión de nuestro dataset.

print("El dataset tiene %d registros, de %d dimensiones" %(n, d))
# visualizar distribución de los datos
dataf.hist(bins=20)
    
#%% Dividir Training/testing
porc_test= 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= porc_test)
print("\t %d datos para training" %x_train.shape[0])
print("\t %d datos para testing" %x_test.shape[0])

#%% Entrenar modelo de Regresión Lineal

# Creación del objeto modelo
modelo= LinearRegression()

# Entrenar el modelo.
modelo.fit(x_train, y_train)

# Realizar predicción
y_train_predict= modelo.predict(x_train) # Con los datos de entrenamiento.
y_test_predict= modelo.predict(x_test)  # Con los datos de prueba.

# Evaluar los scores
error_train= mean_squared_error(y_train_predict, y_train)
print("Error en training: %s" % error_train)
error_test= mean_squared_error(y_test_predict, y_test)
print("Error en testing: %s" % error_test)

#%% Evaluar con datos de Testing
nuevo= 25
x_nuevo=x_test[nuevo, :] # Tomamos uno de los datos.
print(x_nuevo) # Lo miramos.
y_real= y_test[nuevo] # Tomamos la anotación real.
y_predict= modelo.predict(x_nuevo.reshape(1, -1)) # Le preguntamos al modelo su predicción.

# Comparamos.
print("\nNuevo dato:\nEl modelo predice: %f " % (y_predict) )
print("El valor real es: %f" % (y_real))