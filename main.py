from quickda.clean_data import *
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

datos = pd.read_csv('breast-cancer-wisconsin.csv')


ProfileReport(datos).to_file("report.html")

# Eliminamos las filas duplicadas
datos = clean(datos, "duplicates")

# Eliminamos la primera fila ya que no nos interesa
datos = clean(datos, "dropcols", ["id"])

# Estandarizamos las columnas
datos = clean(datos, "standardize")

datos = clean(datos, "replaceval",
              columns=["class"],
              to_replace=2,
              value="benign")

datos = clean(datos, "replaceval",
              columns=["class"],
              to_replace=4,
              value="malignant")

datos = clean(datos, "replaceval",
              columns=[],
              to_replace="?")

datos = clean(datos, method="dropmissing")


def codify_bind(original_dataframe: pd.DataFrame, variables: list) -> pd.DataFrame:

    for variable in variables:

        dummies = pd.get_dummies(original_dataframe[[variable]])

        new_dataframe = pd.concat([original_dataframe, dummies], axis=1)

        new_dataframe = new_dataframe.drop([variable], axis=1)

        original_dataframe = new_dataframe

    return original_dataframe

datos = codify_bind(datos, ["class"])



# Escalamiento de los datos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

edatos = pd.DataFrame(scaler.fit_transform(datos), columns=datos.columns)

edatos.describe()

datos.to_csv("clean_data.csv", index=False)
edatos.to_csv("scaled_data.csv", index=False)