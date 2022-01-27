import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('age_gender.csv') #Wczytanie bazy danych # zdjęcia są monochromatyczne o rozmiarze 48x48

data['pixels'] = data['pixels'].map(lambda x: np.array(x.split(' '), dtype=np.float32).reshape(48, 48))
data['pixels'] = data['pixels'].apply(lambda x: x/255) #normalizacja

data = data[data["age"]>=30]
data = data[data["age"]<=50]

X = np.array(data['pixels'].tolist())

y1 = data.gender #wybranie etykiety

#podzielenie zbioru na zbiór uczący oraz zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=30)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

model = LogisticRegression()
model.fit(X_train, y_train) #uczenie modelu
acc = model.score(X_test,y_test) #obliczenie dokładnoci
print(acc)
