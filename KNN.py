import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set()

data = pd.read_csv('age_gender.csv') #Wczytanie bazy danych # zdjęcia są monochromatyczne o rozmiarze 48x48
print(data.head()) #wyświetlenie 5 pierwszych wierszy
print(data.describe().T) #wyświetlenie podtawowych danych statytycznych

data = data.drop("img_name", axis=1)  #usunięcie zbędnej kolumny

plt.figure(figsize=(12,8))
sns.histplot(x = data['age']);
#plt.show() #wyświetlenie rozkładu wieku

plt.figure(figsize=(12,8))
sns.countplot(x = data['gender']);
#plt.show() #wyświetlenie wykresu słupkowego płci

data['pixels'] = data['pixels'].map(lambda x: np.array(x.split(' '), dtype=np.float32).reshape(48, 48))

data['pixels'] = data['pixels'].apply(lambda x: x/255) #normalizacja

y = data.drop(["pixels","ethnicity"], axis = 1) #usunięcie kolumn tak, aby zostały kolumny age i gender
X = data.pixels

print(X.head())
print(y.head())


fig, axes = plt.subplots(1, 4, figsize=(20, 10))

#wyświetlenie losowych 4 obrazów
for i in range(4):
    random_face = np.random.choice(len(data))

    age = data['age'][random_face]
    gender = data['gender'][random_face]

    axes[i].set_title('Age: {0}, Gender: {1}'.format(age, gender))
    axes[i].imshow(data['pixels'][random_face])
    axes[i].axis('off')
plt.show()

#wybranie wartości z przedziału
data = data[data["age"]>=30]
data = data[data["age"]<=50]

X = np.array(data['pixels'].tolist())

y1 = data.gender #wybranie etykiety
y2 = data.age

#podzielenie zbioru na zbiór uczący oraz zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=30)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

model = KNeighborsClassifier(n_neighbors = 3) #klasyfikator KNN

model.fit(X_train, y_train) #uczenie modelu
acc = model.score(X_test,y_test) #obliczenie dokładności
print(acc)