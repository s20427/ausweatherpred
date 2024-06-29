import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Wczytanie danych z pliku CSV
data = pd.read_csv('./data/weatherAUS.csv')

# Wybór istotnych kolumn
data = data[['Date', 'Location', 'RainTomorrow']]

# Usunięcie wierszy z brakującymi wartościami
data = data.dropna()

# Mapowanie wartości 'RainTomorrow' na wartości binarne: 'No' na 0, 'Yes' na 1
data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Konwersja kolumny 'Date' na typ daty
data['Date'] = pd.to_datetime(data['Date'])
# Dodanie kolumn dla roku, miesiąca i dnia
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# One-hot encoding dla kolumny 'Location'
data = pd.get_dummies(data, columns=['Location'])

# Wybranie wszystkich kolumn poza 'RainTomorrow' i 'Date' jako cechy (features)
columns = data.drop(['RainTomorrow', 'Date'], axis=1).columns

# Podział danych na cechy (X) i etykiety (y)
X = data[columns]
y = data['RainTomorrow']

# Podział danych na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utworzenie i trenowanie modelu RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcja wyników na zestawie testowym
y_pred = model.predict(X_test)

# Obliczenie metryk jakości modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Wyświetlenie wyników
print(f"Dokładność: {accuracy:.2f}")
print(f"Precyzja: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Zapisanie wytrenowanego modelu i kolumn do pliku za pomocą pickle
with open('./rain_prediction_model.pkl', 'wb') as file:
    pickle.dump((model, columns), file)
