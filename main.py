import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
plt.show()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop('target', axis='columns')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Dokładność modelu: {accuracy:.4f}")

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz Pomyłek')
plt.show()

index = 0
sample_image = digits.images[index]
sample_input = digits.data[index].reshape(1, -1)

predicted_label = model.predict(sample_input)[0]
true_label = digits.target[index]

plt.imshow(sample_image, cmap='gray')
plt.title(f'Predykcja: {predicted_label} (Prawidłowa: {true_label})')
plt.axis('off')
plt.show()
