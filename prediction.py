import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the dataset
dataset = pd.read_csv(r"C:\Users\dell\Desktop\task3\Iris.csv")

# Display basic information about the dataset
print(dataset.head(10))
print(dataset.shape)
print(dataset.columns)
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())
print(dataset['Species'].unique())
print(dataset['Species'].value_counts())

# Visualize the data
sns.pairplot(dataset, hue='Species')
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 5))
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=dataset, hue='Species', ax=ax1, s=300, marker='o')
sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm', data=dataset, hue='Species', ax=ax2, s=300, marker='o')
plt.show()

# Violin plots
sns.violinplot(y='Species', x='SepalLengthCm', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='SepalWidthCm', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalLengthCm', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalWidthCm', data=dataset, inner='quartile')
plt.show()

# Pie chart
colors = ['#66b3ff', '#ff9999', 'green']
dataset['Species'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, explode=[0.08, 0.08, 0.08])
plt.figure(figsize=(7, 5))
plt.show()

# Heatmap excluding non-numeric columns
numeric_columns = dataset.select_dtypes(include=[np.number])
sns.heatmap(numeric_columns.corr(), annot=True, cmap='CMRmap')
plt.show()

# Machine Learning part
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = dataset.loc[:, features].values
y = dataset.Species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)

y_pred = decisiontree.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

def report(model):
    preds = model.predict(X_test)
    print(classification_report(preds, y_test))
    confusion_mtx = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(confusion_mtx)

report(decisiontree)
print(f'Accuracy: {round(score*100, 2)}%')
