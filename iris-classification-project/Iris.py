import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Loading the data
df = sns.load_dataset('iris')

# Visualisation of the data
df['species'].value_counts().plot(kind='bar')
sns.pairplot(df, hue='species')

print(df.describe())
print(df.info())
plt.show()

# df.species = df.species.map({'setosa': 1, 'versicolor': 2, 'virginica': 3})

# Encoding the labels
le = LabelEncoder()

X = df.drop(['species'], axis='columns')
y = le.fit_transform(df['species'])

scaler = StandardScaler()

# Splitting the data - 80% data for training and 20% data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scaling the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
model = SVC(kernel='rbf', C=1)
model.fit(X_train, y_train) # fitting part

# Evaluation
score_test = model.score(X_test, y_test)
score_train = model.score(X_train, y_train)
print(f"Test score: {score_test:.2f}")
print(f"Train score: {score_train:.2f}")
print(f"CV score: {cross_val_score(model, X, y, cv=5)}")

# First try:
# what_class = model.predict([[6, 3, 5, 2]])# if the flower has those parameters, what is the class of the flower?
# print(what_class) # the class is 3, so it's a virginica
# what_class_2 = model.predict([[4.8, 3.0, 1.5, 0.3]])
# print(what_class_2)

# After better implementation:
sample1 = scaler.transform([[6, 3, 5, 2]])
sample2 = scaler.transform([[4.8, 3.0, 1.5, 0.3]])

prediction1 = le.inverse_transform(model.predict(sample1))
prediction2 = le.inverse_transform(model.predict(sample2))

print(prediction1)
print(prediction2)


