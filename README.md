# GENDER CLASSIFICATION

## Algorithm
1. Import libraries required.
2. Load dataset through local or drive link.
3. Train the datasets.
4. Train the model with neural networks.
5. Compile the code.
6. Fit the model and check accuracy.
7. Evaluate performance on test dataset.

## Program
```
/*
Program to implement Gender Classification.
Developed by   :  S. Sanju
RegisterNumber :  212219040137
*/
```

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\SANJU\Raw\gender_classification_v7.csv")

df.head()

```
![Screenshot_2](https://user-images.githubusercontent.com/94214195/174537006-c12a2fdc-db85-4aba-b57a-551ffa2ba4b1.png)
```
df.info()
```
![Screenshot_3](https://user-images.githubusercontent.com/94214195/174538564-89b714fc-1adb-4e91-b0ba-e8b44825b263.png)

### To check outliers in forehead_width_cm and forehead_height_cm
```
sns.boxplot(data=df,x="forehead_width_cm")
```
![Screenshot_4](https://user-images.githubusercontent.com/94214195/174538599-f23cbe0f-a839-4745-8a7d-a01ce898c287.png)

```
sns.boxplot(data=df,x="forehead_height_cm")
```
![Screenshot_5](https://user-images.githubusercontent.com/94214195/174538654-269bafb2-9004-4394-a601-0bf57a2e71d6.png)

### split data
```
from sklearn.model_selection import train_test_split
x=df.drop(columns="gender")
y=df["gender"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
```
### Logistic Regression
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)
lr.fit(X_train,y_train)
```
![Screenshot_6](https://user-images.githubusercontent.com/94214195/174538733-2195e439-b7f6-4f45-bb4d-7fb3c3117620.png)

```
y_pred=lr.predict(X_test)
y_pred

```
![Screenshot_6 - Copy](https://user-images.githubusercontent.com/94214195/174539137-b101242f-9836-4685-bde3-b4456141d875.png)

### To check accurecy model on test data using confusion_matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
![Screenshot_6 - Copy (2)](https://user-images.githubusercontent.com/94214195/174539194-dc1d51f1-0c40-4f87-a615-afcdff7bb80d.png)

```
sns.heatmap(cm,annot = True, fmt = ".0f")
```
![Screenshot_7](https://user-images.githubusercontent.com/94214195/174539235-733ff2ea-a01c-4283-bcad-7fca06141a1d.png)

```
print("model_accuracy =",accuracy_score(y_test, y_pred))
```
![Screenshot_8](https://user-images.githubusercontent.com/94214195/174539534-5fba982d-f1b4-42a2-816c-0d34dad82277.png)

### An other way to check accurecy model test and train data
```
print("model_accuracy_on_train_data = ",lr.fit(X_train,y_train).score(X_train,y_train))
```
![Screenshot_8 - Copy](https://user-images.githubusercontent.com/94214195/174539558-46762783-ce17-4e5d-912f-6be7b9c62853.png)


```
print("model_accuracy_on_train_data = ",lr.fit(X_train,y_train).score(X_test,y_test))
```
![Screenshot_8 - Copy (2)](https://user-images.githubusercontent.com/94214195/174539572-a5e520e8-a9cb-4773-bd9d-1df05c52da05.png)

### KNN model
```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train,y_train)
```
![Screenshot_8 - Copy (3)](https://user-images.githubusercontent.com/94214195/174539618-5e560097-ba91-4d6e-8aaf-93905312bae0.png)


```
y_pred = knn.predict(X_test)
y_pred
```
![Screenshot_8 - Copy (4)](https://user-images.githubusercontent.com/94214195/174539663-74b2b934-02f0-40a0-ab24-c35f309e110b.png)


```
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
![Screenshot_8 - Copy (5)](https://user-images.githubusercontent.com/94214195/174539685-1a46f286-b1e6-45ae-8e5d-0dd6196d861b.png)

```
sns.heatmap(cm,annot = True, fmt = ".0f")
```
![Screenshot_9](https://user-images.githubusercontent.com/94214195/174539763-218b873c-82fe-4b2f-8113-7b8177a9f793.png)

```
print("model_accuracy =",accuracy_score(y_test, y_pred))
```
![Screenshot_10](https://user-images.githubusercontent.com/94214195/174539825-779fbdeb-2e35-4a34-a128-9e309daa280b.png)

```
score_list = []
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors =k)
    knn.fit(X_train, y_train)
    score_list.append(knn.score(X_test, y_test))

plt.plot(range(1,15), score_list)
plt.xlabel("k values")
plt.ylabel("Accuracy")
plt.show()
```

## Output
![download (1)](https://user-images.githubusercontent.com/94214195/174539868-16565b79-a4e0-425f-ade2-860121e6c1de.png)

