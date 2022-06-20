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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\SANJU\Raw\gender_classification_v7.csv")

df.head()

```
![Screenshot_2](https://user-images.githubusercontent.com/94214195/174537006-c12a2fdc-db85-4aba-b57a-551ffa2ba4b1.png)
```
sns.boxplot(data=df,x="forehead_width_cm")
```
![Screenshot_3](https://user-images.githubusercontent.com/94214195/174538564-89b714fc-1adb-4e91-b0ba-e8b44825b263.png)

```
sns.boxplot(data=df,x="forehead_height_cm")
```
![Screenshot_4](https://user-images.githubusercontent.com/94214195/174538599-f23cbe0f-a839-4745-8a7d-a01ce898c287.png)

```
from sklearn.model_selection import train_test_split
x=df.drop(columns="gender")
y=df["gender"]
```
![Screenshot_5](https://user-images.githubusercontent.com/94214195/174538654-269bafb2-9004-4394-a601-0bf57a2e71d6.png)

```
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
```
![Screenshot_6](https://user-images.githubusercontent.com/94214195/174538733-2195e439-b7f6-4f45-bb4d-7fb3c3117620.png)

```

```

```
```

```
```

```
```
