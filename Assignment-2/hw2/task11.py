import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load our data, including the 3 features and the target variable
df = (pd.read_csv("Homework 2/hw2/data/pokemon.csv", 
                  usecols=["name", "defense", "attack", "speed", "legendary"], 
                  index_col = 0).reset_index()
      )

# our 3 dimensions
X = df[['attack', 'defense', 'speed']]
# our target variable
y = df['is_legendary']

# balance the dataset between 70 legendary pokemon and 70 non-legendary pokemon
legendary_df = df[df['is_legendary'] == 1].sample(70, random_state=1)
non_legendary_df = df[df['is_legendary'] == 0].sample(70, random_state=1)
balanced_df = pd.concat([legendary_df, non_legendary_df])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_df[['attack', 'defense', 'speed']], balanced_df['is_legendary'], test_size=0.2, random_state=1)

# train the model to perform our logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# make our prediction
y_pred = model.predict(X_test)

# calculate the accuracy of our prediction when compared with the testing data
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# get the optimal weights
weights = model.coef_[0]
print(f"Optimal weights: {weights}")