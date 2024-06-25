import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import plotting as libs
import torch
import plotly.io as pio

df = (pd.read_csv("Homework 2/hw2/data/pokemon.csv", 
                  usecols=["name", "defense", "attack"], 
                  index_col = 0).reset_index()
      )

df_10 = df.head(10)

x = df_10['defense']
y = df_10['attack']

print(df_10)

defense_range = df['defense'].max() - df['defense'].min()
attack_range = df['attack'].max() - df['attack'].min()


print("Range of the defense values: ")
print(defense_range)
print("Range of the attack values: " )
print(attack_range)
print("Number of data samples: ")
print(df.count())
y_hat = 10  + 0.5 * x
lin_reg = 0.9 * x
mse = libs.mean_squared_error(y, y_hat)

slopes = np.arange(0.4, 1.65, 0.05)
mse = pd.DataFrame ({"slope": slopes,
                        "MSE": [mean_squared_error (y, m * x) for m in slopes]})
print(mse)

fig = libs.plot_pokemon(x, y)
fig = libs.plot_pokemon(x, y_hat)
fig = libs.plot_pokemon(x, lin_reg)
fig = libs.plot_grid_search(x,y, slopes, mean_squared_error)
#fig.show()


# fig.write_image('Homework 2/hw2/figs')
pio.write_image(fig, 'Homework 2/hw2/figs/x_y.png', format='png')

def gradient(x, y, w):
      return 2 * (x * (w * x - y)).mean()

def gradient_descent(x, y, w, alpha, epsilon= 2e-4, max_iterations=5000, print_progress = 10):
      print(f"Iteration 0. w = {w:.2f}")
      iterations = 1 
      delta_w = 2 * epsilon
      
      while abs(delta_w) > epsilon and iterations <= max_iterations:
            g = gradient(x,y,w)
            delta_w = alpha * g
            # our code
            w -= delta_w
            if iterations % print_progress ==  0:
                  print(f"Iteration {iterations}. w = {w:.2f}")
            iterations += 1
            
      print("Terminated")
      print(f"Iteration {iterations - 1}. w = {w:.2f}")

gradient_descent(x, y, w=0.5, alpha=0.00001)

fig = libs.plot_gradient_descent(x, y, w=0.5, alpha=0.00001)
pio.write_image(fig, 'Homework 2/hw2/figs/gradient_descent3.png', format='png')

slopes = np.arange(0, 2.05, 0.05)
intercepts = np.arange(-30, 31, 2)
fig = libs.plot_grid_search_2d(x, y, slopes, intercepts)
#fig.show()
fig.write_html("mse_gridsearch_2d.html")

def gradient2(x, y, w):
      grad_w0 = (1/len(x)) * 2 * sum(w[0] + w[1] * x - y)
      grad_w1 = (1/len(x)) * 2 * sum(x * (w[0] + w[1] * x - y ))
      return np.array([grad_w0, grad_w1])

print(gradient2(x, y, w=[10, 0.5]))

def gradient_descent2(x, y, w, alpha, epsilon= 2e-4, max_iterations=5000, print_progress = 10):
      print(f"Iteration 0. Intercept = {w[0]:.2f}. Slope = {w[1]:.2f}")
      iterations = 1 
      delta_w = np.array([epsilon, epsilon])
      
      while abs(delta_w.sum()) > epsilon and iterations <= max_iterations:
            g = gradient2(x,y,w)
            delta_w = alpha * g
            # our code
            w -= delta_w
            if iterations % print_progress ==  0:
                  print(f"Iteration {iterations}. Intercept = {w[0]:.2f}. Slope = {w[1]:.2f}")
            iterations += 1
            
      print("Terminated")
      print(f"Iteration {iterations - 1}. Intercept = {w[0]:.2f}. Slope = {w[1]:.2f}")
      
gradient_descent2(x, y, w=[10, 0.5], alpha=0.00001)

# TASK 8
df = (pd.read_csv("Homework 2/hw2/data/pokemon.csv", index_col=0, usecols=["name", "defense", "legendary"]).reset_index()
      )
leg_ind = df["legendary"] == 1
df = pd.concat(
      (df[~leg_ind].sample(sum(leg_ind), random_state=123), df[leg_ind]),
      ignore_index=True,
).sort_values(by='defense')

print(df.head(10))

x = StandardScaler().fit_transform(df[['defense']]).flatten()
y = df['legendary'].to_numpy()
fig = libs.plot_logistic(x, y)
pio.write_image(fig, 'Homework 2/hw2/figs/lg.png', format='png')

def sigmoid(x, w, output="soft", threshold=0.5):
      p = 1/ (1 + np.exp(-x @ w))
      if output == "soft":
            return p
      elif output == "hard":
            return np.where(p > threshold, 1, 0)
      
ones = np.ones((len(x), 1))
X = np.hstack((ones, x[:, None]))
w = [-1, 3]

y_soft = sigmoid(X, w)
y_hard = sigmoid(X, w, "hard")
fig = libs.plot_logistic(x, y, y_soft, threshold=0.5)
pio.write_image(fig, 'Homework 2/hw2/figs/lg1.png', format='png')

def accuracy(y, y_hat):
      return(y_hat == y).sum() / len(y)

print(accuracy(y, y_hard))

def logistic_loss(w, X, y):
      return -(y * np.log(sigmoid(X, w)) + (1 - y) * np.log(1 - sigmoid(X, w))).mean()

def logistic_loss_grad(w, X, y):
      return(X.T @ (sigmoid(X, w) - y)) / len(X)

w_opt = minimize(logistic_loss, np.array([-1, 1]), jac=logistic_loss_grad, args=(X, y)).x
print(w_opt)

lr = LogisticRegression(penalty=None).fit(x.reshape(-1,1), y)
print(f"w0: {lr.intercept_[0]:.2f}")
print(f"w1: {lr.coef_[0][0]:.2f}")

y_soft = sigmoid(X, w_opt)
fig = libs.plot_logistic(x, y, y_soft, threshold=0.5)
pio.write_image(fig, 'Homework 2/hw2/figs/lg3.png', format='png')

# # features to select
# X = df[['attack', 'defense', 'speed']]
# # y is the target variable
# y = df['lengendary']

df = (pd.read_csv("Homework 2/hw2/data/pokemon.csv", 
                  usecols=["name", "defense", "attack", "speed"], 
                  index_col = 0).reset_index()
      )

df['legendary'] = np.where(df['name'].str.contains('Legendary'), 1, 0)

# get correct categories from data
X = df[['defense', 'attack', 'speed']].values
y = df['legendary'].values

# set the initial weights
w_initial = np.zeros(X.shape[1])

# find the optimal weights using logistic regression
w_opt = minimize(logistic_loss, w_initial, jac=logistic_loss_grad, args=(X, y)).x
print("Optimal Weights:")
print("Attack:", w_opt[0])
print("Defense:", w_opt[1])
print("Speed:", w_opt[2])


