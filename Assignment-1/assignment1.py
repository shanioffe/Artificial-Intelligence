import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

def warmUpExercise():
    """
        Example function in python which computes the identity matrix
        
        Returns
        -------
        A   : array_like
            The 5x5 identity matrix
    """
    A = np.identity(5) # creates a 5x5 identity matrix
    print(A) # shows matrix
    return A # returns A
warmUpExercise()

# to read comma separated data
data = np.loadtxt('Homework 1/Data/ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]

N = y.size # number of training examples

def plotData(x,y):
    """
    
    """
    fig = pyplot.figure()
    pyplot.plot(x, y, 'ro')
    pyplot.show()
     
plotData(X, y)

def computeError(X, y, w):
    """
    def computeError(X, y, w) :
    Compute cost for linear regression. Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    Parameters
    X : array_like
        The input dataset of shape (N, d+1), where N is the number of examples,
        and d is the number of features. We assume a vector of one's already 
        appended to the features so we have n+1 columns.
        
    y : array_like
        The values of the function at each data point. 
        This is a vector of shape (N, )

    theta : array_like
        The parameters for the regression function. 
        This is a vector of shape (d+1, )
    
    Returns
    J: float
        The value of the regression cost function.
    
    Instructions
        Compute the cost of a particular choice of theta.
        You should set J to the cost.
    """
    # initialize some useful values
    N = y.size
    E = 0
    J = 0
    # use the equation for to calculate error
    # E(w) = (1/N) || Xw - y || ^ 2
    # use the norm function for || Xw - y|| and raise it to power of 2
    for weight in w:    # loops through the array of weights
        # sum the results
        J += norm(np.dot(X, weight) - y) ** 2
    # divide by N and return the calculated error
    return J / N

E = computeError(X, y, w = np.array([0.0, 0.0]))
print ('With w = [0, 0] \nError computed = %.2f' % E)
print ('Expected error value (approximately) 32.07\n')

# further testing of the error function
E = computeError(X, y, w=np.array([-1, 2]))
print ('With w = [-1, 2]\nError computed = %.2f' % E)
print('Expected error value (approximatelv) 54.24')