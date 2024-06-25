# 1. Import the necessary libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 2. Load train and test datasets

# import the test data set
df_test = (pd.read_csv("hw3pt2/data/SalaryData_Test.csv"))
# import the train data set
df_train = (pd.read_csv("hw3pt2/data/SalaryData_Train.csv"))
# store the first 10 rows in test
df_10 = df_test.head(10)
# print the 10 rows
print(df_10)

# # prints the dataset size
# print(df.size)