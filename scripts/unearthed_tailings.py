#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:42:50 2018

@author: kevin
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
df = pd.read_csv("../data/train.csv")

# Remove the timestamp column
df.drop("timestamp", axis=1, inplace=True)

# Drop the columns that have no data (from looking at the csv)
for col in df:
  # Drop rows contianing 'Bad Input' 
  search = ["Bad Input", "Scan Off", "I/O Timeout"]
  df = df[~df[col].isin(search)]
  
  if df[col][0] == "No Data":
    # axis=1 refers to column not row!
    df.drop(col, axis=1, inplace=True)
    
# Print to confirm dimensions are smaller
print(df.shape)

# Split into features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Now split a training and test set from the data
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)