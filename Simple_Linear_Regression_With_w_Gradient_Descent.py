# Goal: discover a set of coefficients that results in good predictions y: y = b0 + b1 * x1 + b2 *x2 + ...

# there are 2 objective functions which we can use to derive the gradient by take the partial derivative
# 1. Maxmizing Conditional liklihood
# 2. Minimizing sum of squared error
# Drivation:
#http://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/07_GenDiscr2_2-4-2015-ann.pdf

import csv
'''Handle data'''
# load
def load_csv(filename):
  with open(filename) as csv_file:
    lines = csv.reader(csv_file, delimiter = ';')
    dataset = list(lines)
# convert string to float
  for i in range(1, len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
# break into header and acutual data
  header = list(dataset)[0]
  dataset = list(dataset)[1:]
  return dataset, header

# test load_csv()
filename= 'winequality-white.csv'
dataset, header= load_csv(filename)
print('loaded {} with {} rows'.format(filename,len(dataset)))

'''split data'''
import random
def split_data(dataset, split_ratio):
  trainSize = round(len(dataset)*split_ratio)
  trainSet = []
  testSet = []
  random.shuffle(dataset) 
  trainSet = dataset[:trainSize]
  testSet= dataset[trainSize:]
  return trainSet, testSet
# test split_data()
trainSet, testSet = split_data(dataset, 0.8)
print('split {} rows into train with {} and test with {}'.format
(len(dataset),len(trainSet),len(testSet)))

'''Make a prediction function'''
# based the functional form of multivariate linear regression
def predict(row, coefficent):
  yhat = coefficent[0]
  coef_len = len(coefficent)
  for i in range(1, coef_len):
    yhat += coefficent[i] * row[i -1]
  return yhat 

# test predict()
test_1 = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
coef_1 = [0.4, 0.8]
for row in test_1:
	yhat = predict(row, coef_1)
	print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))

'''stochastic gradient descent'''
# error(t) = y_true - prediction(t)
#b_i(t+1) = b_i(t) - learning_rate*(-error(t)*x_i(t))
#b_0 = b_0 - learning_rate*(-error)
def coefficients_update_sgd(trainSet, l_rate, n_epoch):
  coef_len = len(trainSet[0]) + 1 - 1
  coef = [0 for _ in range(coef_len)]
  for epoch in range(n_epoch):
    sum_error = 0
    for row in trainSet:
      y_hat = predict(row, coef)
      error = row[-1] - y_hat
      #print(error, coef)
      sum_error += error**2
      coef[0] = coef[0] - l_rate * (-error)
      for i in range(1, coef_len):
        coef[i] = coef[i] - l_rate*(-error*row[i-1])
    print('epoch=%d,l_rate=%.3f, sum_error=%.2f' % (epoch,l_rate, sum_error))
      
  return coef
#NOTE# Step size is not suitbale if feature value is not scaled, meaning, for feature A it could be too tiny of a step but for feature B it could be way too large. Thus, weight for B could blow up, so the optimization will never converge
#test_2 = trainSet[:5]
#print(coefficients_update_sgd(test_2,0.005,5))

'''scale data'''
def scale_dataset(dataset):
  # get minmax for each column
  dataset_zipped = zip(*dataset)
  minmax = []
  for row in dataset_zipped:
    minmax.append([min(row),max(row)])
  # scale value for each feature into 0 t0 1
  for row in dataset:
    for i in range(len(row) - 1):
      # not scaling y, the last column
      row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])

  return dataset
#test_2_scaled = scale_dataset(test_2)
#print(coefficients_update_sgd(test_2_scaled,0.01,20))

'''split dataset into k folds'''
# and return all k folds in a list, from the k fold we can from trainSet and testSet in later algorithm_valuation()
from random import randrange

# using randrange from randomly forming folds
def cross_validation_split1(dataset, k_fold):
  fold_len = int(len(dataset)/k_fold)
  # use int since we want a lower bound
  data_split = list()
  dataset_copy = list(dataset)
  # what if we don't make a copy?
  for i in range(k_fold):
    # the goal is to have k folds
    fold = list()
    count_pop = 0
    while count_pop < fold_len:
      # forming 1 fold, remove rec has been used
      data_len = len(dataset_copy)
      index = randrange(data_len)
      fold.append(dataset_copy.pop(index))
      count_pop += 1
    data_split.append(fold)

  return data_split
# test
# data_split = cross_validation_split1(trainSet[:10],3)

# use random shuffle, instead of randrange
def cross_validation_split2(dataset, k_fold):
  # use int since we want a lower bound
  fold_len = int(len(dataset)/k_fold)
  data_split = list()
  random.shuffle(dataset)
  dataset_copy = list(dataset)
  
  for i in range(k_fold):
    fold = list()
    count_pop = 0

    while count_pop < fold_len:
      fold.append(dataset_copy.pop(0))
      count_pop += 1
    
    data_split.append(fold)  
  return data_split

#print(cross_validation_split2(trainSet[:10],3))

'''make prediction on all testSet'''
def predict_all(testSet, coef):
  predictions = []
  for row in testSet:
    pred = predict(row, coef)
    predictions.append(pred)
  return predictions


'''calculate root mean squared error'''
def rmse_metric(y_true, y_pred):
  squared_error = 0
  y_len = len(y_true)
  for i in range(y_len):
    squared_error += (y_true[i] - y_pred[i])**2
    rmse = (squared_error/y_len)**(1/2)
  return rmse

'''evaluate algorithm'''
def evaluate_algorithm(dataset, k_fold, algorithm, *arg):
  scores = list()
  data_split = cross_validation_split2(dataset,k_fold)
  
  for fold in data_split:
    # form a train and test set for each fold
    data_split_copy = list(data_split)
    data_split_copy.remove(fold)
    # .remove() removes the frist same 'fold' it encountered
    trainSet = sum(data_split_copy, [])
    # sum(iterable,[]) meaning the start is now [], not the default 0, when iterable = [[a],[b]], it now calculate [] + [a] + [b]]
    testSet = list(fold)

    coef = algorithm(trainSet, *arg)
    
    y_pred = predict_all(testSet, coef)
    y_true = [row[-1] for row in testSet]
    
    rmse = rmse_metric(y_true, y_pred)
    scores.append(rmse)

  score_mean = sum(scores)/len(scores)
  return scores, score_mean

dataset_scaled = scale_dataset(dataset)
print(evaluate_algorithm(dataset_scaled, 5, coefficients_update_sgd, 0.001, 20))

#NOTE#, the rmse is 6 times of the rmse with scaled y, since max - min = 6
#dataset_zipped = zip(*dataset)
#minmax = []
#for row in dataset_zipped:
#  minmax.append([min(row),max(row)])
#print(minmax)
