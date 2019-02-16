
# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
 
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)
 
# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


import csv

'''Handle data'''
def load_csv(filename):
  lines = csv.reader(open(filename))
  dataset = list(lines)
  
  for i in range(1, len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  
  header = list(dataset)[0]
  dataset = list(dataset)[1:]

  return dataset, header

filename= 'pima-indians-diabetes.csv'
dataset, header= load_csv('diabetes.csv')
print('loaded {} with {} rows'.format(filename,len(dataset)))

'''split data'''
import random
def split_date(dataset, split_ratio):
  trainSize = round(len(dataset)*split_ratio)
  trainSet = []
  testSet = []
  random.shuffle(dataset) 
  trainSet = dataset[:trainSize]
  testSet= dataset[trainSize:]
  return trainSet, testSet

trainSet, testSet = split_date(dataset, 0.8)

print('split {} rows into train with {} and test with {}'.format
(len(dataset),len(trainSet),len(testSet)))


import math

def predict(row, coefficients):
  #print('row=',row,'coefficient=',coefficients)
  yhat = coefficients[0]
  for i in range(len(row)-1):
    yhat += coefficients[i+1]*row[i]
  # the if statement is to avoid overflow
  if yhat >= 0:
    prob = 1/(1 + math.exp(-yhat))
  elif yhat < 0:
    prob = 1 - 1/(math.exp(yhat)+ 1)
  #print('yhat', yhat, 'prob', prob)
  return prob

data_test = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

coef = [-0.406605464, 0.852573316, -1.104746259]


#for row in data_test:
#  yhat = predict(row, coef)
#  print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))

'''stochastic gradient discent'''
# 1. loop each epoch
# 2. loop each row in trainSet for each epoch
# 3. loop each coef for each row of trainset for each epoch
# also record error for each epoch
def coef_update_sgd(trainSet, l_rate, n_epoch):
  coef_len = len(trainSet[0]) # constant coef + 1, label - 1
  coef = [0 for _ in range(coef_len)]
  for epoch in range(n_epoch):
    #print(coef)
    error_sum = 0
    for row in trainSet:
      prob_hat = predict(row, coef)
      error = row[-1] - prob_hat
      error_sum += error**2
      #print('prob_hat', prob_hat, 'error', error)
      coef[0] = coef[0] + l_rate*error*prob_hat*(1-prob_hat)
      for i in range(1, coef_len):
        coef[i] = coef[i] + l_rate*error*prob_hat*(1-prob_hat)*row[i-1]
        
    print('epoch=%f, l_rate=%f, error_sum=%f.3' % (epoch, l_rate, error_sum))

  return coef

#print(coef_update_sgd(data_test, 0.5, 100))



''' apply logistic regression on the PIMA Diabete Dataset'''

'''rescale data'''
# if data is not rescaled, the yhat = b0 + b1*X becomes a big integer, causings the logit function gives 0 or 1 probablity 
# get all the minimum and maximum of each col of the dataset
def get_minmax(dataset):
  n_col = len(dataset[0])
  minmax = []
  for i in range(n_col):
    col_value = [row[i] for row in dataset]
    min_col = min(col_value)
    max_col = max(col_value)
    minmax.append([min_col, max_col])
  return minmax
  
print(get_minmax(dataset),'minmax')
minmax = get_minmax(dataset)


import copy
#scaling
def scale_dataset(dataset, minmax):
  # deep copy works for 2D data
  scaled_set = copy.deepcopy(list(dataset[:]))
  n_col = len(scaled_set[0]) - 1 
  n_row = len(scaled_set)
  for j in range(n_col):
    for i in range(n_row):
      scaled_set[i][j] = (scaled_set[i][j] - minmax[j][0])/(minmax[j][1] - minmax[j][0])
  return scaled_set

data_scaled = scale_dataset(dataset, minmax)


''' train logistic regression model using stochastic gradient discent
# use k-fold cross validation to estimate the performance of unseen data'''
# 1. split data set
# 2. for each fold, train (stochastic gd) model on the k-1 folds, and test on the k fold. Get the average accuracy of all k rounds of training. Goal is to pick the coefficent that minize result of CV 


# split a dataset into k folds
from random import randrange
def cross_validation_split(dataset,n_folds):
  dataset_split = list()
  dataset_copy = list(dataset)
  fold_size = int(len(dataset)/n_folds)
  
  for i in range(n_folds):
    fold = list()
    while len(fold) < fold_size:
      index = randrange(len(dataset_copy))
      fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
  print ('record used',len(fold)*n_folds)
  return dataset_split

#print(cross_validation_split(data_test,3))


# calc accuracy score
def accuracy_metric(y_true, y_pred):
  count = 0
  for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
      count += 1
  score = count/len(y_true)
  return score

# evaluate algorithm using cross validation

def evaluate_algorithm(trainSet, n_folds):

# split data into k folds
  folds = cross_validation_split(trainSet, n_folds)
  scores = []
  for fold in folds:
    trainSet_cv = list(folds)
    trainSet_cv.remove(fold)
    testSet_cv = copy.deepcopy(fold)
    coef = logistic_train(trainSet_cv, 0.1, 100)
    y_pred = predict_all(testSet_cv,coef)
    y_true = [row[-1] for row in testSet_cv]
    accuracy = accuracy_metric(y_true, y_pred)
    scores.append(accuracy)
  return scores

# fit model using logistic regression on the kth fold, and predict on the kth fold
  

#print(evaluate_algorithm(trainSet, logistic_train, 5))


def logistic_train(trainSet, l_rate, n_epoch):
  minmax = get_minmax(trainSet)
  trainScaled = scale_dataset(trainSet, minmax)
  coef = coef_update_sgd(trainScaled, l_rate, n_epoch)
  return coef

coef = logistic_train(trainSet, 0.1, 100)

def predict_all(testSet, coef):
  minmax = get_minmax(testSet)
  testScaled = scale_dataset(testSet,minmax)
  y_pred = []
  for row in testScaled:
    pred = round(predict(row, coef))
    y_pred.append(pred)
  return y_pred

#y_pred = [round(row) for row in predict_all(testSet, coef)]

y_true = [row[-1] for row in testSet]
#print(accuracy_metric(y_true, y_pred))



scores = evaluate_algorithm(trainSet, 5)

