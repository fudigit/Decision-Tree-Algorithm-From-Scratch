
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



# load data
import csv
with open('Go_out_data.csv') as data_csv:
    data_list = list(csv.reader(data_csv))

header = data_list[0]
data = data_list[1:]
test_data = [row[:2] for row in data]

# 1. Calculate the class probability: P(Y = y_i)
# Helper function
def value_unique(column, data):
    return set(row[column] for row in data)
    
#--------------------------------------------------------------
# calculate class probability and store in a dictionary, alternative way
# is to calc on demand
def class_prob(data,column):
    prob_Y = {}
    #labels = value_unique(data, column)
    for row in data:
        label = row[column]
        if label not in prob_Y:
            prob_Y[label] = 0
        prob_Y[label] += 1/len(data)
    return prob_Y
        
# 2. Calculate the conditional probability(ies): P(X = x_i| Y = y_i)

# Try to creat a class for con_prob. 
# The hope is con_prob.prob(y_i, x_i) give the conditional prob
class con_prob:
    def __init__(self, y_i, x_i, prob):
        self.y_i = y_i
        self.x_i = x_i
        self.prob = prob
#-----------------------------------------------------------------   

def calc_prob(case_i, case_col, case_set):
    '''calculate probability of occurrence of a feature value in a given dataset'''
    size = len(case_set)
    count = 0
    for row in case_set:
        if case_i in row[case_col]:
            '''this does not consider feature value, but if the entire row contain the value'''
            count += 1
    prob = count/size
    return prob

print(calc_prob('go-out', 2, data))


def calc_con_prob(x_i, x_col, y_i, y_col, data): 
    '''calculate conditional prob'''
    #filter dataset to subset by y_i
    subset_yi = []
    for row in data:
        '''again, this does not consider feature value, but if the entire row contain the value'''
        if y_i in row[y_col]:
            subset_yi.append(row)
    #calc prob in subset
    con_prob = calc_prob(x_i, x_col, subset_yi)
    return con_prob
    
#calc_con_prob('sunny', 'go-out', data)

    

# 3. Product them so P(Y|X) ~  P(X = x_i| Y = y_i)* P(Y = y_i), denominator P(X) is ignored 
# since if we want to classify Y, P(X) is just a scaling factor. True P(Y|X) = P(X|Y)*P(Y)/P(X)

#probability of one case
def prob(test_row, y_i):
    prod_condi_prob = 1
    y_col = len(test_row)
    for i in range(len(test_row)):
        prod_condi_prob *= calc_con_prob(test_row[i], i, y_i, y_col, data)
    class_prob = calc_prob(y_i, y_col, data)
    
    bayes_est = prod_condi_prob*class_prob
    return bayes_est
#predict(test_data[3], 'go-out')


# 4. predict prob all cases:
for test_row in test_data:
    for y_i in value_unique(2, data):
        print(test_row, y_i, prob(test_row, y_i))

#calculate probability distribution:
def joint_prob_dist(data):
    for i in value_unique(0,data):
        for j in value_unique(1,data):
            for k in value_unique(2,data):
                print('P(L=%s|w = %s, c = %s)' %(k, i, j), '=',
                 prob([i,j],k))


joint_prob_dist(data)


# predicion of one case
def classify(test_row):
  label = None
  max_prob = 0
  y_col = len(data[0]) - 1
  #find the y_i with max probability
  for y_i in value_unique(y_col, data):
    yi_prob = prob(test_row, y_i)
    if yi_prob > max_prob:
      max_prob, label = yi_prob, y_i
  
  return label
      
print(classify(test_data[3]))
  

#prediction of all data
def predict(test_data):
  predictions = []
  for test_row in test_data:
    label = classify(test_row)
    predictions.append(label)
  return predictions

print(predict(test_data))

#accuracy
y_true = []
y_col = len(data[0])-1
for row in data:
  y_true.append(row[y_col])


def accuracy(predictions, y_true):
  count_true = 0
  n_rows = len(data)
  for i in range(n_rows):
    if predictions[i] == y_true[i]:
      count_true += 1
  return count_true/n_rows

print(accuracy(predict(test_data), y_true))


      
  
  
