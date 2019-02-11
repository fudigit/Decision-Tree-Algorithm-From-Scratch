# -*- coding: utf-8 -*-
"""
@author: Di Fu
"""

import csv
#################
'''I. Handle data'''
def load_csv(filename):
  lines = csv.reader(open(filename))
  dataset = list(lines)
  
  for i in range(1, len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]

  #exclude the headers
  dataset = dataset[1:]
  
  return dataset

filename= 'pima-indians-diabetes.csv'
dataset= load_csv('diabetes.csv')
print('loaded {} with {} rows, and the first row is {}'.format(filename,len(dataset),dataset[0]))

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

trainSet, testSet = split_data(dataset, 0.8)

print('split {} rows into train with {} and test with {}'.format
(len(dataset),len(trainSet),len(testSet)))

############################
''' II. summarize data'''
# 1. Separate data by class
# 2. calculate mean
# 3. calc standard deviation
# 4. summarize dataset
# 5. summarize attribute by class

# 1. Separate data by class
def separateByClass(dataset):
  '''create dictionary to hold label and dataset'''
  separated = {}
  for i in range(len(dataset)):
    vector = dataset[i]
    label = vector[-1]
    if label not in separated:
      separated[label] = []
    separated[label].append(vector)
  return separated

print('---result of separateByClass---\n', separateByClass(dataset[:10]))
# 2. calculate mean
# 3. calc standard deviation
def mean(numbers):
  return sum(numbers)/len(numbers)

def stdev(numbers):
  avg = mean(numbers)
  sum_sqaure = 0
  for x in numbers:
    square_diff = (x-avg)**2
    sum_sqaure += square_diff
  
  length = len(numbers) -1
  variance = (sum_sqaure/length)**(1/2)
  return variance

#separate by class, pick 1 record, calc mean & std
sep_top10 = separateByClass(dataset[:10])
print('pick class 1, row 0, and first 8 columns:\n',sep_top10[1][0][:8])
pick_1 = sep_top10[1][0][:8]
print('calc mean：',mean(pick_1), 'calc stdev', stdev(pick_1))

# 4. summarize dataset by attribute
def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  del summaries[-1]
  return summaries

#print('This is how zip work, by-row -> by-column \n', list(zip(*dataset[:10])))
#can unzip a list with *
print('lenth of summaries is:', len(summarize(dataset)), 
      '\n---II. 4. summarize dataset by attribute---:\n', summarize(dataset))

# 5. summarize attribute by class
def summarizeByClass(dataset):
  '''seprate data into a dictionary with key = label, value = data'''
  separated = separateByClass(dataset)
  summaries_class = {}
  for classValue, class_subset in separated.items():
    #print(classValue,'the subset',class_subset)
    'create new dictionary with key = label, value = (mean,std) for each column'
    summaries_class[classValue] = summarize(class_subset)
  return summaries_class

print('---II. 5. summarize attribute by class----\n',len(dataset),summarizeByClass(dataset))

###############################################
'''III. make prediction'''
# 1. calculate Gaussian Probablity Density Function
# 2. calculate Class Probablities
# 3. Make a prediction
# 4. Make predictions for testSet
# 5. Estiamte Accuracy

# 1. calculate Gaussian Probablity Density Function
import math
'''calc prob of X_new according pdf of normal distribution, refer PDF of Gaussian distribution'''
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

x_test, x_mean, x_stdev = 70,73,3
#mean = 73 this will mass mass up mean(attribute), since mean is now an int obj
print('calculated probability for 1 record\n', calculateProbability(x_test, x_mean, x_stdev))

# 2. calculate Conditional Probablities by Class(This is conditional probability)
'''clacualte P(X_1,...X_n|Y) = P(X_1|Y)*...*P(X_n|Y), for each given value of Y'''
'''Summaries gives parameters per column for each class, we can calculate probility mass for given vector'''
def Cal_ConditionalProb_Class(summaries_class, inputVector):
    probabilities_class = {}
    for classValue, classSummaries in summaries_class.items():
    # take each label and it's attribute parameters
        probabilities_class[classValue] = 1
        # the probability is the product, so set initial value as 1
        for i in range(len(classSummaries)):
        # inpiut vector has the same #of col as classSummaries
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities_class[classValue] *= calculateProbability(x, mean, stdev)
            #print(i, probabilities_class)
    return probabilities_class
  
summaries_class_test = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVec_test = [20, '?']
prob_test = Cal_ConditionalProb_Class(summaries_class_test, inputVec_test)
print('---III. 2. conditional probability using mean, stdev, and x_i for given y_i---\n', prob_test)

summaries_class_data = summarizeByClass(dataset)
inputVec_dataset = dataset[0]
prob_xi_test = Cal_ConditionalProb_Class(summaries_class_data,inputVec_dataset)
print('III. 2. conditional probability on a real x_i and whole data summaries by class\n', prob_xi_test)

# 3. Make a prediction
def predict(summaries, inputVector):
    '''get the label with largest probability'''
    probabilities_class = Cal_ConditionalProb_Class(summaries,inputVector)
    bestLabel, bestProb = None, -1
    for label, prob in probabilities_class.items():
        if prob > bestProb:        
            bestLabel, bestProb = label, prob
    return bestLabel, bestProb

predict_test = predict(summaries_class_test, inputVec_test)
print('III, 3. ---prediction on test data---\n','(labelpr obability):',predict_test)
predict_xi_test = predict(summaries_class_data, inputVec_dataset)
print('prediction on a dataset[i]\n','(labelpr obability):',predict_xi_test)

# 4, make prediction for all testSet, based on summaries of conditional prob
def predict_all(summaries_class,testSet):
    y_pred = []
    for vec in testSet:
        label,prob = predict(summaries_class,vec)
        y_pred.append(label)
    return y_pred

#print(predict_all(summarizeByClass(trainSet),testSet))

#####################################################
# 4.v2 make prediction for all testSet, based on prior*conditional probablity by class
# 4.v2.1
# calculate class probability
'''Prior P(Y = y_i)'''
def calculatePrior(trainSet):
    prior_prob = {}
    data_len = len(trainSet)
    for vector in trainSet:
        label = vector[-1]
        if label not in prior_prob:
            prior_prob[label] = 0
        prior_prob[label] += 1/data_len
    return prior_prob
print('~~~prior probability~~~\n',calculatePrior(trainSet))

# 4.v2.2 make a prodiction based on prior*conditional probablity
def predict_chain(summaries, prior_prob, inputVector):
    '''get the label with largest probability'''
    probabilities_chain = {}
    probabilities_class = Cal_ConditionalProb_Class(summaries,inputVector)
    for key, con_prob in probabilities_class.items():
        #take product of 2 probabilities
        probabilities_chain[key] = con_prob*prior_prob[key] 
    
    bestLabel, bestProb = None, -1
    for label, prob in probabilities_chain.items():
        if prob > bestProb:        
            bestLabel, bestProb = label, prob
    return bestLabel, bestProb

prior_prob_data = calculatePrior(trainSet)
print('~~~predict_chain~~~\n',predict_chain(summaries_class_data, prior_prob_data,inputVec_dataset))

# 4.v2.3, make prediction for all testSet, based on sumarries of conditional prob and prior prob, 
def predict_all_chain(summaries_class,prior_prob, testSet):
    y_pred = []
    for vec in testSet:
        label,prob = predict_chain(summaries_class,prior_prob,vec)
        y_pred.append(label)
    return y_pred
#print('~~~predict_all_chain~~~\n',predict_all_chain(summaries_class_data,prior_prob,testSet))


# 5. Estiamte Accuracy
def getAccuracy(y_true, y_pred):
    count = 0
    data_len = len(y_true)
    for i in range(data_len):
        if y_true[i] == y_pred[i]:
            count += 1
    score = count/data_len
    return score


###############################################
'''IV. main function'''

def NaiveBayesWhole(filename,split_ratio):
    # load file
    dataset = load_csv(filename)
    # split data
    trainSet, testSet = split_data(dataset, split_ratio)
    # summarise data:
    summaries_class = summarizeByClass(trainSet)
    # make prediction based on conditional prob only
    y_pred = predict_all(summaries_class, testSet)
    # make prediction based on both conditional prob and prior
    prior_prob = calculatePrior(trainSet)
    y_pred_chain = predict_all_chain(summaries_class, prior_prob, testSet)
    '''Bonus: make prediction based both conditional prob and prior of whole dataset,
    this is invalid since label of test data is unknown. Used to Demo: would better prior improve result?'''
    prior_prob_whole = calculatePrior(dataset)
    y_pred_chain_whole = predict_all_chain(summaries_class, prior_prob_whole, testSet)
    # calc accuracy or both prodicton method
    y_true = [vec[-1] for vec in testSet]
    score = getAccuracy(y_true, y_pred)
    score_chain = getAccuracy(y_true, y_pred_chain)
    '''Bonus:interstingly, not alway improve'''
    score_chain_whole = getAccuracy(y_true, y_pred_chain_whole)
    print()
    return score, score_chain, score_chain_whole

filename= 'diabetes.csv'
print(NaiveBayesWhole(filename, 0.6))
