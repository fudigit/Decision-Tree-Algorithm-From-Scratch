import csv

'''Handle data'''
def load_csv(filename):
  lines = csv.reader(open(filename))
  dataset = list(lines)
  
  for i in range(1, len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]

  return dataset

filename= 'pima-indians-diabetes.csv'
dataset= load_csv('diabetes.csv')
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


''' summarize data'''
# 1. Separate data by class
# 2. calculate mean
# 3. calc standard deviation
# 4. summarize dataset
# 5. summarize attribute by dataset

# 1. Separate data by class
def separateByClass(dataset):
  separated = {}
  for i in range(len(dataset)):
    vector = dataset[i]
    label = vector[-1]
    if label not in separated:
      separated[label] = []
    separated[label].append(vector)
  return separated

print(separateByClass(dataset[:3]))
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
a = [1,2,3,4,5,6,7,8,9]
print(mean(a))
print(stdev(a))

# 4. summarize dataset
def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  del summaries[-1]
  return summaries

summary = summarize(dataset[:5])
print(len(summary), summary)
#* can unzip a list
#print(list(zip(*[[1,2,3],[4,5,6]])))

# 5. summarize attribute by datase
def summarizeByClass(dataset):
  separated = separateByClass(dataset)
  summaries = {}
  for classValue, class_subset in separated.items():
   summaries[classValue] = summarize(class_subset)
  return summaries

print(summarizeByClass(dataset[:10]))

''' make prediction'''
# 1. calculate Gaussian Probablity Density Function
# 2. calculate Class Probablities
# 3. Make a prediction
# 4. Estiamte Accuracy

# 1. calculate Gaussian Probablity Density Function
import math
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

x = 70
mean = 73
stdev = 3
print(calculateProbability(x, mean, stdev))

# 2. calculate Class Probablities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
  

summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print(probabilities)

# 3. Make a prediction
def predict(summaries, inputVector):
  probabilities = calculateClassProbabilities(summaries,inputVector)
  bestLabel, bestProb = None, -1
  for classValue, probability in probabilities.items():
    if bestLabel is None or probability > bestProb:
      bsetProb = probability
      bestLabel = classValue
  return bestLabel

summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print(result)

# 4. Estiamte Accuracy
