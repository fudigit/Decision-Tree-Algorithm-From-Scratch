# load data

import csv
with open('breast-cancer-data-categorical.txt') as cancer_csv:
  data_list = list(csv.reader(cancer_csv))


data = [row[1:]+ [row[0]] for row in data_list]

test_data = [row[:9] for row in data]

header = ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat','Class']


# 1. Calculate the class probability: P(Y = y_i)
# Helper function
def value_unique(column, data):
    return set(row[column] for row in data)
    
#--------------------------------------------------------------
# calculate class probability and store in a dictionary, alternative way is to calc on demand
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

print(calc_prob('no-recurrence-events', 9, data))


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
    
print(calc_con_prob('30-39',0,'no-recurrence-events',9, data))

# 3. Product them so P(Y|X) ~  P(X = x_i| Y = y_i)* P(Y = y_i), denominator P(X) is ignored 
# since if we want to classify Y, P(X) is just a scaling factor. True P(Y|X) = P(X|Y)*P(Y)/P(X)

#probability of one case
def prob(test_row, y_i):
    prod_condi_prob = 1
    y_col = len(data[0]) - 1
    for i in range(len(test_row)):
        prod_condi_prob *= calc_con_prob(test_row[i], i, y_i, y_col, data)
    class_prob = calc_prob(y_i, y_col, data)
    
    bayes_est = prod_condi_prob*class_prob
    return bayes_est

print(prob(test_data[0], 'recurrence-events'))


# 4. predict prob all cases:
#y_col = len(data[0]) - 1
#for test_row in test_data:
#    for y_i in value_unique(y_col, data):
#        print(test_row, y_i, prob(test_row, y_i))

#calculate probability distribution:
#def joint_prob_dist(data):
#    for i in value_unique(0,data):
#        for j in value_unique(1,data):
#            for k in value_unique(2,data):
#                print('P(L=%s|w = %s, c = %s)' %(k, i, j), '=',
#                 prob([i,j],k))


#joint_prob_dist(data)


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
      
print(classify(test_data[1]))
  

#prediction of all data
def predict(test_data):
  predictions = []
  for test_row in test_data:
    label = classify(test_row)
    predictions.append(label)
  return predictions

#print(predict(test_data))

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


'''
Class: no-recurrence-events, recurrence-events
   2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
   3. menopause: lt40, ge40, premeno.
   4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44,
                  45-49, 50-54, 55-59.
   5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26,
                 27-29, 30-32, 33-35, 36-39.
   6. node-caps: yes, no.
   7. deg-malig: 1, 2, 3.
   8. breast: left, right.
   9. breast-quad: left-up, left-low, right-up,	right-low, central.
  10. irradiat:	yes, no.
'''
