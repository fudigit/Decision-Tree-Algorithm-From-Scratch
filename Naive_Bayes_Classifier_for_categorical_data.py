'''
The alroghrim is basically counting frequency by class, and frequency of (data attribute, given a class) 
to calculate required probability distribution for P(Y|X)

# If we are given the join distribution of P(Y|X_1, ..., X_n), then we are done,
# however, it will be a very sparse joint distribution given binary Y, X, with n > 10. Hard to derive from sample data

# Use Bayes rule we can estimate P(X_1, ..., X_n|Y) instead, which is also not cheap without assumption. 
# Note with the assumption of condition independent, P(x_1, x_2|Y) = P(x_1|Y)*(x_2|Y), which is cheaper to calculate.  
'''
# 1. Calculate the class probability: P(Y = y_i)
# 2. Calculate the conditional probability(ies): P(X = x_i| Y = y_i)
# 3. Product them so P(Y|X) ~  P(X = x_i| Y = y_i)* P(Y = y_i), denominator P(X) is ignored 
# since if we want to classify Y, P(X) is just a scaling factor. True P(Y|X) = P(X|Y)*P(Y)/P(X)




# load data
import csv
with open('Go_out_data.csv') as data_csv:
    data_list = list(csv.reader(data_csv))

header = data_list[0]
data = data_list[1:]

# 1. Calculate the class probability: P(Y = y_i)


def value_unique(column, data):
    return set(row[column] for row in data)
    
    

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
class con_prob:
    def __init__(self, y_i, x_i, prob):
        self.y_i = y_i
        self.x_i = x_i
        self.prob = prob
    

def calc_prob(case_i, case_set):
    '''calculate probability of occurrence of a feature value in a given dataset'''
    size = len(case_set)
    count = 0
    for row in case_set:
        if case_i in row:
            '''this does not consider feature value, but if the entire row contain the value'''
            count += 1
    prob = count/size
    return prob

calc_prob('go-out', data)


def calc_con_prob(x_i, y_i, data): 
    '''calculate conditional prob'''
    #filter dataset to subset by y_i
    subset_yi = []
    for row in data:
        '''again, this does not consider feature value, but if the entire row contain the value'''
        if y_i in row:
            subset_yi.append(row)
    #calc prob in subset
    con_prob = calc_prob(x_i, subset_yi)
    return con_prob
    
calc_con_prob('sunny', 'go-out', data)
calc_con_prob('rainy', 'go-out', data)
    

# 3. Product them so P(Y|X) ~  P(X = x_i| Y = y_i)* P(Y = y_i), denominator P(X) is ignored 
# since if we want to classify Y, P(X) is just a scaling factor. True P(Y|X) = P(X|Y)*P(Y)/P(X)

#predict one case
def predict(test_row, y_i):
    prod_condi_prob = 1
    for x_i in test_row:
        prod_condi_prob *= calc_con_prob(x_i, y_i, data)
    class_prob = calc_prob(y_i, data)
    
    bayes_est = prod_condi_prob*class_prob
    return bayes_est

test_data = [row[:2] for row in data]

predict(test_data[3], 'go-out')

#predict all cases:

for test_row in test_data:
    for y_i in value_unique(2, data):
        print(test_row, y_i, predict(test_row, y_i))
    
    
#calculate probability distribution:

def joint_prob_dist(data):
    for i in value_unique(0,data):
        for j in value_unique(1,data):
            for k in value_unique(2,data):
                print('P(L=%s|w = %s, c = %s)' %(k, i, j), '=', predict([i,j],k))


joint_prob_dist(data)

