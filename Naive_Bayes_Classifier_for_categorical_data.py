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
