#from __future__ import print_function

# Toy dataset.
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
# Feel free to play with it by adding more features & examples.
# Interesting note: I've written this so the 2nd and 5th examples
# have the same features, but different labels - so we can see how the
# tree handles this case.
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    #['Yellow', 2, 'Lemon'],
    #['Green', 8, 'Watermelon'],
]
header = ["color", "diameter", "label"]

def unique_vals(rows,col):
  """Find the unique values for a column in a dataset."""
  return set(row[col] for row in rows)

#print(unique_vals(training_data,0))

def class_counts(rows):
  """Counts the number of each type of examples in a dataset."""
  counts = {}
  for row in rows:
    label = row[-1]
    if label not in counts:
      counts[label] = 0
    counts[label] += 1
  return counts

#print(class_counts(training_data))

def is_numeric(value):
  """Test if a value is numeric."""
  return isinstance(value, int) or isinstance(value, float)

#print(is_numeric(7), is_numeric('Red'))

class Question:
  """A Question is used to partition a dataset
    
  This class just records a 'column number' (e.g., 0 for Color) and a
  'column value' (e.g., Green). The 'match' method is used to compare
  the feature value in an example to the feature value stored in the
  question. See the demo below.
  """
  
  def __init__(self, column, value):
    self.column = column
    self.value = value

  def match(self, example):
    #compare the feature value in example to
    #the feature value in this question
    val = example[self.column]
    if is_numeric(val):
      return val >= self.value
    else:
      return val == self.value
  
  def __repr__(self):
    #This is just a helper method to print
    #the question in a readable format
    condition = '=='
    if is_numeric(self.value):
      condition = ">="
    return "Is %s %s %s?" % (
      header[self.column], condition, str(self.value))


######
# Demo:
# Let's write a question for a numeric attribute
#p = Question(1,3)
# How about one for a categorical attribute
#q = Question(0, 'Green')
# Let's pick an example from the trainin set...
#example = training_data[0]
# and see if it matches the question
#q.match(example)
#p.match(example)
######

def partition(rows, question):
  """partition a dataset
  
For each row in the dataset, check if it matches the question. If
so, add it to 'true rows', otherwise, add it to 'false rows'.
  """
  true_rows, false_rows = [], []
  for row in rows:
    if question.match(row):
      true_rows.append(row)
    else:
      false_rows.append(row)

  return true_rows, false_rows

######
# Demo:
# Let's partition the training data based on whether color are Red.
#true_rows, false_rows = partition(training_data, Question(0,'Red'))
#true_rows
#false_rows

def gini(rows):
  """Calculate the Gini Impurity for a list of rows.
  
There are a few different way to do this, I thought this one was the most consise. See:
https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
"""
  
  counts = class_counts(rows)
  impurity = 1
  for lbl in counts:
    prob_of_lbl = counts[lbl]/len(rows)
    impurity -= prob_of_lbl**2
  return impurity

######
# Demo:
# Let's look at some examples to understand how Gini Impurity works.
#
# 1st, we'll look at a dataset with no mixing.
#no_mixing = [['apple'], ['apple']]
#print(gini(no_mixing))

#Now, we'll look at dataset witha 50:50 apples:oranges ratio
# 0.5 meaning there is a 50% chance of mis classifying a random sample drawn from dataset
#some_mixing = [['Apple'],['Oranges']]
#print(gini(some_mixing))

#Now, we'll look at a dataset with many different lables
#lots_of_mixing = [['Apple'],['Orange'],['Grape'],['Grapefruit'],['Blueberry']]
#print(gini(lots_of_mixing))
#print(gini(training_data))

def info_gain(left, right, current_uncertainty):
  """Information info_gain
  
  The uncertainty of the starting node, minus the weighted impurity of 2 child nodes.info_gain
  """
  wl = len(left)/(len(left)+len(right))
  return current_uncertainty - (wl*gini(left) + (1-wl)*gini(right))
  
######
# Demo:
# Calculate the uncertainy of our training data.
#current_uncertainty = gini(training_data)

#How much information do we gain by partitioning on'Green'?
#true_rows, false_rows = partition(training_data, Question(0,'Green'))
#print(info_gain(true_rows,false_rows,current_uncertainty),
#'\n', true_rows, '\n', false_rows)

#What about if we partioned on 'Red' instead?
#true_rows, false_rows = partition(training_data, Question(0,'Red'))
#print(info_gain(true_rows,false_rows,current_uncertainty),
#'\n',true_rows, '\n', false_rows)

def find_best_split(rows):
  """find the best question to ask by iterating over every feature/value
  and calculating the information gain
  """
  best_info_gain = 0 # keep track of the best information gain
  best_question = None # keep track of the feature/value produced it
  current_uncertainty = gini(rows)
  n_features = len(rows[0]) -1 # number of columns

  for feature in range(n_features):
    values = unique_vals(rows, feature) #unique values in the column
    for val in values:
      
      question = Question(feature, val) #creat question for each feature value
      true_rows, false_rows = partition(rows, question) #split data set
      
      #skip this split if it doesn't devide the dataset
      if len(true_rows) == 0 or len(false_rows) == 0:
        continue

      #calc info gain from this split
      gain = info_gain(true_rows, false_rows, current_uncertainty)
      #print(question, gain)
      
      #the best split stays on top, both '>' or '>=' works, use '>=' for demo
      if gain >= best_info_gain:
        best_info_gain, best_question = gain, question

  return best_info_gain, best_question

######
# Demo:
#print(find_best_split(training_data))

class Leaf:
  """A Leaf node classifies data.

This hold a dictionary of class(e.g., {'Apple':2}) -> number of examples(rows)from the training data that reached this leaf.
  """
  def __init__(self,rows):
    self.predictions = class_counts(rows)

class Decision_Node:
  """A Decision Node asks a question.
  
  This holds a reference to the question, and to the 2 child nodes.
  """
  def __init__(self,question,true_branch,false_branch):
    self.question = question
    self.true_branch = true_branch
    self.false_branch = false_branch

def build_tree(rows):
  """Builds the tree.
  
  Rules of recursion: 1) Believe that it works. 2) Start by checkings for the
  base case(no further information gain), 3)Prepare for giant stack traces.
  """
  #print('build tree is called on', rows)
  # Try partition the dataset on each of unique attribute, calc infomation gain
  # return the question that produce the best information info_gain.
  gain, question = find_best_split(rows)
  
  # Base cases, no further infomation gain, since no further questions to ask
  # All examples have same value for each feature, can't split further.
  if gain == 0:
    #print('Leaf_Node',Leaf(rows).predictions)
    return Leaf(rows)

  # If reach here, we've found a useful feature/value to partition on.
  #print("good !", question, 'using this question to split')
  true_rows, false_rows = partition(rows, question)

  # Recursively build the true branch.
  true_branch = build_tree(true_rows)
  
  # Recursively build the false branch.
  false_branch = build_tree(false_rows)

  # Return a Question node.
  # This records the best feature/value to ask at this point, as well as the
  # branches to follow depending on the answer.
  #print(Decision_Node(question, true_branch, false_branch))
  return Decision_Node(question, true_branch, false_branch)

#node = build_tree(training_data)
#print(node.true_branch.true_branch)

def print_tree(node, spacing=""):
  """World's most elegent tree printing function."""
  # Base case: we've reached a Leaf
  if isinstance(node,Leaf):
    print(spacing + "Predict", node.predictions)
    return
  
  #Print the question at this node
  print(spacing + str(node.question))

  # Call this function recursively on the true branch
  print (spacing + '--> True:')
  print_tree(node.true_branch, spacing + "    ")

  #Call this function recursively on the false branch
  print (spacing + '--> False:')
  print_tree(node.false_branch, spacing + "    ")

#print_tree(node)


def classify(row, node):
  """see the 'rules of recursion' above."""
  
  # base case: we've reached a leaf
  if isinstance(node,Leaf):
    return node.predictions

  # Decide whether to follow the true-branch or the false-branch
  # Compare the feeature/value stored in the node, to the example we're considering
  if node.question.match(row):
    return classify(row, node.true_branch)
  else:
    return classify(row, node.false_branch)
  
######
# Demo:
# The tree predicts the 1st row of our training data is an apple with confidence 1 in 1.
my_tree = build_tree(training_data)
print(classify(training_data[1], my_tree))


def print_leaf(counts):
  """A nicer way to print the predictionis at a leaf."""
  total = sum(counts.values())
  probs = {}
  for lbl in counts.keys():
    probs[lbl] = str(counts[lbl]/total*100) +'%'
  return probs

######
# Demo:
# Printing that a bit nicer
print(print_leaf(classify(training_data[1],my_tree)))

######
# Demo:
# On the second example, the confidence is lower
# print(print_leaf(classify(training_data[1],my_tree)))
import operator

if __name__ == '__main__':
  my_tree = build_tree(training_data)
  print_tree(my_tree)

  # Evaluate
  testing_data = [
      ['Green', 3, 'Apple'],
      ['Yellow', 4, 'Apple'],
      ['Red', 2, 'Grape'],
      ['Red', 1, 'Grape'],
      ['Yellow', 3, 'Lemon'],
      #['Green', 8, 'Watermelon'],
    ]
  
  result = []
  for row in testing_data:
    predict = classify(row, my_tree)
    vote = sorted(predict.items(), key = operator.itemgetter(1), reverse = True)
    result.append(vote[0][0])
       
  for row in testing_data:
    print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, my_tree))))
    
def accuracy(testing_data, result):
  success = 0
  size = len(testing_data)
  for i in range(size):
    if testing_data[i][-1] == result[i]:
      success += 1
  score = success/size
  return score

print(result)
print(accuracy(testing_data, result))

###############################################
######
# Titanic
import csv
import random
with open('titanic3.csv') as titanic3:
  dataset = csv.reader(titanic3)
  whole_ds = list(dataset)
  
  #pick feature
  feature_set = []
  header = [whole_ds[0][i] for i in (0,3,4,5,6,8,1)]
  for row in whole_ds[1:]:
    new_row = [row[i] for i in (0,3,4,5,6,8,1)]
    feature_set.append(new_row)

  #convert to float
  for i in range(len(feature_set)):
    for j in (2,3,4,5):
      if feature_set[i][j] =='':
        feature_set[i][j] = 'NA'
      else:
        feature_set[i][j] = float(feature_set[i][j])
  
  #delete 'NA' in age
  feature_set_DeNA =[]
  for row in feature_set:
    if 'NA' not in row:
      feature_set_DeNA.append(row)
      

  #split
  split = 0.8
  random.shuffle(feature_set_DeNA)
  cut = round(len(feature_set_DeNA)*split)
  train_titan = feature_set_DeNA[:cut]
  test_titan = feature_set_DeNA[cut:]


def build_tree(rows, n_split):
  """Builds the tree.
  
  Rules of recursion: 1) Believe that it works. 2) Start by checkings for the
  base case(no further information gain), 3)Prepare for giant stack traces.
  """
  #print('build tree is called on', rows)
  # Try partition the dataset on each of unique attribute, calc infomation gain
  # return the question that produce the best information info_gain.
  gain, question = find_best_split(rows)
  
  # Base cases, no further infomation gain, since no further questions to ask
  # All examples have same value for each feature, can't split further.
  if gain == 0 or n_split == 0:
    #print('Leaf_Node',Leaf(rows).predictions)
    return Leaf(rows)

  # If reach here, we've found a useful feature/value to partition on.
  #print("good !", question, 'using this question to split')
  true_rows, false_rows = partition(rows, question)
  n_split -= 1

  # Recursively build the true branch.
  true_branch = build_tree(true_rows,n_split)
  
  # Recursively build the false branch.
  false_branch = build_tree(false_rows,n_split)


  # Return a Question node.
  # This records the best feature/value to ask at this point, as well as the
  # branches to follow depending on the answer.
  #print(Decision_Node(question, true_branch, false_branch))
  return Decision_Node(question, true_branch, false_branch)


if __name__ == '__main__':
  my_tree = build_tree(train_titan, 3)
  print_tree(my_tree)

  result = []
  result_train = []
  for row in test_titan:
    predict = classify(row, my_tree)
    vote = sorted(predict.items(), key = operator.itemgetter(1), reverse = True)
    result.append(vote[0][0])
  
  for row in train_titan:
    predict = classify(row, my_tree)
    vote = sorted(predict.items(), key = operator.itemgetter(1), reverse = True)
    result_train.append(vote[0][0])


  for row in test_titan[:10]:
    print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, my_tree))))

print(result)
print('accuracy on training set', accuracy(train_titan, result_train), len(result_train))
print(accuracy(test_titan, result), len(result))

print(print_leaf(class_counts(feature_set_DeNA)))
