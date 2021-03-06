import csv
import random
import operator

###load data###
def load_data(data_file, split):
  # read into a dataset
  with open(data_file, 'rt') as csv_file:
    data_obj = csv.reader(csv_file)
    data_set = list(data_obj)
    #for row in data_obj:
    #  data.append(row)
  
  #numerical
  for x in range(len(data_set)):
    for y in range(4):
      data_set[x][y] = float(data_set[x][y])

  # split data
  random.shuffle(data_set)
  cut = round(len(data_set)*split)
  
  for row in data_set[:cut]:
    trainSet.append(row)
  for row in data_set[cut:]:
    testSet.append(row)
  
  return
#test
#trainSet = []
#testSet = []
#test = load_data('iris.data.txt', split = 0.8)
#print(len(trainSet),len(testSet))

###distance###
def eucl_distance(pointA,pointB):
  sum_square = 0
  for i in range(len(pointA)-1):
    sum_square += (pointA[i] - pointB[i])**2
    sqrt_sum_square = sum_square**(1/2)
  return sqrt_sum_square
#test
#print(eucl_distance([2,2,2,2,'k'],[4,4,4,4,'k']))

#k nearest neighbors
def k_nearest_neighbor(test_rec,trainSet,k):
  distance_list = []
  #calc the distance with all points, create dataset stores results
  for x in trainSet:
    distance = eucl_distance(test_rec,x)
    new_rec = x[:]
    #new_rec = x, !!!note, this will change trainSet, since new_rec and x are referencing the same object 
    new_rec.append(distance)
    distance_list.append(new_rec)
  #rank distance_list and select top k
  distance_list.sort(key = operator.itemgetter(5))
  knn = distance_list[:k]

  return knn 
#test
#print(k_nearest_neighbor(testSet[0], trainSet[:10],2))

#majority vote
def majority_vote(knn):
  vote = {}
  for instance in knn:
    if instance[4] not in vote:
      vote[instance[4]] = 1
    else:
      vote[instance[4]] += 1

  
  vote_list = []
  for key, value in vote.items():
    vote_list.append([key,value])
    
  vote_list.sort(key = operator.itemgetter(1),reverse = True)
  
  winner = vote_list[0][0]

  return winner
#test
#test4 = (majority_vote(k_nearest_neighbor(testSet[0], trainSet[:10],10)))
##print(test4)

def calc_accuracy(testSet, prediction):
  count = 0
  for i in range(len(testSet)):
    if testSet[i][4] == prediction[i]:
      count += 1
    else:
      pass
  accuracy = count/len(testSet)
  return accuracy
#test
#print(calc_accuracy(testSet,['Iris-setosa' for i in range(30)]))

#make prediction
trainSet = []
testSet = []
prediction = []
def main(data_file, split, k):
  load_data(data_file, split)
  for x in testSet:
    knn = k_nearest_neighbor(x, trainSet, k)
    winner = majority_vote(knn)
    prediction.append(winner)
  
  accuracy = calc_accuracy(testSet, prediction)
  return len(testSet), accuracy
# result
# choose k as add to avoid tie, otherwise result may different from
# sklearn result
print(main('iris.data.txt', split = 0.8, k = 9))


#use existing knn package
from sklearn.neighbors import KNeighborsClassifier as knn
#trainSet
X_train = [rec[:4] for rec in trainSet]
y_train = [rec[4] for rec in trainSet]
#testSet
X_test = [rec[:4] for rec in testSet]
y_true = [rec[4] for rec in testSet]
#modeling
neigh = knn(n_neighbors = 9)
neigh.fit(X_train, y_train)
knn_package_pre = neigh.predict(X_test)
#result
print(calc_accuracy(testSet,list(knn_package_pre)))


#for i in range(len(prediction)):
  #print(prediction[i] + ' ' + knn_package_pre[i])
