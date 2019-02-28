# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 22:21:55 2018
@author: Di
"""
'''
try:
    import plotly
    from plotly.graph_objs import Scatter, Scatter3d, Layout
except ImportError:
    plotly = Scatter = Scatter3d = Layout = None
    print('INFO: Plotly is not installed, plots will not be generated.')
'''

'''# pure python inplementation of K-means clustering'''

'''
repeat 2 steps until convergence
step 1: Assignment: for each point, get the distance to each controid, assign point to closest centroid's cluster
step 2: Update: recalculate the centroid of the cluster, it will be the centroid of the new cluster
'''


'''Classes'''
'''define point class'''
# require coords and length

class Point(object):
    '''a point in n dementional space'''
    
    # __init__ special method to take initial value
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)
    # __repr__
    def __repr__(self):
        return str(self.coords)
    
point_a = Point([1])
print(point_a)
point_a.coords.append(2)
print(point_a)

class Cluster(object):
    '''a set of points and their contriod'''
    def __init__(self, points):
        '''Points - A list of point objects'''
        
        # store the points
        self.points = points
        
        # the dimension of the points in this cluster
        self.n = points[0].n
        
        # assert that all points are for the same dimensionality
        for p in points:
            if p.n != self.n:
                raise Exception(p, 'Error, inconsistent dimensionality')
                
        #initial centroid is based of 1 point, later it would be based on multiple points
        self.centroid = self.calculate_centroid()
    
    def __repr__(self):
        '''string repr of cluster object'''
        return str(self.points)
    
    def update(self, points):
        '''
        1. store the old centroid, replace the cluster
        2. calc new centroid
        2. returns the distance between the old centroid and the new after recalc
        Note:
        '''
        # store the old centroid
        old_centroid = self.centroid
        # replace the old cluster with the new cluster
        self.points = points
        # calc the new centroid, this is step is neccesary, otherwise the old centriod remains
        self.centroid = self.calculate_centroid()
        # calc the distance
        shift = get_distance(old_centroid, self.centroid)
        return shift
    

    def calculate_centroid(self):
        '''find the geometric center for a cluster of n dimention points'''
        centriod_coord = []
        num_points = len(self.points)
        # list of all coords in cluster
        coords = [p.coords for p in self.points]
        # calculate mean for each dimension
        for col in range(self.n):
            col_sum = sum([row[col] for row in coords])
            col_mean = col_sum/num_points
            centriod_coord.append(col_mean)
        return Point(centriod_coord)
                
    def get_total_distance(self):
        '''return the sum of all Euclidean distance between each point in cluster and their centroid'''
        dis_total = 0
        for p in self.points:
            dis_total += get_distance(p, self.centroid)
        return dis_total
            

################make testing of Points work!#####################
'''Helper Method'''

import random

#sample each coordinate from the uniform or gaussian distribution
random.gauss(0,1)
random.uniform(0, 10)

def make_random_point(n_dims, lower, upper):
    '''Return a Point object with specified dimensions and bounded value'''
    p = Point([random.uniform(lower, upper) for _ in range(n_dims)])
    return p

# test
make_random_point(2, 0, 10)
#Generate somes points
points1 = [
        make_random_point(2, 0, 10) for _ in range(10)
        ]


points2 =  [
        make_random_point(2, 20, 30) for _ in range(10)
        ]

points3 = [Point([_]) for _ in range(10)]


def get_distance(a,b):
    """
    Suared Euclidean distance between 2 n-dimentional point objects
    """
    sum_square = 0
    p_len = len(a.coords)
    for i in range(p_len):
        sum_square += (a.coords[i] - b.coords[i])**2
    e_dis = sum_square**(1/2)
    return e_dis
# test
a, b = Point([1,1,1]), Point([2,2,2])
get_distance(a, b)
#######################################

# test cluster
#cluster_test = Cluster([Point([1,1,1])])
#print(cluster_test.centroid)
#cluster_test.points.append(Point([2,2,2]))
#print(cluster_test.centroid)
#print(cluster_test.points[0],'1st point')

# test update(self, points)   
#a = Cluster(Points1)
#print(a)
#print(a.update(Points2))
#print(a)

def calculate_error(clusters):
  '''average Euclidean distance between points and centroid'''
  total_EuDis, num_points = 0, 0
  for c in clusters:
    total_EuDis += c.get_total_distance()
    num_points += len(c.points)
  avg_total_dis = total_EuDis/num_points
  return avg_total_dis



'''define kMeans method'''
def kmeans(points, k, cuff):
  
  '''initializing centroids and clusters'''
    # pick out k random points to use as initial centriods
  initial_centroids = random.sample(points, k)
    # create k clusters using the centriods
    # Note: Cluster take list!
    # How to create and store multiple clusters?
  clusters = [Cluster([init_c]) for init_c in initial_centroids]
  
  total_shift = None
  
  while total_shift == None or total_shift > cuff:
  # loop until the centroid stablize
    '''clear cluster'''
    # 1. Clear the initial centroid Point
    # 2. cluster full Due to previous assignment, clearance the list for new Points. (not deleting the list!) 
    for cluster in clusters:
      del cluster.points[:]

    '''assignment step:''' 
    # assign each point to the closest cluster
    for p in points:
      closer_c = None
      closer_dis = None 
      for c in clusters:
        dis = get_distance(p, c.centroid)
        # find min dis and cloest cluster
        if closer_dis == None or dis < closer_dis:
          closer_dis, closer_c = dis, c
      # appending to closer_c is appending to the target cluster!
      closer_c.points.append(p)
      #print(clusters)
    
    '''update step'''
    # calculate the new centroid for each cluster
    total_shift = 0
    # print([c.centroid for c in clusters])
    for c in clusters:
      # print(c.centroid)
      # update calc new centroid, and return distance between old/new. Note the points are already in the cluster!
      total_shift += c.update(c.points)
    print([c.centroid for c in clusters])
    #print(total_shift)
  
  return clusters, total_shift


result = kmeans(points1 + points2, 2, 0.1)[0]



print(calculate_error(result))
