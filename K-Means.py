# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 22:21:55 2018

@author: Di
"""

try:
    import plotly
    from plotly.graph_objs import Scatter, Scatter3d, Layout
except ImportError:
    plotly = Scatter = Scatter3d = Layout = None
    print('INFO: Plotly is not installed, plots will not be generated.')


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
    
a = Point([1])

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
            

# test update(self, points)   
#a = Cluster(Points1)
#print(a)
#print(a.update(Points2))
#print(a)
        

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
Points1 = [
        make_random_point(2, 0, 10) for _ in range(30)
        ]


Points2 =  [
        make_random_point(2, 10, 20) for _ in range(30)
        ]


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





'''define kMeans method'''
def kmeans(points, k, cuff):
    # pick out k random points to use as initial centriods
    initial_centroids = random.sample(points, k)
    
    # create k clusters using the centriods
    # Note: Cluster take list!
    # How to create and store multiple clusters?
    Clusters = [Cluster([init_c]) for init_c in initial_centroids]
    
    # assignment step
    
    
    # update step
    








num_points = 20

dimension = 2

lower = 0
upper = 200

num_clusters = 3

cutoff = 0.2



class Point(object):
  def __init__(self,coords):
    self.coords = coords
    self.n = len(coords)
  
  def __repr__(self):
    return str(self.coords)

'''define cluster class'''
class Cluster(object):
    def __init__(self, points):
        
        if len(points) == 0:
            raise Exception('ERROR: empty cluster')
            
        self.points = points
        
        self.n = points[0].n
        
        for p in points:
            if p.n != self.n:
                raise Exception('ERROR: incosistent dimentions')
                
        self.centroid = self.calculate_centroid()
        
    def __repr__(self):
        return str(self.points)
    
    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        
        if len(self.points) == 0:
            return 0 
        self.centroid = self.calculate_centroid()
        shift = get_distance(old_centroid, self.centroid)
        return shift
        
    def calculate_centroid(self):
        num_points = len(self.points)
        
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList) / num_points for dList in unzipped]
        
        return Point(centroid_coords)
    
    def get_total_distance(self):
        sum_of_distances = 0.0
        for p in self.points:
            sum_of_distances += get_distance(p, self.centroid)

        return sum_of_distances

''' Helper Methods'''
def get_distance(a,b):
    if a.n != b.n:
        raise Exception('Error: non comparable points')
        
    accumulated_difference = 0.0
    for i in range(a.n):
        square_difference = pow((a.coords[i] - b.coords[i]),2)
        accumulated_difference += square_difference
        
    return accumulated_difference

def make_random_point(dimension, lower, upper):
  p = Point([random.uniform(lower,upper) for _ in range(dimension)])
  return p



'''k-means main code'''
points = [
  make_random_point(dimension, lower, upper) for i in range(num_points)
]

#print(points)

#k random points oto use as initial centroids
k = 3

initial_centroids = random.sample(points,k)

#print(initial_centroids)

clusters = [Cluster([p]) for p in initial_centroids]

print(clusters)

loop_counter = 0

'''tesing'''


while True:
    lists = [[] for _ in clusters]
    cluster_count = len(clusters)
    
    loop_counter += 1
    
    for p in points:
         smallest_distance = get_distance(p, clusters[0].centroid)
         print (smallest_distance)
         cluster_index = 0
         
         for i in range(1, cluster_count):
             distance = get_distance(p, clusters[i].centroid)
             print (distance)
             if distance < smallest_distance:
                 smallest_distance = distance
                 cluster_index = i
         lists[cluster_index].append(p)
        
    biggest_shift = 0.0
    

    for i in range(cluster_count):
        shift = clusters[i].update(lists[i])
        biggest_shift = max(biggest_shift, shift)
        
    clusters = [c for c in clusters if len(c.points) != 0]
    
    if biggest_shift < cutoff:
        print('converged after %s iterations' % loop_counter)
        break
    
    
    
print (clusters)
