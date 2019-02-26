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



import math
import random

num_points = 20

dimension = 2

lower = 0
upper = 200

num_clusters = 3

cutoff = 0.2

'''define class'''
class Point(object):
  def __init__(self,coords):
    self.coords = coords
    self.n = len(coords)
  
  def __repr__(self):
    return str(self.coords)


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


