# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:19:02 2019

@author: user
"""

#Premendra Srivastava #Kolkata 



import numpy as np

#read Data4 input file and conver it to a numpy array for easy processing

with open("Data4.csv") as data:
    data_value=data.readlines()
    
list2=[]
for i in range(0,len(data_value)):
    list1=data_value[i].replace("\n","")
    list1=list1.split(',')
    list2.append(list1)
    
#arr is numpy array, feature and class label are assigned from given vector    
arr=np.array(list2)

train_classes=arr[:,-1]
train_features=arr[:,:8]

#Euclidean distance function 

def eucli_dist(each_object,new_object):
    distance=0
    for i in range(0,len(each_object)):
        distance += (int(each_object[i]) - int(new_object[i]))**2
    distance = distance**0.5
    return distance

#methode to get k nearest neighbours by using euclidean distance methode
    
def get_nearest_neighbours(train_features_vector, new_object, k):
    distances = []
    for train_object in train_features_vector:
        distance = eucli_dist(train_object, new_object)
        distances.append(distance)
    return np.argsort(distances[:k])


#methode to find maximum votes for a label  
def get_majority_class(knn):
    votes = {}
    for neighbour in knn:
        neighbour_class = train_classes[neighbour]
        if neighbour_class in votes:
            votes[neighbour_class] += 1
        else:
            votes[neighbour_class] = 1
    max_votes = sorted(votes.items(),key = lambda x: (x[1],x[0]),reverse=True)[0][0]
    return max_votes


#output operation, test file test4 are processed and itereted over each element

with open("C:/Users/user/Downloads/test4.csv") as test:
    test_value=test.readlines()


list_test=[]
for i in range(0,len(test_value)):
    list1=test_value[i].replace("\n","")
    list1=list1.split(',')
    list_test.append(list1)
    
#iteration over each element of test file
    
for i in range(0,len(list_test)):
    knn=get_nearest_neighbours(train_features,list_test[i],5)
    predicted_class = get_majority_class(knn)
    #print(knn)
    with open('premendra_srivastava_4.out','a') as fileout:
        fileout.write(predicted_class)
        fileout.write(" ")
    print(predicted_class)
    
