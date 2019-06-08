# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:19:49 2019

@author: user
"""

#Premendra Srivastava #Kolkata 



import numpy as np

#read Data4 input file and conver it to a numpy array for easy processing

with open("Data3.csv") as data:
    data_value=data.readlines()
    
list2=[]
for i in range(0,len(data_value)):
    list1=data_value[i].replace("\n","")
    list1=list1.split(',')
    list2.append(list1)
    
#arr is numpy array, feature and class label are assigned from given vector    
arr=np.array(list2)


#smooting is 1 and class count is 2 int his example
smoothing = 1
class_count=len(np.unique(arr[:,-1]))


#function to calculate apriori probability for labels
def prob(label,arr1):
    count=0
    for i in arr1:
        if int(i) == int(label):
            count += 1
    prob = count / len(arr1)
    return (prob)

prior_prob_0=prob(0,arr[:,-1])
prior_prob_1=prob(1,arr[:,-1])


#methode to calculate conditional probabilty, using Laplacian smotthing
#e.g count of 1 for a given lable 1 and then prob using Laplacing smoothing
def cond_prob(label,arr_feature,arr_label):
    prob_arr=[]
    for classes in (1,0):
        count_feature = 0
        count_label = 0
        for a in range(0,len(arr_label)):
            if int(arr_label[a])  == int(classes):
                count_label +=1
                #print(count_label)
                if int(arr_feature[a]) == int(label):
                    count_feature +=1
                    #print(F"counte feature is { count_feature }")
        prob = (count_feature + smoothing) / (count_label + class_count)
        prob_arr.append(prob)
    return prob_arr


#function to calculate independent probability
    
def inde_prob(i):
    prob_arr=np.array([])
    a=i
    for item in range(0,len(a)):
        result=cond_prob(a[item],arr[:,item],arr[:,-1])
        prob_arr=np.append(prob_arr,result)
        
    prob_arr=prob_arr.reshape(8,2)
    return prob_arr

#methode to calculate product of probability of feature for a test element
    
def score(array):
    product = 1
    for j in array:
        product *= j
    return product


#read sample file and iterate over each element 
#  prob_cond_score_0(for label 0) and prob_cond_score_1(for label 1) are condition_prob * apriori_prob, class is defind which ever is max
with open("test3.csv") as test:
    test_value=test.readlines()
#print(test_value[0])
for i in range(0,len(test_value)):
    test_value[i]=test_value[i].replace("\n","")
    test_value[i]=test_value[i].split(',')
    #print(type(test_value[i]))
    #list2.append(list1)
    prob_arr=inde_prob(test_value[i])
    score_1=score(prob_arr[:,0])
    score_0=score(prob_arr[:,1])
    prob_cond_score_1 = score_1 * prior_prob_1
    prob_cond_score_0 = score_0 * prior_prob_0
    #print(prob_cond_score_0)
    if prob_cond_score_1 >= prob_cond_score_0:
        predict_class=1
    else:
        predict_class=0
    
    with open('Premendra_srivastava_3.out','a') as fileout:
        fileout.write(str(predict_class))
        fileout.write(" ")
    
