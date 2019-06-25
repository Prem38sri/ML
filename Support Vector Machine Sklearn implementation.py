##################################################
## 1. Load breast cancer dataset from sklearn
## 2. Split the data as 70:30 as train and test data
## 3. Fit the train data into SVM model with diffferent kernels
##    and bar plot the accuracy of different SVM model with the test data
## 4. Fit the above training dataset into a SVM model with ploynomial kernel
##    with varying degree and plot the accuracy wrt. degree of ploynomial kernel with the test data
## 5. Define a custom kernel K(X,Y)=K*XY'+theta where k and theta are constants
## 6. Use the custom kernel and report the accuracy with the given train and test dataset
##################################################

##################################################
## Basic imports
## You are not required to import additional module imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
from sklearn.preprocessing import PolynomialFeatures
###################################################

###################################################
## the method loads breast cancer dataset and returns
## the dataset and label as X,Y
def load_data(): 
	data_set = datasets.load_breast_cancer()
	X=data_set.data
	y=data_set.target
	return X,y
###################################################

###################################################
## this method takes train and test data and different 
## svm models and fit the train data into svm models and 
## do bar plot using sns.barplot() of different svm model 
## accuracy. You need to implement the model fitting and 
## bar plotting in this method.
def svm_models(X_train, X_test, y_train, y_test,models):
	## write your own code here
    #list comprehension to append kernel names and arruracy score in below 2 lists respectevely
    kernel_models= []
    score_list= []
    #list comprehension to append kernel_models list and calculate score after training(fit)
    [(kernel_models.append(model.kernel),model.fit(X_train,y_train),score_list.append(model.score(X_test,y_test))) for model in models]

    print(F" kernel methodes are { kernel_models } and their respective accuracy is { score_list}")
    print(score_list)
    score_list=np.array(score_list)
    
    #Bar chart of Accuracy for different models
    
    plt.figure(figsize=(8,6))
    plt.bar(kernel_models,score_list,0.5,align='center',color='red')
    plt.xlabel('Kernel Methods')
    plt.ylabel('Accuracy')
    plt.show()
    
    return
###################################################

###################################################
## this method fits the dataset to a svm model with 
## polynomial kernel with degree varies from 1 to 3 
## and plots the execution time wrt. degree of 
## polynomial, you can calculate the elapsed time 
## by time.time() method
def ploy_kernel_var_deg(X_train, X_test, y_train, y_test):
	## write your own code here
    X_train = ((X_train - X_train.mean()) // X_train.std())
    X_test = ((X_test - X_test.mean()) // X_test.std())
    y_train = ((y_train - y_train.mean()) // y_train.std())
    y_test = ((y_test - y_test.mean()) // y_test.std())
    elapsed_times = []
    degrees_list = []
    C=1
    for degree in range(1,4):
        start_time = time.time()
        model=svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C)
        model.fit(X_train,y_train)
        end_time = time.time()
        elapsed_times.append(end_time - start_time)
        degrees_list.append(degree)
    
    #plots the execution time wrt. degree of polynomial
    plt.plot(degrees_list,elapsed_times)
    plt.xticks([1,2,3])
    plt.xlabel('Time in seconds')
    plt.ylabel('Degree')
    plt.title('Execution time vs ploynomial degree of kernel poly')
    plt.show()
    
    return
###################################################

###################################################
## this method implements a custom kernel technique 
## which is K(X,Y)=k*XY'+theta where k and theta are
## constants. Since SVC supports custom kernel function
## with only 2 parameters we return the custom kernel 
## function name from another method which takes k and
## theta as input
def custom_kernel(k,theta):
    def my_kernel(X, Y):
	## write your own code here
        return ((k * np.dot(X,Y.T)) + theta)
    	## write your own code here
    return my_kernel
####################################################

####################################################
## this method uses the custom kernel and fit the 
## training data and reports accuracy on test data
def svm_custom_kernel(X_train, X_test, y_train, y_test, model):
    ## write your code here
    model.fit(X_train,y_train)
    print(F" Accuracy for SVM with Custom Kernel is { model.score(X_test,y_test) }")
    return
####################################################

####################################################
## main method:
def main():
	X,y=load_data()
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
	
	C=1
	models = (svm.SVC(kernel='linear', C=C),
          	svm.SVC(kernel='rbf', gamma='auto', C=C),
          	svm.SVC(kernel='poly', degree=2, gamma='auto', C=C))

	svm_models(X_train, X_test, y_train, y_test,models)
	
	ploy_kernel_var_deg(X_train, X_test, y_train, y_test)
	
	k=0.1
	theta=0.1

	model=svm.SVC(kernel=custom_kernel(k,theta))
	svm_custom_kernel(X_train, X_test, y_train, y_test, model)
#####################################################	


if __name__=='__main__':
	main()



	
