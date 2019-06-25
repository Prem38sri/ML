#This code generates random data from Uniform Distribution and assigns labels.
#The data is a non-linear one with points inside a circle of fixed radius marked as -1 and outside as +1.
#We flip the labels of some data (here with 5% probability) to introduce some noise.
#You will be using Decision Tree and Naive Bayes Classifiers to classify the above generated data.


import numpy as np
import matplotlib.pyplot as plt
#Do all the necessary imports here

def generate_data():

	np.random.seed(123) #Set seed for reproducibility. Please do not change/remove this line.
	x = np.random.uniform(-1,1,(3000,2)) #You may change the number of samples you wish to generate
	y=[]
	for i in range(x.shape[0]):
		y.append(np.sign(x[i][0]**2 + x[i][1]**2 - 0.5)) #Forming labels
	return x,y

def flip_labels(y):

	num = int(0.05 * len(y)) #5% of data to be flipped
	np.random.seed(123)
	changeind = np.random.choice(len(y),num,replace=False) #Sampling without replacement
	#For example, np.random.choice(5,3) = ([0,2,3]); first argument is the limit till which we intend to pick up elements, second is the number of elements to be sampled

	#Creating a copy of the array to modify
	yc=np.copy(y) # yc=y is a bad practice since it points to the same location and changing y or yc would change the other which won't be desired always
	#Flip labels -1 --> 1 and 1 --> -1
	for i in changeind:
		if yc[i]==-1.0:
			yc[i]=1.0
		else:
			yc[i]=-1.0

	return yc

#Fill up the below function
def train_test_dt(x,y):
    
    # Perform a k-fold cross validation using Decision Tree
    
    acc_train=[]
    acc_test=[]
    samples=[]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) #Train set = 70%, Test set = 30%
    
    dt_classifier = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=3)
    dt_classifier.fit(X_train, y_train)
    #y_pred=dt_classifier.predict(X_test)
    
    #dt_classifier.fit(X_train, y_train)
    #predicted_y = dt_classifier.predict(X_test)
    #predicted_y=np.asarray(predicted_y)
    #print(predicted_classes)
    
    
    for k in range(2,11):
        acc_train.append(cross_val_score(dt_classifier, X_train, y_train, cv=k).sum() / k)
        acc_test.append(cross_val_score(dt_classifier, X_test, y_test, cv=k).sum() / k)
        samples.append(k)
        #print(F"for { k } -")
        #print(cross_val_score(dt_classifier, X_train, y_train, cv=k))
    
	#Plot train and test accuracy with varying k (1<=k<=10)
    
    plt.plot(samples,acc_train,'bo-',label='Train Accuracy')
    plt.plot(samples,acc_test,'ro-',label='Test Accuracy')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('K-Fold for 500 samples with 2  features',fontsize=15)
    plt.ylabel('Mean Accuracy',fontsize=15)
    plt.legend(fontsize=15)
    plt.title('DT Classifier With noise',fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    plt.show()
    return

#Fill up the velow function
def train_test_nb(x,y):
    # Perform a k-fold cross validation using Decision Tree
    
    acc_train=[]
    acc_test=[]
    samples=[]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #Train set = 70%, Test set = 30%
    
    gnb_classifier = GaussianNB()
    gnb_classifier.fit(X_train, y_train)
    
    
    for k in range(2,11):
        acc_train.append(cross_val_score(gnb_classifier, X_train, y_train, cv=k).sum() / k)
        acc_test.append(cross_val_score(gnb_classifier, X_test, y_test, cv=k).sum() / k)
        samples.append(k)
    
    
	#Plot train and test accuracy with varying k (1<=k<=10)
    
    plt.plot(samples,acc_train,'bo-',label='Train Accuracy')
    plt.plot(samples,acc_test,'ro-',label='Test Accuracy')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('K-Fold for 500 samples with 2  features',fontsize=15)
    plt.ylabel('Mean Accuracy',fontsize=15)
    plt.legend(fontsize=15)
    plt.title('NBB Classifier With noise',fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    plt.show()
    return

def main():

	x,y = generate_data() #Generate data
	y = flip_labels(y) #Flip labels
	y=np.asarray(y) #Change list to array
	train_test_dt(x,y)
	train_test_nb(x,y)


if __name__=='__main__':
	main()
