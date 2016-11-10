#
# THIS IS AN IMPLEMENTATION OF SUPPORT VECTOR MACHINES
#
# COPYRIGHT BELONGS TO THE AUTHOR OF THIS CODE
#
# AUTHOR : LAKSHMAN KUMAR
# AFFILIATION : UNIVERSITY OF MARYLAND, MARYLAND ROBOTICS CENTER
# EMAIL : LKUMAR93@UMD.EDU
# LINKEDIN : WWW.LINKEDIN.COM/IN/LAKSHMANKUMAR1993
#
# THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THE MIT LICENSE
# THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF
# THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
# 
# BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
# BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
# CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
# CONDITIONS.
#

###########################################
##
##	LIBRARIES
##
###########################################

from sklearn import svm,datasets
import numpy
import matplotlib.pyplot as plotter


###########################################
##
##	HELPER FUNCTIONS
##
###########################################

#Split the database to input and output vectors based on number of features
def CreateDataset( database, number_of_features ) :

	X = database.data[:,:number_of_features]
	Y = database.target
	return X,Y

#Plot the output of the SVM
def PlotClassifiedData(xx_set,X,Y,Z,title,label_x,label_y) :
	
	
	#Create a contour plot
	plotter.contourf(xx_set[0],xx_set[1],Z,cmap=plotter.cm.Paired,alpha=0.9)

	#Scatter input data
	plotter.scatter(X[:,0],X[:,1],c=Y,cmap = plotter.cm.Paired)

	#Label X and Y axis
	plotter.xlabel(label_x)
	plotter.ylabel(label_y)

	#Set limits for x axis and y axis
	plotter.xlim(xx_set[0].min(),xx_set[0].max())
	plotter.ylim(xx_set[1].min(),xx_set[1].max())

	#Remove scale markings on the axes
	plotter.xticks(())
	plotter.yticks(())

	#Create a title for the plot
	plotter.title(title)

	#Make the plot tighter
	plotter.tight_layout()
	
	#Display the plot
	plotter.show()




def Classify(database,label_x,label_y,classifier_type = 'linear', number_of_features = 2,  step_size = 0.02, regularization_param = 1.0, gamma = 0.7, degree = 3) :

	#Create dataset from the specified database based on the number of features
	X,Y = CreateDataset( database, number_of_features )

	#Declare a classifer
	Classifier = None

	#Initialize classifier based on the classifier type
	if classifier_type is 'rbf' :
		Classifier = svm.SVC(kernel=classifier_type, gamma = gamma , C = regularization_param).fit(X,Y)

	elif classifier_type is 'poly' :
		Classifier = svm.SVC(kernel=classifier_type, degree = degree , C = regularization_param).fit(X,Y)

	else :
		Classifier = svm.SVC(kernel=classifier_type, C = regularization_param).fit(X,Y)
	
	#Declare required lists
	x_min_set = []
	x_max_set = []
	x_set = []
	xx_set = [None,None]

	#Create a list of values from minimum value to the maximum value based on step size 
	for i in range(0,number_of_features) :
 
		x_min_set.append(X[:,i].min() - 1)
		x_max_set.append(X[:,i].max() + 1)
		x_set.append(numpy.arange(x_min_set[i],x_max_set[i],step_size))

	
	if number_of_features is not 2 :
		print "As of now this SVM implementation only supports 2D data"
		return
		
	#Create a meshgrid from the set
	xx_set[0],xx_set[1] = (numpy.meshgrid(x_set[0],x_set[1]))

	#Predict the class each datapoint belongs to
	Z = Classifier.predict(numpy.c_[xx_set[0].ravel(),xx_set[1].ravel()])

	error = 0;

	
	for i in range(0,150) :

		#Modify category types		
		if Z[i] == 2 :
			Z[i] = 0
		elif Z[i] == 0 :
			Z[i] = 2
		
		#Calculate errors
		if Z[i] != Y[i] :
			error = error + 1

	#Reshape Z according to the first element in xx_set	
	Z = Z.reshape(xx_set[0].shape)

	title = str(classifier_type)+" SVM , Error Percentage = " + str((error/150.0)*100)
	
	#Plot the data
	PlotClassifiedData(xx_set,X,Y,Z,title,label_x,label_y)

	
###########################################
##
##	MAIN FUNCTION
##
###########################################	

if __name__ == '__main__':
	
	#Classify the iris dataset using rbf kernel
	Classify(database=datasets.load_iris(),classifier_type ='rbf',label_x='Sepal Length',label_y='Sepal Width')

