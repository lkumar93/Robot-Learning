#
# THIS IS AN IMPLEMENTATION OF CEREBELLAR MODEL ARTICULATION CONTROLLER
# PROPOSED BY JAMES ALBUS IN 1975
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

#Import Libraries Required For CMAC
import matplotlib.pyplot as plotter
import math
import random
import numpy
import time

#Initialize Variables Required To Test CMAC
max_generalization_factor = 50
min_generalization_factor = 1
max_input_space_size = 2100
min_input_space_size = 100
input_space_step_size = 200
plot_generalization_factor = 31
plot_input_space_size = 600
minimum_output_value = -1.0
maximum_output_value = 1.0
input_space_split_data_size = int((max_input_space_size - min_input_space_size ) / float(input_space_step_size))

#Return Mean Squared Error Between The Desired Output And The Actual Output
def MeanSquaredError( DesiredOutput, ActualOutput) :
	mean_squared_error = ( 0.5 * ( math.pow( DesiredOutput , 2 ) - math.pow( ActualOutput , 2 ) ) )
	return mean_squared_error

#Return Index Of The Nearest Element To The Value In The Array
def find_nearest(array,value):
	values = [ value  for i in range(0,len(array)) ]
	idx = (numpy.abs(numpy.array(array)-numpy.array(values))).argmin()
	return idx

#Create A Common Framework For Creating CMAC Objects
class CMAC:

    #Initialize The Object With User Specified Values And Function
    def __init__(self,  generalization_factor, dataset, CMAC_type = 'CONTINUOUS', minimum_output_value = minimum_output_value, maximum_output_value = maximum_output_value, local_convergence_threshold = 0.001, learning_rate = 0.15, global_convergence_threshold = 0.001, max_global_convergence_iterations = 20):
 
        self.generalization_factor = generalization_factor

	self.neighborhood_parsing_factor = int(math.floor(generalization_factor/2))	

	self.input_space = dataset[0]
	self.output_space = dataset[1]
	self.input_space_size = len(self.input_space)
	self.training_set_input = dataset[2]
	self.training_set_output = dataset[3]
	self.training_set_size = len(dataset[2])
	self.training_set_global_indices = dataset[4]
	self.training_set_CMAC_output = [ 0 for i in range(0,self.training_set_size) ]

	self.weights = [ 0 for i in range(0,self.input_space_size) ]

	self.testing_set_input = dataset[5]
	self.testing_set_true_output = dataset[6]
	self.testing_set_global_indices = dataset[7]
	self.testing_set_size = len(dataset[5])
	self.testing_set_CMAC_output = [ 0 for i in range(0,self.testing_set_size) ]

	self.minimum_output_value = minimum_output_value 
	self.maximum_output_value = maximum_output_value

	self.TrainingError = 1.0
	self.TestingError = 1.0

	self.CMAC_type = CMAC_type

	self.local_convergence_threshold = local_convergence_threshold
	self.learning_rate = learning_rate

	self.global_convergence_threshold = global_convergence_threshold
	self.max_global_convergence_iterations = max_global_convergence_iterations
	self.convergence = False
	self.convergence_time = 1000

    #Train The CMAC With Training Data Until Each Of The DataPoints Achieve Local Convergence	
    def train(self):	
	error = 1000
	#print range(0,self.training_set_size)
	for i in range(0,self.training_set_size) :
		local_convergence = False		
		#print len(self.training_set_global_indices)
		global_index = self.training_set_global_indices[i]
		error = 0
		iteration = 0
		generalization_factor_offset = 0
		
		#Compute An Offset To Account For When Computing Weights For Edge Cases, Where You May Not Get The Specified Neighborhood Size
		if i - self.neighborhood_parsing_factor < 0 :
			generalization_factor_offset = i - self.neighborhood_parsing_factor

		if i + self.neighborhood_parsing_factor >= self.training_set_size :
			generalization_factor_offset = self.training_set_size - (i + self.neighborhood_parsing_factor) 
		
		#Repeat Till You Achieve Local Convergence
		while local_convergence is False :
			local_CMAC_output = 0		
			#Compute Weights Based On The Neighborhood Of The Data Point, That Is Specified By 'generalization_factor'	
			for j in range(0,self.generalization_factor):
				global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
							
				#Make Sure The Index Is Not Beyond Limit When Taking Into Account The Generalization Factor
				if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
					 
					#Update Weights According To The Computed Error
					self.weights[global_neighborhood_index] = self.weights[global_neighborhood_index] + (error/(self.generalization_factor+generalization_factor_offset))*self.learning_rate
					#Compute The Output
					local_CMAC_output = local_CMAC_output + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
					#print self.weights[global_neighborhood_index]

			error = self.training_set_output[i] - local_CMAC_output
			#print 	self.training_set_output[i]	
			#print local_CMAC_output

			iteration = iteration + 1
		
			if iteration > 25 :
				break
			
			#Local Convergence Is Achieved If Absolute Value Of Error Is Within The Threshold
			if abs(MeanSquaredError(self.training_set_output[i],local_CMAC_output)) <= (self.local_convergence_threshold):
				local_convergence = True
		

    #Test CMAC With Argument Datasets Which May Be The Testing Data Or Training Data		
    def test(self, DatasetType):
	CumulativeError = 0;
	

	if DatasetType is 'TestingData' :
		dataset_input = self.testing_set_input
		dataset_true_output = self.testing_set_true_output
		dataset_global_indices = self.testing_set_global_indices

	else :
		dataset_input = self.training_set_input
		dataset_true_output = self.training_set_output
		dataset_global_indices = self.training_set_global_indices

	local_CMAC_output = [ 0 for i in range(0,len(dataset_input)) ]
	
	for i in range(0,len(dataset_input)) :
		global_index = dataset_global_indices[i]
		for j in range(0,self.generalization_factor) :
			global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
			#print global_neighborhood_index
			if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size:
				local_CMAC_output[i] = local_CMAC_output[i] + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
			

		error = dataset_true_output[i] - local_CMAC_output[i]
		
		#Add Up All The Accumulated Errors
		CumulativeError = CumulativeError + abs(MeanSquaredError( dataset_true_output[i] , local_CMAC_output[i] ) )

	if self.CMAC_type is 'DISCRETE':
		#If Discrete Round Off All The Elements In The Output
		return numpy.around(local_CMAC_output,decimals = 1), CumulativeError
	else:
		return local_CMAC_output, CumulativeError		
		
    #Train And Test Data Until You Achieve Global Convergence
    def execute(self):
	iterations = 0
	self.convergence_time = time.time()
	
	#Repeat Until Number Of Iterations Exceed Max Iterations Required For Convergence And Break If Global Convergence Occurs
	while iterations < self.max_global_convergence_iterations :
		self.train()

		self.training_set_CMAC_output,TrainingCumulativeError = self.test('TrainingData')
		self.TrainingError = TrainingCumulativeError/self.training_set_size
	
		self.testing_set_CMAC_output,TestingCumulativeError = self.test('TestingData')
		self.TestingError = TestingCumulativeError/self.testing_set_size

		iterations = iterations + 1

		#If Testing Error Is Below Convergence Threshold Then Global Convergence Is Achieved Within The Maximum Number Of Iterations
		if self.TestingError <= self.global_convergence_threshold :
			self.convergence = True
			break 

	
	
	self.convergence_time = time.time() - self.convergence_time
		
	return self.TrainingError, self.TestingError

	
    #Plot CMAC's Input Space Size Vs Data & Generalization Factor Vs Data Graphs
    def plot_graphs(self,Header):
		
	plotter.figure(figsize=(20,10))
	plotter.suptitle(' Best '+ str(self.CMAC_type) + ' CMAC With Fixed ' + Header)
	plotter.subplot(221)
	plotter.plot(self.training_set_input,self.training_set_output,'ro')
	plotter.title(' Input Space Size = ' + str(self.input_space_size) + '\n Training data' )
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.subplot(223)
	plotter.ylim((self.minimum_output_value,self.maximum_output_value))
	#if self.CMAC_type is 'DISCRETE' :
	#	plotter.step(self.training_set_input,self.training_set_CMAC_output)
	#else :
	plotter.plot(self.training_set_input,self.training_set_CMAC_output,'ro')
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.title('\n Training Error = ' + str(self.TrainingError))
	plotter.subplot(222)
	plotter.plot(self.testing_set_input,self.testing_set_true_output,'ro')
	plotter.title(' Generalization_Factor = ' + str(self.generalization_factor) + ' \n Test Data')
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.subplot(224)
	plotter.ylim((self.minimum_output_value,self.maximum_output_value))
	#if self.CMAC_type is 'DISCRETE' :
	#	plotter.step(self.testing_set_input,self.testing_set_CMAC_output)
	#else :
	plotter.plot(self.testing_set_input,self.testing_set_CMAC_output,'ro')
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.title('\n Testing Error = ' + str(self.TestingError))
	plotter.subplots_adjust(0.1, 0.08, 0.89, 0.89,0.2, 0.35)

	plotter.show()


#Plot CMAC's Performance Graphs(Generalization Factor Vs Error, Input Space Size Vs Error & Convergence Time Vs Error)
def PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, xlabel_value) :
	
	if xlabel_value is str('GeneralizationFactor') :		
		RangeValues = range(1,max_generalization_factor + 1)
		Value = ' Input Space Size = ' + str(plot_input_space_size)
	else :
		RangeValues = range(min_input_space_size, max_input_space_size, input_space_step_size)
		Value = ' Generalization Factor = ' + str(plot_generalization_factor)
	plotter.figure(figsize=(20,11))
	plotter.suptitle( xlabel_value+' Vs Error' )
	plotter.subplot(221)
	plotter.plot(RangeValues,TrainingErrorDiscreteRange )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Training Error')
	plotter.title('DISCRETE CMAC ' +' \n '+ Value +' \n ' + xlabel_value+' Vs Training Error')
	plotter.subplot(223)
	plotter.plot(RangeValues,TestingErrorDiscreteRange )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Testing Error')
	plotter.title(' \n ' + xlabel_value+' Vs Testing Error')
	plotter.subplot(222)
	plotter.plot(RangeValues,TrainingErrorContinuousRange )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Training Error')
	plotter.title('CONTINUOUS CMAC'+' \n '+ Value +' \n ' + xlabel_value+' Vs Training Error')
	plotter.subplot(224)
	plotter.plot(RangeValues,TestingErrorContinuousRange )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Testing Error')
	plotter.title(' \n ' + xlabel_value+' Vs Testing Error')
	plotter.subplots_adjust(0.1, 0.08, 0.89, 0.89,0.2, 0.35)
	plotter.show()

	plotter.figure(figsize=(20,11))
	plotter.subplot(211)
	plotter.suptitle( xlabel_value+' Vs Convergence Times' )
	plotter.plot(RangeValues,DiscreteConvergenceTimes )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Convergence Times')
	plotter.title('DISCRETE CMAC ')
	plotter.subplot(212)
	plotter.plot(RangeValues,ContinuousConvergenceTimes )
	plotter.xlabel(xlabel_value)
	plotter.ylabel('Convergence Times')
	plotter.title('Continuous CMAC ')
	plotter.subplots_adjust(0.29, 0.10, 0.71, 0.88,0.18, 0.59)
	plotter.show()


def GenerateDataset(function, input_space_size = plot_input_space_size, minimum_input_value = 0, maximum_input_value = 360, dataset_split_factor = 1.5) :

	step_size = (maximum_input_value - minimum_input_value)/float(input_space_size)

	input_space = [ math.radians(step_size*(i+1)) for i in range(0,input_space_size) ]
	output_space = [ function(input_space[i])  for i in range(0,input_space_size) ]
	
	training_set_size = int(input_space_size/dataset_split_factor)
	testing_set_size = input_space_size - training_set_size 

	unsorted_training_set_input = [ 0  for i in range(0,training_set_size) ]
	unsorted_training_set_output  = [ 0  for i in range(0,training_set_size)]
	training_set_global_indices = [ 0  for i in range(0,training_set_size)]

	unsorted_testing_set_input = [ 0  for i in range(0,testing_set_size) ]
	unsorted_testing_set_true_output  = [ 0  for i in range(0,testing_set_size)]
	testing_set_global_indices  = [ 0  for i in range(0,testing_set_size)]
	
	count = 0;
	randomized_range_values = [ x for x in range(0,input_space_size) ]
	random.shuffle(randomized_range_values)

	for i in randomized_range_values :

		if count < training_set_size :
			unsorted_training_set_input[count] = input_space[i]
			unsorted_training_set_output[count] = output_space[i]
			training_set_global_indices[count] = i

		else :
			
			unsorted_testing_set_input[count - training_set_size] = input_space[i]
			unsorted_testing_set_true_output[count - training_set_size] = output_space[i]
			testing_set_global_indices[count - training_set_size] = i
		
		count = count + 1

	#sorted_training_set_input = [x for (y,x) in sorted(zip(training_set_global_indices,unsorted_training_set_input))] 
	#sorted_training_set_output = [x for (y,x) in sorted(zip(training_set_global_indices,unsorted_training_set_output))]
	#sorted_testing_set_input =  [x for (y,x) in sorted(zip(testing_set_global_indices,unsorted_testing_set_input))] 
	#sorted_testing_set_output =  [x for (y,x) in sorted(zip(testing_set_global_indices,unsorted_testing_set_true_output))] 

	
	return [input_space,output_space,unsorted_training_set_input, unsorted_training_set_output,training_set_global_indices, unsorted_testing_set_input, unsorted_testing_set_true_output,testing_set_global_indices]

# Run CMAC Function And Generate Graphs 
def RunCMAC(function) :

	Dataset = GenerateDataset(function)

	#Find How Testing Error Varies With Increase In Generalization Factor Also Find The Best CMAC For The Given Input Size
	DiscreteCMAC= [ CMAC(i,Dataset, 'DISCRETE') for i in range(min_generalization_factor,max_generalization_factor + 1) ]
	ContinuousCMAC= [ CMAC(i, Dataset, 'CONTINUOUS') for i in range(min_generalization_factor,max_generalization_factor + 1) ]
	TrainingErrorDiscreteRange = [ 0 for i in range(0,max_generalization_factor) ]
	TestingErrorDiscreteRange = [ 0 for i in range(0,max_generalization_factor) ]
	TrainingErrorContinuousRange = [ 0 for i in range(0,max_generalization_factor) ]
	TestingErrorContinuousRange = [ 0 for i in range(0,max_generalization_factor) ]
	DiscreteConvergenceTimes = [ 100 for i in range(0,max_generalization_factor) ]
	ContinuousConvergenceTimes = [ 100 for i in range(0,max_generalization_factor) ]
	BestDiscreteCMAC = -1
	BestContinuousCMAC = -1
	LowestDiscreteTestingError = 1000
	LowestContinuousTestingError = 1000

	print ' \n  Plot Generalization Factor = ' + str(plot_generalization_factor) + ' with Errors \n'
	
	ContinuousCMAC[plot_generalization_factor].execute()
	ContinuousCMAC[plot_generalization_factor].plot_graphs('InputSpaceSize')
	
	print ' \n Generalization Factor Variance - CMAC Performance \n '
	for i in range( 0, max_generalization_factor ) :
		TrainingErrorDiscreteRange[i],TestingErrorDiscreteRange[i] = DiscreteCMAC[i].execute()
		TrainingErrorContinuousRange[i],TestingErrorContinuousRange[i] = ContinuousCMAC[i].execute()
		print 'Generalization Factor - ' + str(i+1) +' Continuous Testing Error - ' + str(round(TestingErrorContinuousRange[i],3))+ ' Continuous Convergence Time - ' + str(round(ContinuousCMAC[i].convergence_time,2)) + ' Discrete Testing Error - ' + str(round(TestingErrorDiscreteRange[i],3)) 
		#if DiscreteCMAC[i].convergence is True :
		DiscreteConvergenceTimes[i]  = DiscreteCMAC[i].convergence_time
		#if ContinuousCMAC[i].convergence is True :
		ContinuousConvergenceTimes[i] = ContinuousCMAC[i].convergence_time
		if TestingErrorDiscreteRange[i] < LowestDiscreteTestingError  :
			LowestDiscreteTestingError = TestingErrorDiscreteRange[i]
			BestDiscreteCMAC = i
		if TestingErrorContinuousRange[i] < LowestContinuousTestingError :
			LowestContinuousTestingError = TestingErrorContinuousRange[i]
			BestContinuousCMAC = i

 
	if BestDiscreteCMAC  is not  -1 :
		DiscreteCMAC[BestDiscreteCMAC].plot_graphs('InputSpaceSize')
	else :
		print "Error - Discrete CMAC"

	if BestContinuousCMAC  is not  -1 :
		ContinuousCMAC[BestContinuousCMAC].plot_graphs('InputSpaceSize')
	else :
		print "Error - Continuous CMAC"

	#Plot Performance Graphs With Increasing Generalization Factor
	PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, 'GeneralizationFactor' )
	
	print ' \n Input Space Size Variance - CMAC Performance \n'

	#Find How Testing Error Varies With Increase In Input Space Size Also Find The Best CMAC For The Given Generalization Factor
	DiscreteCMAC = [ CMAC(plot_generalization_factor, GenerateDataset(function,i),'DISCRETE') for i in range(min_input_space_size, max_input_space_size, input_space_step_size) ]
	ContinuousCMAC= [ CMAC(plot_generalization_factor, GenerateDataset(function,i),'CONTINUOUS') for i in range(min_input_space_size, max_input_space_size, input_space_step_size) ]

	TrainingErrorDiscreteRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TrainingErrorContinuousRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TestingErrorDiscreteRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TestingErrorContinuousRange = [ 0 for i in range(0,input_space_split_data_size) ]
	DiscreteConvergenceTimes = [ 1 for i in range(0,input_space_split_data_size) ]
	ContinuousConvergenceTimes = [ 1 for i in range(0,input_space_split_data_size) ]
	BestDiscreteCMAC = -1
	BestContinuousCMAC = -1
	LowestDiscreteTestingError = 1000
	LowestContinuousTestingError = 1000
	InputSize = min_input_space_size

	for i in range( 0, input_space_split_data_size ) :
		TrainingErrorDiscreteRange[i],TestingErrorDiscreteRange[i] = DiscreteCMAC[i].execute()
		TrainingErrorContinuousRange[i],TestingErrorContinuousRange[i] = ContinuousCMAC[i].execute()
		print 'Input Space Size - ' + str(InputSize) +' Continuous Testing Error - ' + str(round(TestingErrorContinuousRange[i],3))+ ' Continuous Convergence Time - ' + str(round(ContinuousCMAC[i].convergence_time,2)) + ' Discrete Testing Error - ' + str(round(TestingErrorDiscreteRange[i],3)) 
		#if DiscreteCMAC[i].convergence is True :
		DiscreteConvergenceTimes[i]  = DiscreteCMAC[i].convergence_time
		#if ContinuousCMAC[i].convergence is True :
		ContinuousConvergenceTimes[i] = ContinuousCMAC[i].convergence_time
		if TestingErrorDiscreteRange[i] < LowestDiscreteTestingError  :
			LowestDiscreteTestingError = TestingErrorDiscreteRange[i]
			BestDiscreteCMAC = i
		if TestingErrorContinuousRange[i] < LowestContinuousTestingError :
			LowestContinuousTestingError = TestingErrorContinuousRange[i]
			BestContinuousCMAC = i
		InputSize = InputSize + (max_input_space_size - min_input_space_size)/input_space_split_data_size

 
	if BestDiscreteCMAC  is not  -1 :
		DiscreteCMAC[BestDiscreteCMAC].plot_graphs('GeneralizationFactor')
	else :
		print "Error - Discrete CMAC"

	if BestContinuousCMAC  is not  -1 :
		ContinuousCMAC[BestContinuousCMAC].plot_graphs('GeneralizationFactor')
	else :
		print "Error - Continuous CMAC"	

	#Plot Performance Graphs With Increasing Input Space Size
	PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, 'InputSpaceSize' )		

	

if __name__ == '__main__':
	#Train CMAC To Learn The Sinusoidal Function And Evaluate Its Performance On Test Dataset
	RunCMAC(math.sin)
