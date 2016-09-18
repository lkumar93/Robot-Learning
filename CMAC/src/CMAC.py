#
# THIS IS AN IMPLEMENTATION OF CEREBELLAR MODEL ARTICULATION CONTROLLER
# PROPOSED BY JAMES ALBUS IN 1975
#
# COPYRIGHT BELONGS TO THE AUTHOR OF THIS CODE
#
# AUTHOR : LAKSHMAN KUMAR
# AFFILIATION : UNIVERSITY OF MARYLAND, MARYLAND ROBOTICS CENTER
# EMAIL : LKUMAR93@UMD.EDU
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

import matplotlib.pyplot as plotter
import simplegui
import math
import random
import numpy
import time

max_generalization_factor = 50
min_generalization_factor = 1
max_input_space_size = 2000
min_input_space_size = 100
input_space_step_size = 50
plot_generalization_factor = 41
plot_input_space_size = 1000
input_space_split_data_size = int((max_input_space_size - min_input_space_size ) / float(input_space_step_size))


class CMAC:
    def __init__(self,  generalization_factor, function, input_space_size = 100, CMACType = 'CONTINUOUS', dataset_split_factor = 1.5,  minimum_input_value = 0, maximum_input_value = 360, error_threshold = 0.02, learning_rate = 0.1, convergence_threshold = 0.03, max_convergence_iterations = 20):
 
        self.function = function
 	self.training_set_size = int(input_space_size/dataset_split_factor)
	self.testing_set_size = input_space_size - self.training_set_size
        self.generalization_factor = generalization_factor
	self.input_space_size = input_space_size
	self.CMACType = CMACType
	self.minimum_input_value = minimum_input_value
	self.maximum_input_value = maximum_input_value
	self.step_size = (maximum_input_value - minimum_input_value)/float(input_space_size)
	self.neighborhood_parsing_factor = int(math.floor(generalization_factor/2))
	self.error_threshold = error_threshold
	self.weights = [ 0 for i in range(0,self.input_space_size) ]
	self.input_space = [ math.radians(self.step_size*(i+1)) for i in range(0,self.input_space_size) ]
	self.output_space = [ self.function(self.input_space[i])  for i in range(0,self.input_space_size) ]
	self.training_set_input = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_output = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_CMAC_output = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_global_indices = [ 0 for i in range(0,self.training_set_size) ]
	self.testing_set_input = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_true_output = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_CMAC_output = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_global_indices = [ 0 for i in range(0,self.testing_set_size) ]
	self.learning_rate = learning_rate
	self.TrainingError = 1.0
	self.TestingError = 1.0
	self.convergence_threshold = convergence_threshold
	self.max_convergence_iterations = max_convergence_iterations
	self.convergence = False
	self.convergence_time = 1000


    def create_datasets(self):
	count = 0;
	randomized_range_values = [ x for x in range(0,self.input_space_size) ]
	random.shuffle(randomized_range_values)

	for i in randomized_range_values :

		if count < self.training_set_size :
			self.training_set_input[count] = self.input_space[i]
			self.training_set_output[count] = self.output_space[i]
			self.training_set_CMAC_output[count] = 0
			self.training_set_global_indices[count] = i

		else :
			
			self.testing_set_input[count - self.training_set_size] = self.input_space[i]
			self.testing_set_true_output[count - self.training_set_size] = self.output_space[i]
			self.testing_set_CMAC_output[count - self.training_set_size] = 0
			self.testing_set_global_indices[count - self.training_set_size] = i
		
		count = count + 1
	
    def train(self):	
	error = 1000
	for i in range(0,self.training_set_size) :
		local_convergence = False		
		global_index = self.training_set_global_indices[i]
		error = 0
		iteration = 0
		CMACTypeWeight = 0
		while local_convergence is False :
			local_CMAC_output = 0						
			for j in range(0,self.generalization_factor):
				global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
				if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
					self.weights[global_neighborhood_index] = self.weights[global_neighborhood_index] + (error/self.generalization_factor)*self.learning_rate

					local_CMAC_output = local_CMAC_output + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])

			
			if self.CMACType is 'DISCRETE' :
				error = round(self.training_set_output[i],1) - local_CMAC_output
				CMACTypeWeight = 0.05
				#print str(round(self.training_set_output[i],1))
			else :
				error = self.training_set_output[i] - local_CMAC_output
				

			iteration = iteration + 1
			
			if abs(error) <= (self.error_threshold + CMACTypeWeight ) or (global_neighborhood_index < 0 and iteration > 10):
				local_convergence = True
		
			
    def test(self, dataset_input, dataset_true_output, dataset_global_indices):
	CumulativeError = 0;
	local_CMAC_output = [ 0 for i in range(0,len(dataset_input)) ]
	for i in range(0,len(dataset_input)) :
		global_index = dataset_global_indices[i]
		for j in range(0,self.generalization_factor) :
			global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
			if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
				local_CMAC_output[i] = local_CMAC_output[i] + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
		error = dataset_true_output[i] - local_CMAC_output[i]
		
		CumulativeError = CumulativeError + abs(error)

	if self.CMACType is 'DISCRETE':
		return numpy.around(local_CMAC_output,decimals = 1), CumulativeError
	else:
		return local_CMAC_output, CumulativeError	
		

    def execute(self):
	self.create_datasets()
	iterations = 0
	self.convergence_time = time.time()
	
	while self.TestingError > self.convergence_threshold and iterations < self.max_convergence_iterations :
		self.train()
		self.training_set_CMAC_output,TrainingCumulativeError = self.test(self.training_set_input,self.training_set_output,self.training_set_global_indices)
		self.TrainingError = TrainingCumulativeError/self.training_set_size
		#print "Training Error = " + str(self.TrainingError)
	
		self.testing_set_CMAC_output,TestingCumulativeError = self.test(self.testing_set_input,self.testing_set_true_output,self.testing_set_global_indices)
		self.TestingError = TestingCumulativeError/self.testing_set_size
		#print "Testing Error = " + str(self.TestingError)
		iterations = iterations + 1

	if self.TestingError <= self.convergence_threshold :
		self.convergence = True
		self.convergence_time = time.time() - self.convergence_time
		print ' GeneralizationFactor= '+str(self.generalization_factor) +' InputSize= '+ str(self.input_space_size)+'Time = '+ str(self.convergence_time)
		
	return self.TrainingError, self.TestingError

	

    def plot_graphs(self):
		
	sorted_testing_set_input = [x for (y,x) in sorted(zip(self.testing_set_global_indices,self.testing_set_input))]
	sorted_testing_set_CMAC_output = [x for (y,x) in sorted(zip(self.testing_set_global_indices,self.testing_set_CMAC_output))]
	sorted_training_set_input = [x for (y,x) in sorted(zip(self.training_set_global_indices,self.training_set_input))]
	sorted_training_set_CMAC_output = [x for (y,x) in sorted(zip(self.training_set_global_indices,self.training_set_CMAC_output))]
	plotter.figure(figsize=(20,10))
	plotter.suptitle(self.CMACType + ' CMAC ')
	plotter.subplot(221)
	plotter.plot(self.training_set_input,self.training_set_output,'ro')
	plotter.title(' Input Space Size = ' + str(self.input_space_size) + '\n Training data' )
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.subplot(223)
	if self.CMACType is 'DISCRETE' :
		plotter.step(sorted_training_set_input,sorted_training_set_CMAC_output)#,'r--')
	else :
		plotter.plot(sorted_training_set_input,sorted_training_set_CMAC_output)
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.title('\n Training Error = ' + str(self.TrainingError))
	plotter.subplot(222)
	plotter.plot(self.testing_set_input,self.testing_set_true_output,'ro')
	plotter.title(' Generalization_Factor = ' + str(self.generalization_factor) + ' \n Test Data')
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.subplot(224)
	if self.CMACType is 'DISCRETE' :
		plotter.step(sorted_testing_set_input,sorted_testing_set_CMAC_output)#,'r--')
		print 'DISCRETE plot'
	else :
		plotter.plot(sorted_testing_set_input,sorted_testing_set_CMAC_output)
		print 'CONTINUOUS plot'
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.title('\n Testing Error = ' + str(self.TestingError))
	plotter.subplots_adjust(0.1, 0.08, 0.89, 0.89,0.2, 0.35)
	plotter.show()



def PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, xlabel_value) :
	
	print xlabel_value
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
	


def RunCMAC(function) :
	DiscreteCMAC= [ CMAC(i,function,plot_input_space_size,'DISCRETE') for i in range(min_generalization_factor,max_generalization_factor + 1) ]
	ContinuousCMAC= [ CMAC(i, function,plot_input_space_size,'CONTINUOUS') for i in range(min_generalization_factor,max_generalization_factor + 1) ]
	TrainingErrorDiscreteRange = [ 0 for i in range(0,max_generalization_factor) ]
	TestingErrorDiscreteRange = [ 0 for i in range(0,max_generalization_factor) ]
	TrainingErrorContinuousRange = [ 0 for i in range(0,max_generalization_factor) ]
	TestingErrorContinuousRange = [ 0 for i in range(0,max_generalization_factor) ]
	DiscreteConvergenceTimes = [ 1 for i in range(0,max_generalization_factor) ]
	ContinuousConvergenceTimes = [ 1 for i in range(0,max_generalization_factor) ]

	for i in range( 0, max_generalization_factor ) :
		TrainingErrorDiscreteRange[i],TestingErrorDiscreteRange[i] = DiscreteCMAC[i].execute()
		TrainingErrorContinuousRange[i],TestingErrorContinuousRange[i] = ContinuousCMAC[i].execute()
		if DiscreteCMAC[i].convergence is True :
			DiscreteConvergenceTimes[i]  = DiscreteCMAC[i].convergence_time
		if ContinuousCMAC[i].convergence is True :
			ContinuousConvergenceTimes[i] = ContinuousCMAC[i].convergence_time
		if i is plot_generalization_factor -1 :
			DiscreteCMAC[i].plot_graphs()
			ContinuousCMAC[i].plot_graphs()

	PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, 'GeneralizationFactor' )
	
	DiscreteCMAC = [ CMAC(plot_generalization_factor, function,i,'DISCRETE') for i in range(min_input_space_size, max_input_space_size, input_space_step_size) ]
	ContinuousCMAC= [ CMAC(plot_generalization_factor, function,i,'CONTINUOUS') for i in range(min_input_space_size, max_input_space_size, input_space_step_size) ]

	TrainingErrorDiscreteRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TrainingErrorContinuousRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TestingErrorDiscreteRange = [ 0 for i in range(0,input_space_split_data_size) ]
	TestingErrorContinuousRange = [ 0 for i in range(0,input_space_split_data_size) ]
	DiscreteConvergenceTimes = [ 1 for i in range(0,input_space_split_data_size) ]
	ContinuousConvergenceTimes = [ 1 for i in range(0,input_space_split_data_size) ]

	for i in range( 0, input_space_split_data_size ) :
		TrainingErrorDiscreteRange[i],TestingErrorDiscreteRange[i] = DiscreteCMAC[i].execute()
		TrainingErrorContinuousRange[i],TestingErrorContinuousRange[i] = ContinuousCMAC[i].execute()
		if DiscreteCMAC[i].convergence is True :
			DiscreteConvergenceTimes[i]  = DiscreteCMAC[i].convergence_time
		if ContinuousCMAC[i].convergence is True :
			ContinuousConvergenceTimes[i] = ContinuousCMAC[i].convergence_time	

	PlotCMACPerformance(TrainingErrorDiscreteRange, TestingErrorDiscreteRange, TrainingErrorContinuousRange, TestingErrorContinuousRange, DiscreteConvergenceTimes, ContinuousConvergenceTimes, 'InputSpaceSize' )		



if __name__ == '__main__':
	RunCMAC(math.sin)