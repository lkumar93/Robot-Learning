import matplotlib.pyplot as plotter
import simplegui
import math
import random

class DiscreteCMAC:
    def __init__(self,  generalization_factor, function, training_set_size = 50, input_space_size = 100, minimum_input_value = 0, maximum_input_value = 360, error_threshold = 0.02,learning_rate = 0.3):
 
        self.function = function
 	self.training_set_size = training_set_size
	self.testing_set_size = input_space_size - training_set_size
        self.generalization_factor = generalization_factor
	self.input_space_size = input_space_size
	self.minimum_input_value = minimum_input_value
	self.maximum_input_value = maximum_input_value
	self.step_size = (maximum_input_value - minimum_input_value)/input_space_size
	self.neighborhood_parsing_factor = int(math.floor(generalization_factor/2))
	self.error_threshold = error_threshold
	self.weights = [ 0 for i in range(0,self.input_space_size) ]
	self.input_space = [ math.radians(round(self.step_size*i)) for i in range(0,self.input_space_size) ]
	self.output_space = [ self.function(self.input_space[i])  for i in range(0,self.input_space_size) ]
	self.training_set_input = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_output = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_CMAC_output = [ 0 for i in range(0,self.training_set_size) ]
	self.training_set_global_indices = [ 0 for i in range(0,self.training_set_size) ]
	self.testing_set_input = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_true_output = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_CMAC_output = [ 0 for i in range(0,self.testing_set_size) ]
	self.testing_set_global_indices = [ 0 for i in range(0,self.testing_set_size) ]
	self.learning_rate = 0.2

	#for i in range(0,self.input_space_size) :
	#	self.weights[i] = 0
	#	self.input_space[i] = math.radians(round(self.step_size*i))
	#	self.output_space[i]= self.function(self.input_space[i])

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
		print "training sample number = " + str(i) 
		while local_convergence is False :
			local_CMAC_output = 0						
			for j in range(0,self.generalization_factor):
				global_neighborhood_index = global_index - j - self.neighborhood_parsing_factor
				if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
					self.weights[global_neighborhood_index] = self.weights[global_neighborhood_index] + (error/self.generalization_factor)*self.learning_rate
					local_CMAC_output = local_CMAC_output + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
			
			error = self.training_set_output[i] - local_CMAC_output
			print "error = " + str(error)
			#print "input = " + str(self.training_set_input[i])
			#print "output = " + str(self.training_set_output[i])
			#print "CMAC output = " + str(local_CMAC_output)
			if abs(error) <= self.error_threshold :
				local_convergence = True
				print "locally converged"
			if abs(error) > 1000 :
				return
					
			
    def test(self, dataset_input, dataset_true_output, dataset_global_indices):
	SuccessfulSamples = 0;
	local_CMAC_output = [ 0 for i in range(0,len(dataset_input)) ]
	for i in range(0,len(dataset_input)) :
		global_index = dataset_global_indices[i]
		for j in range(0,self.generalization_factor) :
			global_neighborhood_index = global_index - j - self.neighborhood_parsing_factor
			if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
				local_CMAC_output[i] = local_CMAC_output[i] + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
		error = dataset_true_output[i] - local_CMAC_output[i]
		if abs(error) <= self.error_threshold+0.05 :
			SuccessfulSamples = SuccessfulSamples + 1

	SuccessPercentage = (SuccessfulSamples/len(dataset_input))*100
	return local_CMAC_output,SuccessPercentage	
		

    def execute(self):
	self.create_datasets()
	self.train()
	self.training_set_CMAC_output,TrainingSuccessPercentage = self.test(self.training_set_input,self.training_set_output,self.training_set_global_indices)
	print "Training Success Percentage " + str(TrainingSuccessPercentage)
	print "Training set input"
	print self.training_set_input
	print "CMAC output"
	print self.training_set_CMAC_output
	print "Training set output"
	print self.training_set_output
	self.testing_set_CMAC_output,TestingSuccessPercentage = self.test(self.testing_set_input,self.testing_set_true_output,self.testing_set_global_indices)
	print "Testing Success Percentage " + str(TestingSuccessPercentage)


SineWaveCMAC= DiscreteCMAC(3,math.sin)
SineWaveCMAC.execute()
