import matplotlib.pyplot as plotter
import simplegui
import math
import random

class DiscreteCMAC:
    def __init__(self,  generalization_factor, function, training_set_size = 750, input_space_size = 1000, minimum_input_value = 0, maximum_input_value = 360, error_threshold = 0.05,learning_rate = 0.1):
 
        self.function = function
 	self.training_set_size = training_set_size
	self.testing_set_size = input_space_size - training_set_size
        self.generalization_factor = generalization_factor
	self.input_space_size = input_space_size
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
				global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
				if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
					self.weights[global_neighborhood_index] = self.weights[global_neighborhood_index] + (error/self.generalization_factor)*self.learning_rate
					local_CMAC_output = local_CMAC_output + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
				
			
			error = self.training_set_output[i] - local_CMAC_output
			print "error = " + str(error) + " glob neig idx = " +  str(global_neighborhood_index ) + " global idx = " + str(global_index)	
			
			#print "input = " + str(self.training_set_input[i])
			#print "output = " + str(self.training_set_output[i])
			#print "CMAC output = " + str(local_CMAC_output)
			if abs(error) <= self.error_threshold or global_neighborhood_index < -2:
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
			global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
			if global_neighborhood_index >= 0 and global_neighborhood_index < self.input_space_size :
				local_CMAC_output[i] = local_CMAC_output[i] + (self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])
		error = dataset_true_output[i] - local_CMAC_output[i]
		if abs(error) <= self.error_threshold :
			SuccessfulSamples = SuccessfulSamples + 1

	SuccessPercentage = (SuccessfulSamples/len(dataset_input))*100
	return local_CMAC_output,SuccessPercentage	
		

    def execute(self):
	self.create_datasets()
	self.train()
	self.training_set_CMAC_output,TrainingSuccessPercentage = self.test(self.training_set_input,self.training_set_output,self.training_set_global_indices)
	print "Training Success Percentage " + str(TrainingSuccessPercentage)
	
	self.testing_set_CMAC_output,TestingSuccessPercentage = self.test(self.testing_set_input,self.testing_set_true_output,self.testing_set_global_indices)
	print "Testing Success Percentage " + str(TestingSuccessPercentage)

    def plot_graphs(self):
	plotter.figure(figsize=(20,10))
	plotter.title('Lakshman')
	plotter.subplot(221)
	plotter.plot(self.training_set_input,self.training_set_CMAC_output,'ro')
	plotter.title('Training data , Input Space Size = ' + str(self.input_space_size) + ' Generalization_Factor = ' + str(self.generalization_factor))
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.subplot(223)
	plotter.plot(self.training_set_input,self.training_set_output,'ro')
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.subplot(222)
	plotter.plot(self.testing_set_input,self.testing_set_CMAC_output,'ro')
	plotter.title('Test Data')
	plotter.ylabel('CMAC Output')
	plotter.xlabel('Input')
	plotter.subplot(224)
	plotter.plot(self.input_space,self.output_space,'ro')
	plotter.plot(self.testing_set_input,self.testing_set_true_output,'ro')
	plotter.ylabel('True Output, sin(x)')
	plotter.xlabel('Input , x')
	plotter.show()


SineWaveCMAC= DiscreteCMAC(5,math.sin)
SineWaveCMAC.execute()
SineWaveCMAC.plot_graphs()
