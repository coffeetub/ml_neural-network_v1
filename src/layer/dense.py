import numpy as np 
import

class dense(layer):
    def init(self , input_size , output_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size , output_size)*0.01
        self.biases = np.zeros(1 , output_size)
        self.output_size = output_size
    
    def forward(self):
        pass

    def backward(Self):
        pass


