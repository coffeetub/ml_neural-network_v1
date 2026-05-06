import numpy as np
import os

class Utils:
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def hidden_delta(delta_next, W_next, z):
        return (delta_next @ W_next.T) * Utils.relu_derivative(z)

    @staticmethod
    def compute_dW(a_prev, delta):
        return np.outer(a_prev, delta)

    @staticmethod
    def compute_db(delta):
        return delta
    
    @staticmethod
    def build(label):
        y = np.zeros(10)
        y[label] = 1
        return y




class ActivationError(Exception):
    pass


class layer_activation:
    def __init__(self, index, L):
        self.index = index
        self.L = L
       
        self.activation = np.zeros(self.L)
        self.z = np.zeros(self.L)


class layer_weights:
    def __init__(self, index, L1, L2):
        self.index = index
        self.L1 = L1
        self.L2 = L2
        path = f'../models/w{index}.npy'

        if not os.path.exists(path):
            self.weights = np.random.randn(self.L1, self.L2) * 0.01
            np.save(path, self.weights)
        else:
            loaded_weights = np.load(path)
            if loaded_weights.size == 0:
                self.weights = np.random.randn(self.L1, self.L2) * 0.01
                np.save(path, self.weights)
            else:
                self.weights = loaded_weights


class layer_biases:
    def __init__(self, index, L):
        self.index = index
        self.L = L
        path = f'../models/b{index}.npy'

        if not os.path.exists(path):
            self.biases = np.zeros(self.L)
            np.save(path, self.biases)
        else:
            loaded_bias = np.load(path)
            if loaded_bias.size == 0:
                self.biases = np.zeros(self.L)
                np.save(path, self.biases)
            else:
                self.biases = loaded_bias


class layer_data:
    def __init__(self, index, L1, L2):
        self.index = index
        self.L1 = L1
        self.L2 = L2
        self.activation = layer_activation(index, L2)
        self.biases = layer_biases(index, L2)
        self.weights = layer_weights(index, L1, L2)

class input_layer:
    def __init__(self, L):
        self.L = L
        self.activation = layer_activation(0, L)  

class data_processing:
    def __init__(self, layer, previous_layer , activation_fn):
        self.layer = layer
        self.prev_layer = previous_layer
        self.activation_fn = activation_fn

    def calculate_activation(self):
        
        z = (
            np.dot(self.prev_layer.activation.activation, self.layer.weights.weights)
            + self.layer.biases.biases
        )
        self.layer.activation.z = z
        self.layer.activation.activation = self.activation_fn(z)

    def save_data(self):
        np.save(f'../models/w{self.layer.index}.npy', self.layer.weights.weights)
        np.save(f'../models/b{self.layer.index}.npy', self.layer.biases.biases)



class training:
    def __init__(self, layer , epoch , learning_rate , Processors , n , Path):
        self.layers = layer
        self.epoch = epoch
        self.a = learning_rate
        self.path = Path
        self.n = n
        self.data_processors = Processors


    def forward_propagation(self, input_data):
        
        self.layers[0].activation.activation = input_data  

        for i in range(0 , self.n - 1):
            dp = self.data_processors[i]
            dp.calculate_activation()
    
    def backward_propagation(self, y_true):
        y_pred = self.layers[-1].activation.activation
        delta = y_pred - y_true

        for i in range(len(self.layers)-1, 0, -1):
            curr = self.layers[i]
            prev = self.layers[i-1]

            dW = Utils.compute_dW(prev.activation.activation, delta)
            db = Utils.compute_db(delta)

            if i > 1:
                new_delta = np.dot(delta, curr.weights.weights.T) * Utils.relu_derivative(prev.activation.z)

            curr.weights.weights -= self.a * dW
            curr.biases.biases   -= self.a * db

            if i > 1:
                delta = new_delta


    def train(self):
        dataset = np.loadtxt(self.path , delimiter=',' , skiprows=1)

        for epoch in range(self.epoch):
            total_loss = 0
            np.random.shuffle(dataset)
            self.a = self.a * 0.95
            for row in dataset:
                input_data = row[1:] / 255.0
                y_true = Utils.build(int(row[0]))
                self.forward_propagation(input_data)
                y_predicted = self.layers[3].activation.activation
                total_loss += -np.sum(y_true * np.log(y_predicted + 1e-8))
                self.backward_propagation(y_true)
            print(f"Epoch {epoch+1}/{self.epoch} loss: {total_loss/len(dataset):.4f}")
            for dp in self.data_processors:
                dp.save_data()

    def test(self, test_path):
        test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)  
        print(f"Loading test data from: {test_path}")
        print(f"Test data shape: {test_data.shape}")                  
        correct = 0
        for row in test_data:
            x = row[1:] / 255.0
            self.forward_propagation(x)
            pred = np.argmax(self.layers[-1].activation.activation)
            if pred == int(row[0]):
                correct += 1
        print(f"Test Accuracy: {correct/len(test_data)*100:.2f}%")

def predict(input_data , layers , dp):
    layers[0].activation.activation = input_data  

    for i in dp:
        i.calculate_activation()
    
    return np.argmax(layers[-1].activation.activation)
    

layer1 = layer_data(1 , 784 , 128)
layer2 = layer_data(2, 128, 64)
layer3 = layer_data(3, 64, 10)
layer0 = input_layer(784)

model = [layer0 , layer1 , layer2 , layer3]
dp = [data_processing(layer1, layer0, Utils.relu),
      data_processing(layer2, layer1, Utils.relu),
      data_processing(layer3, layer2, Utils.softmax)]


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'training_data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'test_data.csv')

os.makedirs('../models', exist_ok=True)

#t = training(model, 10, 0.01, dp, len(model), DATA_PATH)
#t.train()
#t.test(TEST_PATH)