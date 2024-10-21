import numpy as np 

from ActivationFunctions import ActivationFunctions

class MultiLayerPerceptron:
    def __init__(self, layers, activation='sigmoid', learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.01
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activate(self, z):
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid(z)
        elif self.activation == 'relu':
            return ActivationFunctions.relu(z)
        elif self.activation == 'tanh':
            return ActivationFunctions.tanh(z)
        else:
            raise ValueError("Unsupported activation function.")

    def activate_derivative(self, z):
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid_derivative(z)
        elif self.activation == 'relu':
            return ActivationFunctions.relu_derivative(z)
        elif self.activation == 'tanh':
            return ActivationFunctions.tanh_derivative(z)
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, X):
        self.a = [X]
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], weight) + bias
            a = self.activate(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y):
        m = y.shape[0]
        delta = self.a[-1] - y
        
        for i in reversed(range(len(self.weights))):
            weight_gradient = np.dot(self.a[i].T, delta) / m
            bias_gradient = np.sum(delta, axis=0, keepdims=True) / m
            delta = np.dot(delta, self.weights[i].T) * self.activate_derivative(self.a[i])
            
            self.weights[i] -= self.learning_rate * weight_gradient
            self.biases[i] -= self.learning_rate * bias_gradient

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = np.mean((self.a[-1] - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MultiLayerPerceptron(layers=[2, 2, 1], activation='sigmoid', learning_rate=0.1)
    
    mlp.train(X, y, epochs=1000)

    print("Predictions:")
    for x in X:
        print(f"Input: {x}, Output: {mlp.forward(x.reshape(1, -1))}")
