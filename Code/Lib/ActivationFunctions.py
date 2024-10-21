import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return z * (1 - z)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - z ** 2

# if __name__ == "__main__":
#     z = np.array([-2, -1, 0, 1, 2])
#     print("Sigmoid:", ActivationFunctions.sigmoid(z))
#     print("Sigmoid Derivative:", ActivationFunctions.sigmoid_derivative(ActivationFunctions.sigmoid(z)))
#     print("ReLU:", ActivationFunctions.relu(z))
#     print("ReLU Derivative:", ActivationFunctions.relu_derivative(z))
#     print("Tanh:", ActivationFunctions.tanh(z))
#     print("Tanh Derivative:", ActivationFunctions.tanh_derivative(ActivationFunctions.tanh(z)))
