import numpy as np
import time

from matplotlib import pylab as plt


class Perceptron(object):

    def __init__(self, alpha=0.01, n_iter=50, random_state=None, title=['X1','X2']):
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state #-- asignar el valor 1 para fijar la semilla por defecto es aleatorio
        self.title = title

    def fit(self, X, T, W, b):
        self.w_ = W 
        self.b_ = b 


        self.errors_ = []
        errors=1
        i = 0
        while ((i<self.n_iter) and (errors > 0.0)):
            errors = 0
            for xi, target in zip(X, T):
                update = self.alpha * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update

                errors += int(update != 0.0)
            self.errors_.append(errors)

            
            i = i + 1
        return self, i

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def activacion(self, W, X, b):
        # w1*x1 + w2*x2 + ⋯ + wn*xn
        suma_ponderada = (W * X).sum() + b
        return (suma_ponderada > 0) *1
        
