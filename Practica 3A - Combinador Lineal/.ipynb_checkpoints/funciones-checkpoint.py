import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import meshgrid
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def graficar3D(X,Y,Z):
    fig = plt.figure(figsize=(10,8))
    # ax = fig.gca(projection='3d') # error por actualizacion de matplotlib
    ax = fig.add_subplot(projection = '3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                           edgecolors='k',
                           cmap=cm.Wistia,
                           linewidth=1, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def meshgridXY():
    RNG = 5.0 # Generador de Numeros Aleatorios
    x = np.arange(-RNG, RNG, RNG/20)
    y = np.arange(-RNG, RNG, RNG/20)
    #Se define la grilla de puntos
    X, Y = meshgrid(x, y)
    return X, Y


def optimize_function(dx_fun, dy_fun, fun):
    def inner(x, y, alfa, MAX_ITE=100, cota_error=1e-6):
        ite = 1
        v_new = fun(x, y)   # valor inicial
        v = v_new - 1 # fuerza entrada al while
        
        try:
            while ((ite < MAX_ITE) and (math.fabs(v - v_new) > cota_error)):
                # print('dif error', math.fabs(v - v_new))

                v = v_new
    
                grad_x = dx_fun(x, y)  # derivada respecto de x en punto actual
                grad_y = dy_fun(x, y)  # derivada respecto de y en punto actual
    
                x = x - alfa * grad_x  # avanza en dirección al gradiente en x
                y = y - alfa * grad_y  # avanza en dirección al gradiente en y
    
                v_new = fun(x, y)  # calcula valor en nueva posición
                ite = ite + 1
    
            print("iteraciones = %d   x= %.5f   y=%.5f   v=%.8f" % (ite, x, y, v_new))
        except OverflowError as e:
            print("Oooops: después de la iteracion %d algún valor tiende a infinito:\n x = %.1e \n y = %.1e" % (ite, x,y))
        print('Dif error: ', math.fabs(v - v_new))
    return inner

