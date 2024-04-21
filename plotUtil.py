import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

class plotter():

    def __init__(self, title, label, xlabel, ylabel, y, X = None):
        self.fig, self.ax = plt.subplots()
        self.y = y
        if X == None:
            self.X = range(len(y))
            self.ax.plot(self.X, y, color = "green", label = label)
        else:
            self.X = X
            self.ax.plot(X, y, color = "green", label = label)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.xaxis.set_major_locator(ticker.AutoLocator())
    
    def add_predicted(self, label, color, yPredicted, X = None, linestyle = ':'):
        if X == None:
            self.ax.plot(self.X, yPredicted, color = color, linestyle = linestyle, label = label)
        else:
            newX = [self.X[-1]]
            newX.extend(X)
            newY = [self.y[-1]]
            newY.extend(yPredicted)
            self.ax.plot(newX, newY, color = color, linestyle = linestyle, label = label)
        self.ax.legend()
    
    def get_plot(self, verbose = False):
        if verbose:
            plt.show()
        else:
            self.fig.show()