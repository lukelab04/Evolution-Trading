# MIT License

# Copyright (c) 2021 Luke LaBonte

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import numpy as np
import math

maxbias = 3

class model:
    def __init__(self, structure : list, mutationRate : float, generationSize : int):
        self.structure = structure
        self.mutationRate = mutationRate
        self.generationSize = generationSize

        self.generation = []

        for i in range(generationSize):
            self.generation.append(network(self.structure))
        

    def repopulate(self):
        self.generation.sort(key=lambda x: x.fitness)

        minval = abs(self.generation[0].fitness)
        maxval = self.generation[-1].fitness + minval

        for i in self.generation:
            i.fitness += minval
            if maxval != 0: changeby = (1 - i.fitness / maxval ) + self.mutationRate
            else: changeby = 1

            for layer in range(len(self.structure)):
                if layer == 0: continue
                for neuron in range(self.structure[layer]):
                    if random.uniform(0, 1) < changeby: i.biases[layer - 1][neuron] = random.uniform(-maxbias, maxbias)
                    for weight in range(self.structure[layer - 1]):
                        if(random.uniform(0, 1)) < changeby: i.weights[layer - 1][neuron][weight] = random.uniform(-maxbias, maxbias)

        print("Best performing:", maxval - minval)




class network:
    def __init__(self, structure):
        self.structure = structure
        self.weights = []
        self.biases = []
        self.values = []
        self.fitness = 0

        for layer in range(len(structure)):
            if layer == 0: continue
            tempbias = []
            tempweights = []

            for neuron in range(structure[layer]):
                tempbias.append(random.uniform(-maxbias, maxbias))
                tempneuronweights = []
                for weight in range(structure[layer - 1]):
                    tempneuronweights.append(random.uniform(-maxbias, maxbias))
                tempweights.append(tempneuronweights)

            
            self.weights.append(tempweights)
            self.biases.append(tempbias)


    def train(self, netinput):
        self.values = []
        for layer in range(len(self.structure)):
            if layer == 0:
                self.values = netinput
                continue
            self.values = np.dot(self.weights[layer - 1], self.values)
            np.add(self.values, self.biases[layer - 1])
            self.values = self.sigmoid(self.values)
        return list(self.values)

    
    def sigmoid(self, values):
        for value in range(len(values)):
            values[value] = round(values[value], 8)
            if(values[value]) > 600: values[value] = 1
            if(values[value]) < -600: values[value] = -1
            else:
                values[value] = 1 / (1 + math.pow(2.71828, -values[value]))
        return values
