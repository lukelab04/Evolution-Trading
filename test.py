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


import Network
import csv

trainSize = 5

stockData = []
tempStocks = []

origprice = 0
endprice = 0

#Helper method to fill stockData with the closing price of whichever stock we chose, broken up into arrays of size trainSize
def fillStockData(filename : str):

    global stockData, tempStocks, origprice, endprice

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        linecount = 0

        for row in reader:
            if linecount == 0:
                linecount+=1
                continue
            elif linecount == 1: origprice = row[4]
            tempStocks.append(float(row[4]))
            if len(tempStocks) == trainSize:
                stockData.append(tempStocks)
                tempStocks = []
            linecount+=1
        endprice = row[4]

fillStockData('AMD.csv')

#Create network object
net = Network.model([5, 32, 16, 3], 0.0001, 30)

#Loop 25 times for 25 training iterations
for k in range(25):
    #Loop though every network in the generation
    for i in range(net.generationSize):
        startmoney = 0
        currentMoney = startmoney
        bought = False

        #Loop though training sets in stockData
        for j in stockData:
            #Get the output of the network
            netoutput = net.generation[i].train(j)
            #Outupt is in the format [out1, out2, out3]. One of those will be the largest, and the index of that is the output we're looking for.
            if netoutput.index(max(netoutput)) == 0 and not bought:
                currentMoney -= j[-1]
                bought = True
            elif netoutput.index(max(netoutput)) == 1 and bought:
                currentMoney += j[-1]
                bought = False
            else: continue

        #Set the fitness to the money made
        net.generation[i].fitness = currentMoney - startmoney

    #Regenerate the model
    net.repopulate()

bestnetwork = net.generation[-1]

currentMoney = 0
startmoney = 0

#Run same simulation as above, but only with the best network
for j in stockData:
    netoutput = bestnetwork.train(j)
    
    if netoutput.index(max(netoutput)) == 0 and not bought:
        currentMoney -= j[-1]
        bought = True
        print("bought - ", j[-1])
    elif netoutput.index(max(netoutput)) == 1 and bought:
        currentMoney += j[-1]
        bought = False
        print("sold - ", j[-1])
    else: continue

#Show how much the network made
print(currentMoney - startmoney)

#Show how much the stock increased
print("Stock increase:", float(endprice) - float(origprice))