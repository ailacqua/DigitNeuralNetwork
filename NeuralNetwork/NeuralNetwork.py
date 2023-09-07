"""
Neural network to recognize digits

Uses 5x3 tile input to recognize digits using backpropogation and sigmoid function.

Author: Alex Ilacqua
Date: 4/9/2023
"""
import random
import math
import csv

#method to read files
def readFile(filePath):
    data = []
    #opens file from path
    f = open(filePath)
    #reads each row
    for row in csv.reader(f):
        #if row is not blank then it is added to the data list
        if row != []:
            data.append(row)
    return data

#finding expected output integer from expected output
def findExpectedOutput(data):
    for row in data:
        #loops through expected output
        for i in range(0,len(row[1])):
            #appends index where output is 1 to the row of the data for testing
            if int(row[1][i]) == 1:
                row.append(i)
                break
    return data
         
#method generates hidden weights between -0.2 and 0.2
def createHiddenWeights(inputNum,hiddenNum):
    hiddenLayer = []
    #row represents input node number
    for r in range(0,inputNum):
        tempRow = []
        #column represents hidden node number
        for c in range(0,hiddenNum):
            #random weight generated and added to matrix
            tempRow.append(random.uniform(-0.2,0.2))
        hiddenLayer.append(tempRow)
    return hiddenLayer

#method generates output weights between -0.2 and 0.2
def createOutputWeights(hiddenNum,outputNum):
    outputLayer = []
    #row represents hidden node number
    for r in range(0,hiddenNum):
        tempRow = []
        #column represents output node number
        for c in range(0,outputNum):
            #random weight generated and added to matrix
            tempRow.append(random.uniform(-0.2,0.2))
        outputLayer.append(tempRow)
    return outputLayer

#selects random input from sample data
def selectInput(sample):
    randomVal = random.randint(0,len(sample)-1)
    return sample[randomVal]

#calculates sigmoid value of input list using formula
def calcInputSigmoid(values):
    sigmoidList = []
    for i in range(0,len(values)):
        #application of sigmoid formula
        sigmoidList.append((1.0)/(1.0 + (math.exp(-int(values[i])))))
    return sigmoidList

#calculates sigmoid value of list using formula
def calcSigmoid(values):
    sigmoidList = []
    for i in range(0,len(values)):
        #application of sigmoid formula
        sigmoidList.append((1.0)/(1.0 + (math.exp(-values[i]))))
    return sigmoidList

#calculates value of hidden nodes
def calcHiddenNodes(hiddenWeights,inputSigmoid):
    hiddenNodes = []
    for c in range(0,len(hiddenWeights[0])):
        temp = 0
        for r in range(0,len(hiddenWeights)):
            #summation of input times weights for each hidden node
            temp = temp + (hiddenWeights[r][c]*inputSigmoid[r])
        hiddenNodes.append(temp)
    return hiddenNodes

#calculates value of output nodes
def calcOutputNodes(outputWeights,hiddenSigmoid):
    outputNodes = []
    for c in range(0,len(outputWeights[0])):
        temp = 0
        for r in range(0,len(outputWeights)):
            #summation of input times weights for each output node
            temp = temp + (outputWeights[r][c]*hiddenSigmoid[r])
        outputNodes.append(temp)
    return outputNodes

#finds delta value of output nodes
def calcOutputDelta(outputSigmoid, expectedOutput):
    outputDelta = []
    for i in range(0,len(outputSigmoid)):
        expOut = 0
        #if expected output is same as node number, exp out is 1
        if expectedOutput == i:
            expOut = 1
        #application of delta formula
        outputDelta.append(outputSigmoid[i] * (1-outputSigmoid[i]) * (expOut-outputSigmoid[i]))
    return outputDelta

#calculates adjustment to output weights
def calcOutputAdjustment(learningRate, outputDelta, hiddenSigmoid):
    outputAdjustment = []
    for r in range(0,len(hiddenSigmoid)):
        temp = []
        for c in range(0,len(outputDelta)):
            #application of deltaWij formula
            temp.append(learningRate * outputDelta[c] * hiddenSigmoid[r])
        outputAdjustment.append(temp)
    return outputAdjustment

#adjusts weights using old weights and adjustment
def adjustNodes(oldWeights,adjustment):
    newWeights = []
    for r in range(0,len(oldWeights)):
        temp = []
        for c in range(0,len(oldWeights[0])):
            #adds adjustment to each weight
            temp.append(oldWeights[r][c] + adjustment[r][c])
        newWeights.append(temp)
    return newWeights

#calculates hidden alpha value
def calcHiddenAlpha(outputDelta,outputWeights):
    hiddenAlpha = []
    for r in range(0,len(outputWeights)):
        temp = 0
        for c in range(0,len(outputDelta)):
            #finds summation of all weights from hidden node to output node times delta output
            temp = temp + (outputWeights[r][c] * outputDelta[c])
        hiddenAlpha.append(temp)
    return hiddenAlpha

#calculates hidden delta value
def calcHiddenDelta(hiddenAlpha,hiddenSigmoid):
    hiddenDelta = []
    for i in range(0,len(hiddenSigmoid)):
        #applies delta formula to calculate delta value
        hiddenDelta.append(hiddenSigmoid[i]*(1-hiddenSigmoid[i])*hiddenAlpha[i])
    return hiddenDelta

#calculates hidden weights adjustment
def calcHiddenAdjustment(learningRate,hiddenDelta,inputSigmoid):
    hiddenAdjustment = []
    for r in range(0,len(inputSigmoid)):
        temp = []
        for c in range(0,len(hiddenDelta)):
            #application of adjustment formula
            temp.append(learningRate * hiddenDelta[c] * inputSigmoid[r])
        hiddenAdjustment.append(temp)
    return hiddenAdjustment
    
def main():
    #declaration of iteration size, learning rate, and node numbers
    iterationSize = 30000
    learningRate = 0.25
    inputNum = 15
    hiddenNum = 12
    outputNum = 10
    sampleDataPath = "/Users/ailacqua/Ansari CS Programs/Python Programs/NeuralNetwork/SampleData.txt"#INSERT SAMPLE DATA PATH FILE
    validationDataPath = "/Users/ailacqua/Ansari CS Programs/Python Programs/NeuralNetwork/ValidationData.txt"#INSERT SAMPLE DATA PATH FILE
    
    #reading sample data and setting expected output
    sampleData = readFile(sampleDataPath)
    sampleData = findExpectedOutput(sampleData)
    
    #reading validation data and setting expected output
    validationData = readFile(validationDataPath)
    validationData = findExpectedOutput(validationData)
    
    #instatiation of hidden and output weights
    hiddenWeights = createHiddenWeights(inputNum, hiddenNum)
    outputWeights = createOutputWeights(hiddenNum, outputNum)
    
    #forward and backward propogates until the iteration size is reached
    for i in range(0,iterationSize):
        #forward prop
        inputNodes = selectInput(sampleData)
        inputSigmoid = calcInputSigmoid(inputNodes[0])
        hiddenNodes = calcHiddenNodes(hiddenWeights,inputSigmoid)
        hiddenSigmoid = calcSigmoid(hiddenNodes)
        outputNodes = calcOutputNodes(outputWeights, hiddenSigmoid)
        outputSigmoid = calcSigmoid(outputNodes)
        
        #back prop
        outputDelta = calcOutputDelta(outputSigmoid, inputNodes[2])
        outputAdjustment = calcOutputAdjustment(learningRate,outputDelta,hiddenSigmoid)
        outputWeights = adjustNodes(outputWeights,outputAdjustment)
        hiddenAlpha = calcHiddenAlpha(outputDelta,outputWeights)
        hiddenDelta = calcHiddenDelta(hiddenAlpha,hiddenSigmoid)
        hiddenAdjustment = calcHiddenAdjustment(learningRate,hiddenDelta,inputSigmoid)
        hiddenWeights = adjustNodes(hiddenWeights,hiddenAdjustment)
    
    #declaration of counter for accuracy
    accuracyCount = 0
    #forward propogates for all validation data to find output
    for i in range(0,len(validationData)):
        #forward prop
        inputNodes = validationData[i]
        inputSigmoid = calcInputSigmoid(inputNodes[0])
        hiddenNodes = calcHiddenNodes(hiddenWeights,inputSigmoid)
        hiddenSigmoid = calcSigmoid(hiddenNodes)
        outputNodes = calcOutputNodes(outputWeights, hiddenSigmoid)
        outputSigmoid = calcSigmoid(outputNodes)
        #finds highest outputsigmoid value node number
        best = outputSigmoid.index(max(outputSigmoid))
        #printing input and expected output and produced output
        print("Input: ",inputNodes[0])
        print("Expected output: ",inputNodes[2])
        print("Actual output: ",best)
        #if output is the expected output then accuracy count incremented
        if best == inputNodes[2]:
            accuracyCount = accuracyCount + 1
    #percent accuracy printed
    print("Percent accuracy:",(accuracyCount/26)*100)
    
#calling main method
main()