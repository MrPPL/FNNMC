import LSM
import FFMC
timeStepsTotal = 50
normalizeStrike=40
putOption = LSM.Option(strike=1,payoffType="Put", timeToMat=1)
marketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=40/normalizeStrike)
pricingPaths = LSM.generateSDEStockPaths(pathTotal=1*10**1, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=marketVariablesEx1)
hyperparameters = FFMC.Hyperparameters(learningRate=0.001, hiddenlayer1=10, hiddenlayer2=10, epochs=10, batchSize=10**4)
price = FFMC.priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEx1, hyperparameters=hyperparameters)*normalizeStrike
print("Price of American put: ", price)