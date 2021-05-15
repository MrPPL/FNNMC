import LSM

################
# calling functions
################
#Price American Put
timeStepsTotal = 50
normalizeStrike=40
putOption = LSM.Option(strike=1,payoffType="Put", timeToMat=1)
for spot1 in range(36,46,2):
    MarketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=spot1/normalizeStrike)
    learningPaths= LSM.generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    regressionCoefficient = LSM.findRegressionCoefficient(basisFuncTotal=5, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    pricingPaths= LSM.generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    priceAmerPut = LSM.priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
    print("Spot: ", spot1, "Price American Put: ", priceAmerPut)
    

#Price American call aka european call
timeStepsTotal = 50
normalizeStrike=40
putOption = LSM.Option(strike=1,payoffType="Call", timeToMat=1)
MarketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=1)
learningPaths= LSM.generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
regressionCoefficient = LSM.findRegressionCoefficient(basisFuncTotal=2, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
pricingPaths= LSM.generateSDEStockPaths(pathTotal=10**4, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
priceAmerCall = LSM.priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
print("Spot: ", MarketVariablesEx1.spot*normalizeStrike, "American Call Price: ", priceAmerCall)


