import LSM    # The code to test
import numpy as np
import pytest

def test_RegressionPhase():
    learningPaths = np.array([[1,1.09,1.08,1.34],
                             [1, 1.16,1.26,1.54],
                             [1, 1.22, 1.07, 1.03],
                             [1, 0.93, 0.97, 0.92],
                             [1, 1.11, 1.56, 1.52],
                             [1, 0.76, 0.77, 0.90],
                             [1, 0.92, 0.84, 1.01],
                             [1, 0.88, 1.22, 1.34]])
    marketVariablesEX = LSM.MarketVariables(r=0.06,vol=0,spot=1)
    putOption = LSM.Option(strike=1.1, payoffType="Put",timeToMat=3)
    actualRegressionCoef = LSM.findRegressionCoefficient(simulatedPaths=learningPaths, basisFuncTotal=2, Option=putOption, MarketVariables=marketVariablesEX) 
    expectedRegressionCoef = np.array([[0,1.356,-1.813],
                                      [0,-3.335,2.983],
                                      [0,2.038,-1.070]])
    assert np.allclose(actualRegressionCoef, expectedRegressionCoef, 0.01) == True

def test_PricePhase():
    marketVariablesEX = LSM.MarketVariables(r=0.06,vol=0,spot=1)
    putOption = LSM.Option(strike=1.1, payoffType="Put",timeToMat=3)
    regressionCoef = np.array([[0,1.356,-1.813],
                                      [0,-3.335,2.983],
                                      [0,2.038,-1.070]])
    pricingPaths = np.array([[1,1.09,1.08,1.34],
                             [1, 1.16,1.26,1.54],
                             [1, 1.22, 1.07, 1.03],
                             [1, 0.93, 0.97, 0.92],
                             [1, 1.11, 1.56, 1.52],
                             [1, 0.76, 0.77, 0.90],
                             [1, 0.92, 0.84, 1.01],
                             [1, 0.88, 1.22, 1.34]])
    priceAmerOption = LSM.priceAmericanOption(coefficientMatrix=regressionCoef, simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEX)
    assert priceAmerOption == pytest.approx(0.1144, 0.01)