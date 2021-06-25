from context import SimulationPaths
from SimulationPaths import GBM
from context import LSM # The code to test
from context import Products
import unittest   # The test framework
import numpy as np

class Test_LSM(unittest.TestCase):
    def testRegresion(self):
        """Use the regression example in LSM paper chapter 1
        """
        learningPaths = np.array([[1,1.09,1.08,1.34],
                                [1, 1.16,1.26,1.54],
                                [1, 1.22, 1.07, 1.03],
                                [1, 0.93, 0.97, 0.92],
                                [1, 1.11, 1.56, 1.52],
                                [1, 0.76, 0.77, 0.90],
                                [1, 0.92, 0.84, 1.01],
                                [1, 0.88, 1.22, 1.34]])
        marketVariablesEX = Products.MarketVariables(r=0.06,vol=0,spot=1)
        putOption = Products.Option(strike=1.1, typeOfContract="Put",timeToMat=3)
        actualRegressionCoef = LSM.findRegressionCoefficient(simulatedPaths=learningPaths, basisFuncTotal=2, Option=putOption, MarketVariables=marketVariablesEX) 
        expectedRegressionCoef = np.array([[0,1.356,-1.813],
                                        [0,-3.335,2.983],
                                        [0,2.038,-1.070]])
        self.assertEqual(np.allclose(actualRegressionCoef[:,1:], expectedRegressionCoef[:,1:], 0.01), True)


    def testPricing(self):
        """Follow the example in chapter 1 in LSM paper
        """
        marketVariablesEX = Products.MarketVariables(r=0.06,vol=0,spot=1)
        putOption = Products.Option(strike=1.1, typeOfContract="Put",timeToMat=3)
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
        self.assertAlmostEqual(first=priceAmerOption,second=0.1144,places=4)
    def testPut(self):
        """Follow chapter 3 put example in LSM paper
        """
        timeStepsPerYear = 50
        normalizeStrike=40
        putOption = Products.Option(strike=1,typeOfContract="Put", timeToMat=1)
        MarketVariablesEx1 = Products.MarketVariables(r=0.06,vol=0.2, spot=36/normalizeStrike)
        pathTotal = 10**5
        learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
        regressionCoefficient = LSM.findRegressionCoefficient(basisFuncTotal=5, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
        pricingPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
        priceAmerPut = LSM.priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
        self.assertAlmostEqual(first=priceAmerPut,second=4.487,places=1)

    def testDividendCall(self):
        """Price American call option with dividend replicated from "Pricing American-style securities using simulation" from Broadie and Glasserman
        """
        timeStepsPerYear = 3
        normalizeStrike=100
        spot = 90
        callOption = Products.Option(strike=1,typeOfContract="Call", timeToMat=1)
        MarketVariablesEx1 = Products.MarketVariables(r=0.05,vol=0.2, spot=spot/normalizeStrike, dividend=0.1)
        pathTotal = 10**6
        learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=callOption.timeToMat, MarketVariables=MarketVariablesEx1)
        regressionCoefficient = LSM.findRegressionCoefficient(basisFuncTotal=5, Option=callOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
        pricingPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=callOption.timeToMat, MarketVariables=MarketVariablesEx1)
        priceAmerCall = LSM.priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=callOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
        self.assertAlmostEqual(first=priceAmerCall,second=2.303,places=1)

if __name__ == '__main__':
    unittest.main()