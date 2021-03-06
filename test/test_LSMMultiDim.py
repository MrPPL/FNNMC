from context import LSMBasketOption
from context import Products
import unittest   # The test framework
import numpy as np

class Test_LSMMultiDim(unittest.TestCase):
    def testRegresion(self):
        """Use the regression example in LSM paper chapter 1
        """
        learningPaths = np.array([[[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1.1, 0.9],
        [1.0, 1.0],
        [0.9, 0.9],
        [0.8, 1.2],
        [1.0, 1.05]],

       [[1.20, 0.85],
        [1.02, 0.93],
        [0.85, 0.87],
        [0.95, 1.11],
        [1.05, 1.07]]])
        marketVariablesEX = Products.MarketVariables(r=0.00,vol=0,spot=1)
        callMaxOption = Products.Option(strike=1.0, typeOfContract="CallMax",timeToMat=2)
        actualRegressionCoef = LSMBasketOption.findRegressionCoefficient(simulatedPaths=learningPaths, Option=callMaxOption, MarketVariables=marketVariablesEX, regressionBasis="Base") 
        expectedRegressionCoef = np.array([[0,3.87],
                                        [0,-1.7],
                                        [0,-2.00]])
        self.assertEqual(np.allclose(actualRegressionCoef[:,1:], expectedRegressionCoef[:,1:], 0.01), True)
    def testPricing(self):
        """Use the regression example in LSM paper chapter 1
        """
        learningPaths = np.array([[[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1.1, 0.9],
        [1.0, 1.0],
        [0.9, 0.9],
        [0.8, 1.2],
        [1.0, 1.05]],

       [[1.20, 0.85],
        [1.02, 0.93],
        [0.85, 0.87],
        [0.95, 1.11],
        [1.05, 1.07]]])
        marketVariablesEX = Products.MarketVariables(r=0.00,vol=0,spot=1)
        callMaxOption = Products.Option(strike=1.0, typeOfContract="CallMax",timeToMat=2)

        regressionCoef = np.array([[0,3.87], [0,-1.7], [0,-2]])

        actualPrice = LSMBasketOption.priceAmericanOption(coefficientMatrix=regressionCoef, simulatedPaths=learningPaths, Option=callMaxOption, MarketVariables=marketVariablesEX, regressionBasis="Base")
        expectedPrice = 0.098
        self.assertAlmostEqual(first=actualPrice,second=expectedPrice,places=3)

if __name__ == '__main__':
    unittest.main()