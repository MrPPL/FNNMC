
from context import SimulationPaths
from context import LSM
from context import Products
from SimulationPaths import SimGBMMultidimensions
import unittest   # The test framework
import numpy as np

class Test_LSM(unittest.TestCase):
    def testRegresion(self):
        """Use the regression example in LSM paper chapter 1
        """
        marketVariablesEX = LSM.MarketVariables(r=0.06,vol=0,spot=1)
        putOption = LSM.Option(strike=1.1, payoffType="Put",timeToMat=3)
        actualRegressionCoef = LSM.findRegressionCoefficient(simulatedPaths=learningPaths, basisFuncTotal=2, Option=putOption, MarketVariables=marketVariablesEX) 
        expectedRegressionCoef = np.array([[0,1.356,-1.813],
                                        [0,-3.335,2.983],
                                        [0,2.038,-1.070]])
        self.assertEqual(np.allclose(actualRegressionCoef[:,1:], expectedRegressionCoef[:,1:], 0.01), True)

if __name__ == '__main__':
    unittest.main()