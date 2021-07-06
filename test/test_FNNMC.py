
from context import LSMBasketOption
from context import Products
from context import FNNMC
import unittest   # The test framework
import numpy as np


class Test_FNNMC(unittest.TestCase):
    def test_PricePhase(self):
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
        hyperParameters = FNNMC.Hyperparameters(learningRate=0.001, hiddenlayer1=100, hiddenlayer2=100, epochs=10, batchSize=2)
        FNNMC.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariablesEX, hyperparameters=hyperParameters) 
        pricingPaths = np.array([[1,1.09,1.08,1.34],
                                [1, 1.16,1.26,1.54],
                                [1, 1.22, 1.07, 1.03],
                                [1, 0.93, 0.97, 0.92],
                                [1, 1.11, 1.56, 1.52],
                                [1, 0.76, 0.77, 0.90],
                                [1, 0.92, 0.84, 1.01],
                                [1, 0.88, 1.22, 1.34]])

        priceAmerOption = FNNMC.priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEX, 
            hyperparameters=hyperParameters)
        self.assertAlmostEqual(first=priceAmerOption,second=0.1144,places=2)


if __name__ == '__main__':
    unittest.main()