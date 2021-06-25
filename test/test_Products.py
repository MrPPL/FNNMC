from context import Products
import unittest   # The test framework
import numpy as np

class Test_Products(unittest.TestCase):
    def testPutContract(self):
        Put = Products.Option(timeToMat=1, strike=40, typeOfContract="Put")
        MarketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=38)
        intrinsicValue = Put.payoff(MarketVariables.spot)
        self.assertEqual(intrinsicValue,2)

    def testCallContract(self):
        Call = Products.Option(timeToMat=1, strike=40, typeOfContract="Call")
        MarketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=45)
        intrinsicValue = Call.payoff(MarketVariables.spot)
        self.assertEqual(intrinsicValue,5)

    def testGeometricPutContract(self):
        GeometricPut = Products.Option(timeToMat=1, strike=41, typeOfContract="PutGeometricAverage")
        MarketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=[80,20])
        intrinsicValue = GeometricPut.payoff(MarketVariables.spot)
        self.assertEqual(intrinsicValue,1)
        
    def testMaximumCallContract(self):
        CallMax = Products.Option(timeToMat=1, strike=92, typeOfContract="CallMax")
        MarketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=[90,100])
        intrinsicValue = CallMax.payoff(MarketVariables.spot)
        self.assertEqual(intrinsicValue,8)

if __name__ == '__main__':
    unittest.main()
