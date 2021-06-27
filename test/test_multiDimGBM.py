import unittest   # The test framework
import numpy as np
from context import SimulationPaths
from SimulationPaths import GBMMultiDim
from context import Products
from Products import MarketVariables

class Test_GBMSim(unittest.TestCase):
    def testInitialMatrix(self):
        pathTotal=10
        spot=[40,50]
        initialMatrix = GBMMultiDim.initialState(pathTotal, spot, len(spot))
        expectedMatrix=np.array([[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[40,50]])
        self.assertTrue(np.allclose(initialMatrix, expectedMatrix))

    def testCholeskyDecompositionSameVol(self):
        vol = 0.2
        rho = 0.1
        spot = [40,50]
        lowerTriangularMatrix = GBMMultiDim.choleskyLowerTriangular(assetsTotal=len(spot), vol=vol, correlation=rho)
        expectedMatrix=np.array([[0.2,0],[0.02, 0.1989975]])
        self.assertTrue(np.allclose(lowerTriangularMatrix, expectedMatrix))

    def testCholeskyDecompositionDifferentVol(self):
        vol = [0.2,0.1]
        rho = 0.1
        spot = [40,50]
        lowerTriangularMatrix = GBMMultiDim.choleskyLowerTriangular(assetsTotal=len(spot), vol=vol, correlation=rho)
        expectedMatrix=np.array([[0.2,0],[0.01, 0.09949874]])
        self.assertTrue(np.allclose(lowerTriangularMatrix, expectedMatrix))

    def testGBMUpdate(self):
        marketVariables = Products.MarketVariables(r=0.1, vol=0.2, spot=[40,50], dividend=0.1, correlation=0.1)
        lowerTriangularMatrix = GBMMultiDim.choleskyLowerTriangular(assetsTotal=len(marketVariables.spot), vol= marketVariables.vol, correlation=marketVariables.correlation)
        timeIncrement = 2
        normVec = [1.0491025, 0.7807751]
        #first asset update
        stockPrice = GBMMultiDim.GBMUpdate(spot=marketVariables.spot[0], marketVariables=marketVariables, timeIncrement=timeIncrement, 
            lowerTriangleMatrixRow=lowerTriangularMatrix[0,:], normVec=normVec)
        expectedStockPrice = 40*np.exp((marketVariables.r-marketVariables.dividend-np.square(marketVariables.vol)*0.5)*timeIncrement + 
            np.sqrt(timeIncrement)*(marketVariables.vol*normVec[0]))
        self.assertAlmostEqual(first=stockPrice,second=expectedStockPrice,places=3)
        #second asset update
        stockPrice = GBMMultiDim.GBMUpdate(spot=marketVariables.spot[1], marketVariables=marketVariables, timeIncrement=timeIncrement, 
            lowerTriangleMatrixRow=lowerTriangularMatrix[1,:], normVec=normVec)
        expectedStockPrice = 50*np.exp((marketVariables.r-marketVariables.dividend-np.square(marketVariables.vol)*0.5)*timeIncrement + 
            np.sqrt(timeIncrement)*(marketVariables.vol*marketVariables.correlation*normVec[0] + marketVariables.vol*np.sqrt(1-marketVariables.correlation**2)*normVec[1]))
        self.assertAlmostEqual(first=stockPrice,second=expectedStockPrice,places=3)

if __name__ == '__main__':
    unittest.main()