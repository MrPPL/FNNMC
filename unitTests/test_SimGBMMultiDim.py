
import SimGBMMultidimensions # The code to test
import numpy as np
import pytest

#####
#
########


def test_initialState():
    pathsTotal = 3 
    spots = [30,40]
    assetsTotal=len(spots)
    expectedInitialState = np.array([[30,40], [30,40], [30,40]])
    calculatedInitialState = SimGBMMultidimensions.initialState(pathsTotal=pathsTotal, spots=spots, assetsTotal=assetsTotal)
    assert np.allclose(calculatedInitialState ,expectedInitialState, 0.0001) == True

def test_GBMUpdate(spot, marketVariables, timeIncrement, lowerTriangleMatrixRow):
    pass

def test_generateCovarianceMatrix(assetsTotal, vol, correlation):
    pass

def test_updateState(currentState, marketVariables, timeIncrement):
    pass

def test_simulatePaths(timeStepsTotal, pathsTotal, marketVariables, timeToMat):
    pass

##############
#
############


def test_PricePhase():
    pass
    #assert priceAmerOption == pytest.approx(0.1144, 0.01)
