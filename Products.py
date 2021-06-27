import numpy as np

class MarketVariables:
    # The object holds the "observable" market variables
    def __init__(self, r=0, vol=0, spot=0, dividend=0, correlation=0):
        self.r = r
        self.vol = vol
        self.spot = spot
        self.dividend = dividend
        self.correlation = correlation


class Option():
    def __init__(self, timeToMat=0, strike=0, typeOfContract="Put"):
         self.timeToMat = timeToMat
         self.strike = strike
         self.typeOfContract = typeOfContract

    def __str__(self):
        return 'Time to Maturity %.d years\nStrike %.d \nType of Contract %s Option' % (self.timeToMaturity, self.strike, self.typeOfContract)

    def payoff(self, spots):
        if (self.typeOfContract=="Put"):
            return np.maximum(0, self.strike - spots)
        elif (self.typeOfContract=="Call"):
            return np.maximum(0,spots - self.strike)
        elif (self.typeOfContract=="PutGeometricAverage"):
            return np.maximum(0, self.strike - np.prod(spots)**(1/len(spots)))
        elif(self.typeOfContract=="CallMax"):
            return np.maximum(0, np.amax(spots,1)-self.strike)
        else:
            print("Invalid input for the payoff function")