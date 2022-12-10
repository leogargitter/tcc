from numpy import linspace


class ModelParameters:
    '''Receives and stores model and LUT parameters.'''

    def __init__(self, memory_effect, polynomial_order, lut_order=0):

        self.M = memory_effect
        self.P = polynomial_order
        self.Q = 2**lut_order

        if lut_order > 0:
            self.in_lut = linspace(0, 1, self.Q)
