from numpy import linspace


class ModelParameters:
    '''Receives and stores model and LUT parameters.'''

    def __init__(self, polynomial_order, memory_order, lut_order=0):

        self.polynomial_order = polynomial_order
        self.memory_order = memory_order
        self.Q = 2**lut_order

        if lut_order > 0:
            self.in_lut = linspace(0, 1, self.Q)


    def __str__(self):
        return f"Polynomial order (P) = {self.polynomial_order}\nMemory order (M) = {self.memory_order}\nLUT order (Q): 2^{self.Q}"

    def get_polynomial_order(self):
        return self.polynomial_order

    def get_memory_order(self):
        return self.memory_order

    def get_in_lut(self):
        return self.in_lut