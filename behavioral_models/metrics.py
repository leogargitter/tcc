import numpy as np

class NMSE:
    def __init__(self, calculated_output, validation_output):
        self.result = self.__calculate_nmse(calculated_output, validation_output)

    def __calculate_nmse(self, calculated, validation):
        error = validation - calculated
        error_sum = np.sum(np.absolute(error)**2)
        validation_sum = np.sum(np.absolute(validation)**2)
        return 10*np.log10(error_sum/validation_sum)

    def get_nmse(self):
        return self.result