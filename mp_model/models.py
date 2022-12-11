import numpy as np
from mp_model.metrics import NMSE
from mp_model.io import SystemData
from mp_model.model_parameters import ModelParameters


class ComplexMatrix:
    '''
    Complex matrix multiplication for extraction and validation.
    '''

    def __init__(self, parameters: ModelParameters, data: SystemData):

        polynomial_order = parameters.get_polynomial_order()
        memory_order = parameters.get_memory_order()

        in_extraction = data.get_in_extraction()
        out_extraction = data.get_out_extraction()
        in_validation = data.get_in_validation()
        out_validation = data.get_out_validation()

        self.extraction_x_matrix = self.__calculate_x_matrix(
            in_extraction, memory_order, polynomial_order)

        self.validation_x_matrix = self.__calculate_x_matrix(
            in_validation, memory_order, polynomial_order)

        self.coefficents = self.__calculate_coefficients(
            out_extraction, self.extraction_x_matrix, memory_order)

        self.output = self.__calculate_output(
            self.validation_x_matrix, self.coefficents, memory_order)

        self.nmse = NMSE(self.output, out_validation[memory_order+3:len(
            out_validation)-(memory_order+3), :]).get_nmse()

    def __calculate_x_matrix(self, mat_in, memory_order, polynomial_order):
        x_matrix = np.zeros(
            (len(mat_in), polynomial_order*(memory_order+1)), dtype=complex)
        for row in range(memory_order+1, len(mat_in)):
            for mem in range(memory_order+1):
                for pol in range(1, polynomial_order+1):
                    col = ((mem*polynomial_order)-1)+pol
                    x_matrix[row, col] = mat_in[row-mem, 0] * \
                        ((np.absolute(mat_in)[row-mem, 0])**(pol-1))
        return x_matrix

    def __calculate_coefficients(self, out_extraction, extraction_x_matrix, memory_order):
        extraction_x_matrix = extraction_x_matrix[memory_order+3:len(
            extraction_x_matrix)-(memory_order+3), :]
        out_extraction = out_extraction[memory_order +
                                        3:len(out_extraction)-(memory_order+3), :]
        coefficients = np.linalg.lstsq(
            extraction_x_matrix, out_extraction, rcond=-1)
        return coefficients[0]

    def __calculate_output(self, validation_x_matrix, coefficients, memory_order):
        validation_x_matrix = validation_x_matrix[memory_order+3:len(
            validation_x_matrix)-(memory_order+3), :]
        return validation_x_matrix@coefficients


class RealMatrix:
    def __init__(self):
        pass


class LUT:
    def __init__(self):
        pass
