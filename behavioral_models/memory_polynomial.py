import numpy as np
from behavioral_models.metrics import NMSE
from behavioral_models.io import SystemData
from behavioral_models.model_parameters import ModelParameters
from abc import ABC, abstractmethod


class CalculationStrategy(ABC):
    @abstractmethod
    def calculate_coefficients(self, model_parameters: ModelParameters, data: SystemData):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass


class ComplexMatrix(CalculationStrategy):
    def calculate_coefficients(self, model_parameters: ModelParameters, data: SystemData):
        pass

        extraction_x_matrix = self.__calculate_x_matrix(
            data.in_extraction, model_parameters.memory_order, model_parameters.polynomial_order)

        self.coefficents = self.__calculate_coefficients(
            data.out_extraction, extraction_x_matrix, model_parameters.memory_order)

        return self.coefficents

    def evaluate_model(self, model_parameters: ModelParameters, data: SystemData):
        validation_x_matrix = self.__calculate_x_matrix(data.in_validation, model_parameters.memory_order, model_parameters.polynomial_order)
        output = self.__calculate_output(validation_x_matrix, self.coefficents, model_parameters.memory_order)
        nmse = NMSE(output, data.out_validation[model_parameters.memory_order+3:len(data.out_validation)-(model_parameters.memory_order+3), :]).get_nmse()

        return output, nmse

    def __calculate_x_matrix(self, mat_in, memory_order, polynomial_order):
        abs_mat_in = np.absolute(mat_in)
        x_matrix = np.zeros(
            (len(mat_in), polynomial_order*(memory_order+1)), dtype=complex)

        for row in range(memory_order+1, len(mat_in)):
            for mem in range(memory_order+1):
                for pol in range(1, polynomial_order+1):
                    col = ((mem*polynomial_order)-1)+pol
                    x_matrix[row, col] = mat_in[row-mem, 0] * \
                        ((abs_mat_in[row-mem, 0])**(pol-1))
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


class RealMatrix(CalculationStrategy):
    def calculate_coefficients(self, model_parameters: ModelParameters, data: SystemData):
        pass


class LookUpTable(CalculationStrategy):
    def calculate_coefficients(self, model_parameters: ModelParameters, data: SystemData):
        pass


class MemoryPolynomial:

    def __init__(self, model_parameters: ModelParameters, data: SystemData,  calculation_strategy: CalculationStrategy, evaluate = True) -> None:
        self.data = data
        self.model_paramaters = model_parameters
        self.coefficients = calculation_strategy.calculate_coefficients(
            model_parameters, data)
        self.calculation_strategy = calculation_strategy

        if evaluate:
            self.calculated_output, self.nmse = calculation_strategy.evaluate_model(model_parameters, data)

    def evaluate(self):
        self.calculated_output, self.nmse = self.calculation_strategy.evaluate_model(self.model_parameters, self.data)
