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
        validation_x_matrix = self.__calculate_x_matrix(
            data.in_validation, model_parameters.memory_order, model_parameters.polynomial_order)
        output = self.__calculate_output(
            validation_x_matrix, self.coefficents, model_parameters.memory_order)
        nmse = NMSE(output, data.out_validation[model_parameters.memory_order+3:len(
            data.out_validation)-(model_parameters.memory_order+3), :]).get_nmse()

        self.original_in_validation = data.in_validation[model_parameters.memory_order+3:len(data.out_validation)-(model_parameters.memory_order+3), :]
        self.original_out_validation = data.out_validation[model_parameters.memory_order+3:len(data.out_validation)-(model_parameters.memory_order+3), :]

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
        in_real = self.__in_abs_real(
            data.in_extraction, model_parameters.polynomial_order)
        x_mp_real = self.__x_mp_real(
            data.in_extraction, model_parameters.memory_order)
        mult1 = self.__mult1(
            in_real, x_mp_real, model_parameters.memory_order, model_parameters.polynomial_order)
        out_ext = data.out_extraction[0:len(mult1)]
        coefficients = np.linalg.lstsq(mult1, np.column_stack(
            (out_ext.real, out_ext.imag)), rcond=-1)
        self.coefficients = coefficients[0]

        return self.coefficients

    def evaluate_model(self, model_parameters: ModelParameters, data: SystemData):
        in_real = self.__in_abs_real(
            data.in_validation, model_parameters.polynomial_order)
        x_real = self.__x_mp_real(
            data.in_validation, model_parameters.memory_order)

        mult1_re = self.__mult1(
            in_real, x_real, model_parameters.memory_order, model_parameters.polynomial_order)
        mult2_re = mult1_re@self.coefficients
        output = mult2_re[:, 0]+(mult2_re[:, 1]*1j)
        output = output.reshape(len(output), 1)
        out_val_re = data.out_validation[0:len(
            data.out_validation)-model_parameters.memory_order, :]

        nmse = NMSE(output, out_val_re).get_nmse()

        self.original_in_validation = data.in_validation[0:len(data.out_validation)-model_parameters.memory_order, :]
        self.original_out_validation = out_val_re

        return output, nmse

    def __in_abs_real(self, in_val, P):
        res = np.zeros((len(in_val), P))
        for c in range(1, P+1, 1):
            for r in range(len(in_val)):
                res[r, c-1] = (np.abs(in_val[r]))**(c-1)
        return res

    def __x_mp_real(self, in_val, M):
        res0 = np.zeros((len(in_val), M+1), dtype=complex)
        for c in range(M+1):
            for r in range(len(in_val)-c):
                res0[r+c, c] = in_val[r, 0]

        res = np.zeros((len(in_val), 2*(M+1)))
        c2 = 0
        for c in range(0, 2*(M+1), 2):
            res[:, c] = np.real(res0[:, c2])
            res[:, c+1] = np.imag(res0[:, c2])
            c2 += 1
        return res

    def __mult1(self, in_real, x_real, M, P):
        res = np.zeros((len(in_real), 2*P*(M+1)))
        zero = np.zeros((1, in_real.shape[1]))
        c = -2
        for c1 in range(0, 2*(M+1), 2):
            if c1 != 0:
                in_real = np.append(zero, in_real, axis=0)
                in_real = in_real[0:in_real.shape[0]-1, :]
            for c2 in range(P):
                c += 2
                res[:, c] = x_real[:, c1]*in_real[:, c2]
                res[:, c+1] = x_real[:, c1+1]*in_real[:, c2]
        res = res[0:len(res)-M, :]
        return res


class LookUpTable(CalculationStrategy):
    def calculate_coefficients(self, model_parameters: ModelParameters, data: SystemData):
        in_lut = model_parameters.get_in_lut()
        x_nm = self.__x_n_m(data.in_extraction, model_parameters.memory_order)[
            model_parameters.memory_order:len(data.in_extraction), :]
        abs_x_nm = abs(x_nm)
        self.x_lut = self.__xlut(
            in_lut, x_nm, abs_x_nm, model_parameters.memory_order, model_parameters.Q)
        self.s_lut = self.__slut(
            self.x_lut, data.out_extraction, model_parameters.memory_order, model_parameters.Q)

        self.coefficients = self.s_lut

        return self.coefficients

    def evaluate_model(self, model_parameters: ModelParameters, data: SystemData):
        in_lut = model_parameters.get_in_lut()
        val_nm = self.__x_n_m(data.in_validation,
                              model_parameters.memory_order)
        val_nm = val_nm[model_parameters.memory_order:len(val_nm), :]
        output = self.__interpolacao(
            in_lut, val_nm, self.s_lut, model_parameters.memory_order, model_parameters.Q)
        out_val_lut = data.out_validation[model_parameters.memory_order:len(
            data.out_validation), :]
        nmse = NMSE(output, out_val_lut).get_nmse()
        
        self.original_in_validation = data.in_validation[model_parameters.memory_order:len(data.out_validation), :]
        self.original_out_validation = out_val_lut
        

        return output, nmse

    def __x_n_m(self, in_extraction, memory_order):
        x_nm = np.zeros((len(in_extraction), (memory_order+1)), dtype=complex)
        for r in range(memory_order, len(in_extraction)):
            for m in range(memory_order+1):
                x_nm[r, m] = in_extraction[r-m]
        return x_nm

    def __xlut(self, e_lut, x_nm, abs_x_nm, M, Q):
        # Nesse primero loop é calculado o primero bloco de Q colunas
        x_lut = np.zeros((len(x_nm), Q), dtype="complex_")

        for r in range(len(x_nm)):
            for c in range(1, Q, 1):
                # Aqui se aplica a condição mencionada anteriormente
                if e_lut[c-1] < abs_x_nm[r, 0] < e_lut[c]:
                    # As 2 próximas linhas são as formulas para os valores
                    x_lut[r, c] = x_nm[r, 0] * \
                        ((abs_x_nm[r, 0] - e_lut[c-1]) /
                         ((e_lut[c] - e_lut[c-1])))
                    x_lut[r, c-1] = x_nm[r, 0] * \
                        (1 - ((abs_x_nm[r, 0] - e_lut[c-1]) /
                         ((e_lut[c] - e_lut[c-1]))))

        # A partir daqui a função só continua se M>=1, os calculos se repetem para valores de M diferentes e os blocos calculados são juntados aos anteriores resultando na matriz X_LUT
        if M >= 1:
            for m in range(1, M+1):
                x_lut_temp = np.zeros((len(x_nm), Q), dtype="complex_")
                for r in range(len(x_nm)):
                    for c in range(1, Q, 1):
                        if e_lut[c-1] < abs_x_nm[r, m] < e_lut[c]:
                            x_lut_temp[r, c] = x_nm[r, m] * \
                                ((abs_x_nm[r, m] - e_lut[c-1]) /
                                 ((e_lut[c] - e_lut[c-1])))
                            x_lut_temp[r, c-1] = x_nm[r, m] * \
                                (1 - ((abs_x_nm[r, m] - e_lut[c-1]
                                       )/((e_lut[c] - e_lut[c-1]))))

                x_lut = np.concatenate((x_lut, x_lut_temp), axis=1)
        return x_lut

    def __slut(self, x_lut, out_ext, M, Q):
        # A propriedade de matrizes .H do Numpy aplica o operador transposto complexo conjugado. A propriedade .I representa a matriz inversa.

        x1 = np.asmatrix(x_lut).H
        #s_lut = asmatrix(x1@x_lut).I@asmatrix(x1@out_ext[M:len(out_ext),:])
        s_lut = np.linalg.lstsq(x_lut, out_ext[M:len(out_ext), :], rcond=-1)
        s_lut = s_lut[0]
        s_lut2 = s_lut[0:Q]

        if M >= 1:
            for m in range(1, M+1, 1):
                s_lut3 = s_lut[m*Q:(m+1)*Q]
                s_lut2 = np.concatenate((s_lut2, s_lut3), axis=1)

        return s_lut2

    def __interpolacao(self, e_lut, x_nm, s_lut, M, Q):
        #x_nm = x_nm[M:len(x_nm),:]
        inter = np.zeros((len(x_nm), M+1), dtype="complex_")
        for c in range(M+1):
            for r in range(len(x_nm)):
                for q in range(1, Q, 1):
                    if e_lut[q-1] < abs(x_nm[r, c]) < e_lut[q]:
                        inter[r, c] = (s_lut[q-1, c] + (((s_lut[q, c] - s_lut[q-1, c]) / (
                            e_lut[q] - e_lut[q-1])) * (abs(x_nm[r, c]) - e_lut[q-1]))) * x_nm[r, c]

        inter = inter.sum(axis=1).reshape((len(inter), 1))
        return inter


class MemoryPolynomial:

    def __init__(self, model_parameters: ModelParameters, data: SystemData,  calculation_strategy: CalculationStrategy, evaluate=True) -> None:
        self.data = data
        self.model_paramaters = model_parameters
        self.coefficients = calculation_strategy.calculate_coefficients(
            model_parameters, data)
        self.calculation_strategy = calculation_strategy

        if evaluate:
            self.calculated_output, self.nmse = calculation_strategy.evaluate_model(
                model_parameters, data)
            self.real_input = self.calculation_strategy.original_in_validation
            self.real_output = self.calculation_strategy.original_out_validation

    def evaluate(self):
        self.calculated_output, self.nmse = self.calculation_strategy.evaluate_model(
            self.model_parameters, self.data)
        self.real_output = self.calculation_strategy.original_out_validation
