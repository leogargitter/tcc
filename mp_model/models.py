import numpy as np


class ComplexMatrix:
    '''
    Complex matrix multiplication for extraction and validation.
    '''

    def __init__(self, memory_effect, polynomial_order, in_extraction, out_extraction, in_validation, out_validation):
        self.extraction_x_matrix = self.__calculate_x_matrix(
            in_extraction, memory_effect, polynomial_order)
        self.validation_x_matrix = self.__calculate_x_matrix(
            in_validation, memory_effect, polynomial_order)
        pass

    def __calculate_x_matrix(self, mat_in, memory_effect, polynomial_order):
        x_matrix = np.zeros(
            (len(mat_in), polynomial_order*(memory_effect+1)), dtype=complex)
        for row in range(memory_effect+1, len(mat_in)):
            for mem in range(memory_effect+1):
                for pol in range(1, polynomial_order+1):
                    col = ((mem*polynomial_order)-1)+pol
                    x_matrix[row, col] = mat_in[row-mem, 0] * \
                        ((np.absolute(mat_in)[row-mem, 0])**(pol-1))
        return x_matrix


""" 
    def __calculate_coefficients(self, x_matrix):
        X_ext = x_mp(in_ext, M, P)
        X_ext2 = X_ext[M+3:len(X_ext)-(M+3), :]
        out_ext2 = out_ext[M+3:len(out_ext)-(M+3), :]
        coefs = np.linalg.lstsq(X_ext2, out_ext2, rcond=-1)
        coefs = coefs[0]
        print(coefs)

    def __calculate_output(self):
        X_val = x_mp(in_val, M, P)
        X_val2 = X_val[M+3:len(X_val)-(M+3), :]
        out_calc_mat_cmplx = X_val2@coefs
        print(out_calc_mat_cmplx) """


class RealMatrix:
    def __init__(self):
        pass


class LUT:
    def __init__(self):
        pass
