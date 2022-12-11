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
    def __init__(self, parameters: ModelParameters, data: SystemData):

        #polynomial_order = parameters.get_polynomial_order()
        memory_order = parameters.get_memory_order()
        in_lut = parameters.get_in_lut()
        q = parameters.Q

        in_extraction = data.get_in_extraction()
        out_extraction = data.get_out_extraction()
        in_validation = data.get_in_validation()
        out_validation = data.get_out_validation()


        #============
        x_nm  = self.__x_n_m(in_extraction,memory_order)[memory_order:len(in_extraction),:]
        abs_x_nm = abs(x_nm)
        x_lut = self.__xlut(in_lut, x_nm, abs_x_nm, memory_order, q)
        s_lut = self.__slut(x_lut, out_extraction, memory_order, q)
        val_nm = self.__x_n_m(in_validation, memory_order)
        val_nm = val_nm[memory_order:len(val_nm),:]

        self.out_lut = self.__interpolacao(in_lut,val_nm,s_lut,memory_order,q)
        out_val_lut = out_validation[memory_order:len(out_validation),:]
        self.nmse = NMSE(self.out_lut,out_val_lut).get_nmse()
        #============

        # DEFINIÇÂO DAS MATRIZES CONTENDO OS VALORES DE x(n-m) e |x(n-m)|
    def __x_n_m(self, in_extraction, memory_order):
        x_nm = np.zeros((len(in_extraction),(memory_order+1)),dtype=complex)
        for r in range(memory_order,len(in_extraction)):
            for m in range(memory_order+1):
                x_nm[r,m] = in_extraction[r-m]
        return x_nm


    def __xlut(self, e_lut, x_nm, abs_x_nm, M, Q):
        #Nesse primero loop é calculado o primero bloco de Q colunas
        x_lut = np.zeros((len(x_nm),Q),dtype = "complex_")

        for r in range(len(x_nm)):
            for c in range(1,Q,1):
                #Aqui se aplica a condição mencionada anteriormente
                if e_lut[c-1] < abs_x_nm[r,0] < e_lut[c]:
                    #As 2 próximas linhas são as formulas para os valores
                    x_lut[r,c] = x_nm[r,0] * ((abs_x_nm[r,0] - e_lut[c-1])/((e_lut[c] - e_lut[c-1])))
                    x_lut[r,c-1] = x_nm[r,0] * (1 - ((abs_x_nm[r,0] - e_lut[c-1])/((e_lut[c] - e_lut[c-1]))))

        #A partir daqui a função só continua se M>=1, os calculos se repetem para valores de M diferentes e os blocos calculados são juntados aos anteriores resultando na matriz X_LUT
        if M>=1:
            for m in range(1, M+1):
                x_lut_temp = np.zeros((len(x_nm),Q),dtype = "complex_")
                for r in range(len(x_nm)):
                    for c in range(1,Q,1):
                        if e_lut[c-1] < abs_x_nm[r,m] < e_lut[c]:
                            x_lut_temp[r,c] = x_nm[r,m] * ((abs_x_nm[r,m] - e_lut[c-1])/((e_lut[c] - e_lut[c-1])))
                            x_lut_temp[r,c-1] = x_nm[r,m] * (1 - ((abs_x_nm[r,m] - e_lut[c-1])/((e_lut[c] - e_lut[c-1]))))

                x_lut = np.concatenate((x_lut,x_lut_temp), axis=1)                 
        return x_lut
                
    def __slut(self, x_lut, out_ext, M, Q):
        # A propriedade de matrizes .H do Numpy aplica o operador transposto complexo conjugado. A propriedade .I representa a matriz inversa.

        x1 = np.asmatrix(x_lut).H
        #s_lut = asmatrix(x1@x_lut).I@asmatrix(x1@out_ext[M:len(out_ext),:])
        s_lut = np.linalg.lstsq(x_lut,out_ext[M:len(out_ext),:], rcond=-1)
        s_lut = s_lut[0]
        s_lut2 = s_lut[0:Q]

        if M>=1:
            for m in range(1,M+1,1):
                s_lut3 = s_lut[m*Q:(m+1)*Q]
                s_lut2 = np.concatenate((s_lut2, s_lut3), axis=1)
                
        return s_lut2

    def __interpolacao(self,e_lut, x_nm,s_lut,M,Q):
        #x_nm = x_nm[M:len(x_nm),:]
        inter = np.zeros((len(x_nm),M+1),dtype = "complex_")
        for c in range(M+1):
            for r in range(len(x_nm)):
                for q in range(1,Q,1):
                    if e_lut[q-1] < abs(x_nm[r,c]) < e_lut[q]:
                        inter[r,c] = (s_lut[q-1,c] + (((s_lut[q,c] - s_lut[q-1,c]) / (e_lut[q] - e_lut[q-1])) * (abs(x_nm[r,c]) - e_lut[q-1]))) * x_nm[r,c] 

        inter = inter.sum(axis=1).reshape((len(inter),1))
        return inter