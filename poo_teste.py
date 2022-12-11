from mp_model import SystemData, ModelParameters, ComplexMatrix

data = SystemData("data\data_LDMOS.mat", 'in_extraction', 'out_extraction', 'in_validation', 'out_validation')
parameters = ModelParameters(5,2)

complex_matrix_model = ComplexMatrix(parameters, data)

print(complex_matrix_model.nmse)