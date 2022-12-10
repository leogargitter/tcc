from scipy.io import loadmat


class MatFile:
    '''
    Reads a .mat file with the extration and validation datasets already separated.
    '''

    def __init__(self, filename, in_extraction_name, out_extraction_name, in_validation_name, out_validation_name):

        data = loadmat(filename)
        self.in_extraction = data[in_extraction_name]
        self.out_extraction = data[out_extraction_name]
        self.in_validation = data[in_validation_name]
        self.out_validation = data[out_validation_name]
