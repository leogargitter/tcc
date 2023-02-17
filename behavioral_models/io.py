from scipy.io import loadmat


class SystemData:
    '''
    Reads a .mat file with the extration and validation datasets already separated.
    Can be extended for other file formats.
    '''

    def __init__(self, filename, in_extraction_name, out_extraction_name, in_validation_name, out_validation_name):

        data = loadmat(filename)
        self.in_extraction = data[in_extraction_name]
        self.out_extraction = data[out_extraction_name]
        self.in_validation = data[in_validation_name]
        self.out_validation = data[out_validation_name]

    def get_in_extraction(self):
        return self.in_extraction

    def get_out_extraction(self):
        return self.out_extraction

    def get_in_validation(self):
        return self.in_validation

    def get_out_validation(self):
        return self.out_validation
