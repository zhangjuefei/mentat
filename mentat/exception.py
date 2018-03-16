class UnSupportException(Exception):
    def __init__(self, method):
        super().__init__("Method / Problem UnSupported: {:s}".format(method))


class DataException(Exception):
    def __init__(self, msg):
        super().__init__("Data Problem: {:s}".format(msg))


class ParameterException(Exception):
    def __init__(self, msg):
        super().__init__("Parameter Problem: {:s}".format(msg))


class ModelException(Exception):
    def __init__(self, msg):
        super().__init__(msg)
