from ..exception import ParameterException


class ParamValidator:
    def __init__(self, spec):
        """
        :param spec:
        example
        {
            "param_name": {
                "type": str,
                "allow_none": False
            }
        }
        """
        self.spec = spec

    def check(self, name, value):
        """
        :param name:
        :param value:
        """

        if not self.spec.get(name):
            raise ParameterException("no such parameter {:s}".format(name))

        spec = self.spec.get(name)

        if value is None:
            if not ("allow_none" in spec.keys() and spec.get("allow_none")):
                raise ParameterException("parameter {:s} must not be None".format(name))

        if spec.get("type"):

            allow_type = spec.get("type")

            if isinstance(allow_type, list):

                validate = False
                for tpe in allow_type:
                    if isinstance(value, tpe):
                        validate = True

                if not validate:
                    raise ParameterException(
                        "parameter \"{:s}\" must be of type: {:s}".format(name, " or ".join([str(t) for t in allow_type])))
            elif not isinstance(value, allow_type):
                raise ParameterException("parameter \"{:s}\" must be of type: {:s}".format(name, str(allow_type)))

        if spec.get("range"):
            min_value, max_value = spec.get("range")

            if value < min_value or value > max_value:
                raise ParameterException("parameter \"{:s}\" must be in [{:f}, {:f}]".format(name, min_value, max_value))

        return value

    def __call__(self, name, value):
        return self.check(name, value)
