from enum import Enum


class ModelEnum(Enum):
    """ Enum class with machine learning models. """
    SVC = 'svc'
    LOGISTIC_REGRESSION = 'logistic_regression'
    ELASTIC_NET = 'elastic_net'
    SVR = 'svr'
