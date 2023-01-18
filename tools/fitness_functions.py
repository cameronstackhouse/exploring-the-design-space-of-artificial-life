from enum import Enum

class FitnessFunction(Enum):
    """
    List of fitness functions to test a population for fitness
    """
    MAX_DISTANCE = 1
    VERTICAL_DISTANCE = 2
    INTERACTION = 3
    OBJECT_EXPULSION = 4
    TOTAL_DISTANCE = 5
    #TODO Add more fitness func values here