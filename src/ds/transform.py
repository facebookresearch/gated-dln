from enum import Enum


class InputTransformationMode(Enum):
    ROTATE = "rotate"
    PERMUTE = "permute"


class IntermediateTransformationMode(Enum):
    PERMUTE = "permute"


class OutputTransformationMode(Enum):
    MAP_USING_CLASS_COMBINATION = "map_using_class_combination"
    MAP_USING_CLASS_PERMUTATION = "map_using_class_permutation"
