from typing import NamedTuple

class Feature(NamedTuple):
    type: str

class straightLine(NamedTuple):
    hough: tuple
    x: list
    y: list
    feature: Feature