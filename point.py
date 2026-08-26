import math


class Point:
    def __init__(self, x:int, y:int):
        self.x:int = x
        self.y:int = y

    @staticmethod
    def distance(p1:"Point", p2:"Point") -> int | float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    @staticmethod
    def get_angle(p1:"Point", p2:"Point") -> float:
        if p2.x - p1.x != 0:
            return (p2.y - p1.y) / (p2.x - p1.x)
        else:
            return 0