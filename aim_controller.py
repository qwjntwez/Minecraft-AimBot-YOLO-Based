import math

class Point:
    def __init__(self, x:int | float, y:int | float):
        self.x:int | float = x
        self.y:int | float = y

    @staticmethod
    def distance(p1:Point, p2:Point) -> int | float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    #Calculate angle of linear function
    @staticmethod
    def get_angle(p1:Point, p2:Point) -> int | float:
        return (p2.y - p1.y) / (p2.x - p1.x)

class AimController:
    def __init__(self, monitor_resolution:tuple[int, int]):
        self.dx = 15
        self.monitor_center_x = monitor_resolution[0] / 2
        self.monitor_center_y = monitor_resolution[1] / 2

