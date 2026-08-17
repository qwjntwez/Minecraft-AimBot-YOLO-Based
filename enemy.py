from aim_controller import Point

class Enemy:
    def __init__(self, left_top:Point, right_bottom:Point, confidence:float):
        self.left_top:Point = left_top
        self.right_bottom:Point = right_bottom
        self.center:Point = Point(self.right_bottom.x - self.left_top.x, self.left_top.y - self.right_bottom.y)
        self.confidence:float = confidence
