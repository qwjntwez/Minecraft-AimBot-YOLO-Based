from point import Point

class Enemy:
    def __init__(self, object_id:int, left_top:Point, right_bottom:Point, confidence:float):
        self.object_id:int = object_id
        self.left_top:Point = left_top
        self.right_bottom:Point = right_bottom
        self.center:Point = Point((self.right_bottom.x + self.left_top.x) // 2, (self.left_top.y + self.right_bottom.y) //2)
        self.confidence:float = confidence
