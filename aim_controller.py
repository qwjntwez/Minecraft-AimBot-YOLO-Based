import mouse_input
from enemy import Enemy
from point import Point

class AimController:
    def __init__(self, monitor_resolution:tuple[int, int]):
        self.monitor_center = Point(monitor_resolution[0] // 2, monitor_resolution[1] // 2)

    def aim(self, target_point:Point) -> None:
        dx:int = self.monitor_center.x - target_point.x
        dy:int = self.monitor_center.y - target_point.y

        mouse_input.move_mouse(dx, dy)

    def update(self, enemy_list:list[Enemy])->None:
        while True:
            if not enemy_list:
                closest_enemy:Enemy = min(enemy_list, key=lambda enemy: enemy.center.distance(self.monitor_center))

                self.aim(closest_enemy.center)