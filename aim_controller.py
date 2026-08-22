import mouse_input
from enemy import Enemy
from point import Point

class AimController:
    def __init__(self, monitor_resolution:tuple[int, int]):
        self.monitor_center = Point(monitor_resolution[0] // 2, monitor_resolution[1] // 2)

    def aim(self, target_point:Point) -> None:

        SENSITIVITY = 0.5  # Подбирается экспериментально

        dx = int((target_point.x - self.monitor_center.x) * SENSITIVITY)
        dy = int((target_point.y - self.monitor_center.y) * SENSITIVITY)

        print(
            f"Center: {self.monitor_center.x}, {self.monitor_center.y}, Target: {target_point.x}, {target_point.y} -> Calc DX: {dx},"
            f" DY: {dy}"
        )
        mouse_input.move_mouse(dx, dy)

    def update(self, enemy_list:list[Enemy])->None:
        while True:
            if len(enemy_list) != 0:
                closest_enemy:Enemy | None = min(enemy_list, key=lambda enemy: enemy.center.distance(self.monitor_center, enemy.center), default=None)

                if closest_enemy is not None:
                    self.aim(closest_enemy.center)