import mouse_input
from enemy import Enemy
from point import Point

class AimController:
    def __init__(self, monitor_resolution:tuple[int, int]):
        self.monitor_center = Point(monitor_resolution[0] // 2, monitor_resolution[1] // 2)

    def aim(self, target_point:Point) -> None:
        target_local_point = Point(target_point.x - self.monitor_center.x, target_point.y - self.monitor_center.y)

        SENSITIVITY = 0.1

        if Point.distance(Point(0,0), target_local_point) > 1:
            dx = int(round(target_local_point.x * SENSITIVITY))
            dy = int(round(target_local_point.y * SENSITIVITY))

            mouse_input.move_mouse(dx, dy)

    def update(self, enemy_list:list[Enemy])->None:
        id_to_track:int | None = None

        if len(enemy_list) > 0:
                # if id_to_track is None:
                #     closest_enemy:Enemy | None = min(enemy_list, key=lambda enemy: enemy.center.distance(self.monitor_center, enemy.center), default=None)
                #
                #     if closest_enemy is not None:
                #         id_to_track = closest_enemy.object_id
                # else:
                #     closest_enemy: Enemy | None = next((enemy for enemy in enemy_list if enemy.object_id == id_to_track), None)

            closest_enemy: Enemy | None = min(enemy_list,
                                                key=lambda enemy: enemy.center.distance(self.monitor_center,
                                                                                        enemy.center), default=None)

            if closest_enemy is not None:
                self.aim(closest_enemy.center)