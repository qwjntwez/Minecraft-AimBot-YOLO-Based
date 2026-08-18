import threading
import multiprocessing
import dxcam_cpp as dxcam
import cv2
import onnxruntime
import numpy

import aim_controller
from aim_controller import Point
from enemy import Enemy
from data_processor import preprocess_image, denormalize_coordinate
from aim_controller import AimController

def main() -> None:
    AIMING:bool = True
    SHOW_BBOX_SCREEN:bool = False

    screen_resolution = (1920, 1080)
    model_image_size = (640, 640)

    model_path = "model/model.onnx"

    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGR"
    )

    camera.start(region=(0, 0, 1920, 1080), target_fps=60)

    print(camera.is_capturing)

    controller = AimController(camera)
    manager = multiprocessing.Manager()
    enemies = manager.list()

    aiming_process = multiprocessing.Process(target=controller.update, args=(enemies,))
    aiming_process.start()

    while True:
        frame = camera.get_latest_frame()

        preprocessed_frame = preprocess_image(frame, model_image_size)

        outputs = session.run([label_name], {input_name:preprocessed_frame})

        predicts = outputs[0]
        predicts = numpy.squeeze(predicts, axis=0)

        for obj in predicts:
            x_left, y_top, x_right, y_bottom, conf, cls = obj.tolist()

            if conf < 0.55:
                continue

            #Кординати переводяться із нормалізації 640х640 у розміри екрану/зони захвату зображення TODO:Зробить вибір розширень
            x_left = denormalize_coordinate(x_left, screen_resolution[0], model_image_size[0])
            y_top = denormalize_coordinate(y_top, screen_resolution[1], model_image_size[1])
            x_right = denormalize_coordinate(x_right, screen_resolution[0], model_image_size[0])
            y_bottom = denormalize_coordinate(y_bottom, screen_resolution[1], model_image_size[1])

            left_top:Point = Point(x_left, y_top)
            right_bottom:Point = Point(x_right, y_bottom)

            enemies.append(Enemy(left_top, right_bottom, conf))

            if SHOW_BBOX_SCREEN:
                cv2.rectangle(
                    frame,
                    (x_left, y_top),
                    (x_right, y_bottom),
                    (255, 0, 0),
                    2
                )

            # print(f  "Center: ({x_center * scale_x}, {y_center * scale_y}), Size: {width * scale_x}x{height*scale_y}, Conf: {conf:.2f}, Class: {cls}")
        enemies.clear()
        
        if SHOW_BBOX_SCREEN:
            cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    aiming_process.join()
    camera.stop()

    print(camera.is_capturing)

if __name__ == '__main__':
    main()