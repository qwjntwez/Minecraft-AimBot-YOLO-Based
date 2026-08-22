import multiprocessing
import dxcam_cpp as dxcam
import cv2
import onnxruntime
import numpy
from onnxruntime.capi.onnxruntime_inference_collection import Session

from point import Point
from enemy import Enemy
from data_processor import preprocess_image, denormalize_bbox
from aim_controller import AimController

AIMING:bool = True
SHOW_BBOX_SCREEN: bool = False

model_image_size:tuple[int, int] = (640, 640)
screen_resolution:tuple[int, int] = (1920, 1080)

model_path:str = "model/temp.onnx"

def model_processing(frame_queue, enemies):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    while True:
        frame = frame_queue.get()

        if frame is None:
            continue

        preprocessed_frame = preprocess_image(frame, model_image_size)

        outputs = session.run([label_name], {input_name: preprocessed_frame})

        predicts = outputs[0]
        predicts = numpy.squeeze(predicts, axis=0)

        for obj in predicts:
            x_left, y_top, x_right, y_bottom, conf, cls = obj.tolist()

            if conf < 0.4:
                continue

            # Кординати переводяться із нормалізації 640х640 у розміри екрану/зони захвату зображення TODO:Зробить вибір розширень
            x_left, y_top, x_right, y_bottom = denormalize_bbox([x_left,y_top,x_right,y_bottom], model_image_size, screen_resolution)

            left_top: Point = Point(x_left, y_top)
            right_bottom: Point = Point(x_right, y_bottom)

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

def main() -> None:
    #Variables for processes
    controller = AimController(screen_resolution)
    manager = multiprocessing.Manager()
    enemies = manager.list()
    frame_queue = multiprocessing.Queue(maxsize=1)

    #Processes
    model_process = multiprocessing.Process(target=model_processing, args=(frame_queue, enemies, ))
    aiming_process = multiprocessing.Process(target=controller.update, args=(enemies,))

    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGR"
    )

    camera.start(region=(0, 0, 1920, 1080), target_fps=240)

    print(camera.is_capturing)

    model_process.start()
    aiming_process.start()

    while True:
        frame = camera.get_latest_frame()
        frame_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    aiming_process.join()
    model_process.join()
    camera.stop()

    print(camera.is_capturing)

if __name__ == '__main__':
    main()