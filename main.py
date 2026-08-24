import multiprocessing
import queue
import dxcam_cpp as dxcam
import onnxruntime
import numpy

from boxmot.trackers.results import TrackResults
from point import Point
from enemy import Enemy
from data_processor import preprocess_image, denormalize_bbox
from aim_controller import AimController
from boxmot.trackers.bbox import ocsort

AIMING:bool = True

model_image_size:tuple[int, int] = (640, 640)
screen_resolution:tuple[int, int] = (1920, 1080)

model_path:str = "model/model.onnx"

def model_inference(frame_input_queue, predict_queue):
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    while True:
        frame = frame_input_queue.get()

        if frame is None:
            continue

        preprocessed_frame = preprocess_image(frame, model_image_size)

        outputs = session.run([label_name], {input_name: preprocessed_frame})

        predicts = outputs[0]
        predicts = numpy.squeeze(predicts, axis=0)

        try:
            predict_queue.put_nowait((predicts, frame))
        except queue.Full:
            try:
                predict_queue.get_nowait()
                predict_queue.put_nowait((predicts, frame))
            except queue.Empty:
                pass

def model_processing(predict_queue, aim_controller:AimController):
    tracker = ocsort.OcSort()

    enemies = []

    while True:
        predict, frame = predict_queue.get()

        if predict is None or predict.size == 0:
            continue

        tracks:TrackResults = tracker.update(predict, frame)

        if len(tracks) == 0:
            continue

        for obj in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = obj

            if conf < 0.5:
                continue

            # Кординати переводяться із нормалізації 640х640 у розміри екрану/зони захвату зображення
            x_left, y_top, x_right, y_bottom = denormalize_bbox([x1,y1,x2,y2], model_image_size, screen_resolution)

            left_top: Point = Point(x_left, y_top)
            right_bottom: Point = Point(x_right, y_bottom)

            if AIMING:
                enemies.append(Enemy(track_id, left_top, right_bottom, conf))

        if AIMING:
            aim_controller.update(enemies)
            del enemies[:]

def main() -> None:
    #Variables for processes
    controller = AimController(screen_resolution)

    frame_queue = multiprocessing.Queue(maxsize=1)
    predict_queue = multiprocessing.Queue(maxsize=1)
    out_frame_queue = multiprocessing.Queue(maxsize=1)

    #Processes
    model_inference_process = multiprocessing.Process(
        target=model_inference, args=(frame_queue, predict_queue,)
    )
    model_inference_process.start()

    model_process = multiprocessing.Process(target=model_processing, args=(predict_queue,out_frame_queue, controller,))
    model_process.start()

    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGR"
    )

    camera.start(region=(0, 0, screen_resolution[0], screen_resolution[1]), target_fps=240)
    print(camera.is_capturing)

    while True:
        frame = camera.get_latest_frame()
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass


if __name__ == '__main__':
    main()