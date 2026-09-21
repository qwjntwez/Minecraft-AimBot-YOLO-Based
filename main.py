import multiprocessing
import queue
import dxcam_cpp as dxcam
import onnxruntime
import numpy
import time
import threading

from boxmot.trackers.results import TrackResults
from udp_listener import UDPListener
from point import Point
from enemy import Enemy
from data_processor import preprocess_image, denormalize_bbox
from aim_controller import AimController
from boxmot.trackers.bbox import ocsort

AIMING:bool = True

model_image_size:tuple[int, int] = (640, 640)
screen_resolution:tuple[int, int] = (2560, 1440)

model_path:str = "model/model.onnx"

def model_inference(frame_input_queue, predict_queue):
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    while True:
        start_time = time.perf_counter()
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

        end_time = time.perf_counter()
        total = end_time - start_time
        print(f"Model predict delay: {total:.4f} s")

def model_processing(predict_queue, aim_controller:AimController):
    tracker = ocsort.OcSort()

    enemies = []

    while True:
        predict, frame = predict_queue.get()

        if predict is None or predict.size == 0:
            continue

        track_start_time = time.perf_counter()
        tracks:TrackResults = tracker.update(predict, frame)
        track_end_time = time.perf_counter()
        total_track_time = track_end_time - track_start_time

        print(f"Tracking delay: {total_track_time:.4f} s")

        if len(tracks) == 0:
            continue

        for_loop_time_start = time.perf_counter()
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

        for_loop_time_end = time.perf_counter()
        for_loop_time_total = for_loop_time_end - for_loop_time_start

        print(f"For loop every predict and process it delay: {for_loop_time_total:.4f} s")


        if AIMING:
            aim_time_start = time.perf_counter()
            aim_controller.update(enemies)

            del enemies[:]
            aim_time_end = time.perf_counter()

            total_aim_time = aim_time_end - aim_time_start
            print(f"Aim and clear enemies list: {total_aim_time:.4f} s")


def main() -> None:
    controller = AimController(screen_resolution)
    udp_listener = UDPListener("127.0.0.1", 5000)

    frame_queue = multiprocessing.Queue(maxsize=1)
    predict_queue = multiprocessing.Queue(maxsize=1)

    #Processes
    model_inference_process = multiprocessing.Process(
        target=model_inference, args=(frame_queue, predict_queue,)
    )
    model_inference_process.daemon = True
    model_inference_process.start()

    model_process = multiprocessing.Process(target=model_processing, args=(predict_queue, controller,))
    model_process.daemon = True
    model_process.start()

    #Threads
    attack_thread = threading.Thread(target=controller.attack)
    attack_thread.daemon = True
    attack_thread.start()

    udp_thread = threading.Thread(target=udp_listener.state_update, args=(controller,))
    udp_thread.daemon = True
    udp_thread.start()

    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGR"
    )

    camera.start(region=(0, 0, screen_resolution[0], screen_resolution[1]), target_fps=240)
    print(camera.is_capturing)

    while True:
        start_time = time.perf_counter()
        frame = camera.get_latest_frame()
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
        end_time = time.perf_counter()

        total_time = end_time - start_time

        print(f"Screen capture delay: {total_time:.4f} s")

if __name__ == '__main__':
    main()