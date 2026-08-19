import dxcam_cpp as dxcam
import cv2
import onnxruntime
import numpy
import pyautogui

from aim_controller import Point
from enemy import Enemy

def __letter_box_resize(img:numpy.ndarray, new_size:tuple[int, int]):
    native_w ,native_h, _ = img.shape

    multiplier = min(new_size[0] / native_w, new_size[1] / native_h)

    new_w = int(native_w * multiplier)
    new_h = int(native_h * multiplier)

    temp_resize = cv2.resize(img, (new_w, new_h),  interpolation=cv2.INTER_LINEAR)

    padding_size = max(new_size[0] - new_w, new_size[1] - new_h) // 2

    if new_w == 0 and new_h > 0:
        top, bottom, left, right = padding_size, padding_size, 0, 0
    else:
        top, bottom, left, right = 0, 0, padding_size, padding_size

    final_image = cv2.copyMakeBorder(
        temp_resize,
        top, bottom,
        left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114))

    return final_image, padding_size, multiplier

def preprocess_image(img, new_size:tuple[int, int]):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized_image, padding, multiplier = __letter_box_resize(rgb_img, new_size)

    normalized_image = resized_image.astype(numpy.float32) / 255.0
    normalized_image = normalized_image.transpose(2,0,1)
    normalized_image = normalized_image[None, ...]

    return normalized_image, padding, multiplier

def denormalize_coordinate(coord:int, padding:int, multiplier:float):
    return int((coord - padding) / multiplier)

def main() -> None:
    model_path = "model/model.onnx"

    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    screen_resolution = (1920, 1080)
    model_image_size = (640, 640)

    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGR"
    )

    camera.start(region=(0, 0, 1920, 1080), target_fps=60)

    print(camera.is_capturing)

    while True:
        frame = camera.get_latest_frame()

        preprocessed_frame, padding, multiplier = preprocess_image(frame, model_image_size)

        outputs = session.run([label_name], {input_name:preprocessed_frame})

        predicts = outputs[0]
        predicts = numpy.squeeze(predicts, axis=0)

        enemies:list[Enemy] = []

        for obj in predicts:
            x_left, y_top, x_right, y_bottom, conf, cls = obj.tolist()

            if conf < 0.1:
                continue

            #Кординати переводяться із нормалізації 640х640 у розміри екрану/зони захвату зображення TODO:Зробить вибір розширень
            x_left = denormalize_coordinate(x_left, padding, multiplier)
            y_top = denormalize_coordinate(y_top, padding, multiplier)
            x_right = denormalize_coordinate(x_right, padding, multiplier)
            y_bottom = denormalize_coordinate(y_bottom, padding, multiplier)

            left_top:Point = Point(x_left, y_top)
            right_bottom:Point = Point(x_right, y_bottom)

            enemies.append(Enemy(left_top, right_bottom, conf))

            cv2.rectangle(
                frame,
                (x_left, y_top),
                (x_right, y_bottom),
                (255, 0, 0),
                2
            )

            print(
                f"Box: ({x_left}, {y_top}, {x_right}, {y_bottom}), Conf: {conf:.2f}, Class:"
                f" {cls}"
            )
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()

    print(camera.is_capturing)

if __name__ == '__main__':
    main()