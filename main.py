import dxcam_cpp as dxcam
import cv2
import onnxruntime
import numpy
import pyautogui

def __letter_box_resize(img:numpy.ndarray, new_size:tuple[int, int]) -> numpy.ndarray:
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

    return final_image

def preprocess_image(img, new_size:tuple[int, int]):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized_image = __letter_box_resize(rgb_img, new_size)

    normalized_image = resized_image.astype(numpy.float32) / 255.0
    normalized_image = normalized_image.transpose(2,0,1)
    normalized_image = normalized_image[None, ...]

    return normalized_image

def main() -> None:
    model_path = "model/model.onnx"

    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

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

        preprocessed_frame = preprocess_image(frame, model_image_size)

        outputs = session.run([label_name], {input_name:preprocessed_frame})

        predicts = outputs[0]
        predicts = numpy.squeeze(predicts, axis=0)

        scale_x = 1920 / model_image_size[0]
        scale_y = 1080 / model_image_size[1]

        for obj in predicts:
            x_center, y_center, width, height, conf, cls = obj.tolist()

            # x_left = int(x_center - (width / 2))
            # x_right = int(x_center + (width / 2))
            #
            # y_top = int(y_center + (width / 2))
            # y_bottom = int(y_center - (width / 2))

            if conf < 0.55:
                continue

            # cv2.rectangle(
            #     frame,
            #     (x_left, y_top),
            #     (x_right, y_bottom),
            #     (255, 0, 0),
            #     2
            # )

            print(f"Center: ({x_center * scale_x}, {y_center * scale_y}), Size: {width * scale_x}x{height*scale_y}, Conf: {conf:.2f}, Class: {cls}")

        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()

    print(camera.is_capturing)

if __name__ == '__main__':
    main()