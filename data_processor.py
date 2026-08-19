import cv2
import numpy

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

    return numpy.array(normalized_image, dtype=numpy.float32), padding, multiplier

def denormalize_coordinate(coord:int, padding:int, multiplier:float) -> int:
    return int((coord-padding) / multiplier)