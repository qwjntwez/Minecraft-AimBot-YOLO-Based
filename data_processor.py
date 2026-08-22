import cv2
import numpy

def __letter_box_resize(img:numpy.ndarray, new_size:tuple[int, int]):
    native_h, native_w, _ = img.shape

    multiplier = min(new_size[0] / native_w, new_size[1] / native_h)

    new_w = int(native_w * multiplier)
    new_h = int(native_h * multiplier)

    temp_resize = cv2.resize(img, (new_w, new_h),  interpolation=cv2.INTER_LINEAR)

    padding_x = int((new_size[0] - new_w) / 2)
    padding_y = int((new_size[1] - new_h) / 2)

    final_image = cv2.copyMakeBorder(
        temp_resize,
        padding_y, padding_y,
        padding_x, padding_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114))

    return final_image

def preprocess_image(img, new_size:tuple[int, int]):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized_image = __letter_box_resize(rgb_img, new_size)

    normalized_image = resized_image.astype(numpy.float32) / 255.0
    normalized_image = normalized_image.transpose(2,0,1)
    normalized_image = normalized_image[None, ...]

    return numpy.array(normalized_image, dtype=numpy.float32)

def denormalize_bbox(coord:list[int], current_res:tuple[int, int], native_res:tuple[int, int]) -> list[int]:
    multiplier = min(current_res[0] / native_res[0], current_res[1] / native_res[1])

    padding_x = (current_res[0] - native_res[0] * multiplier) / 2
    padding_y = (current_res[1] - native_res[1] * multiplier) / 2

    new_x1 = int((coord[0] - padding_x) / multiplier)
    new_y1 = int((coord[1] - padding_y) / multiplier)
    new_x2 = int((coord[2] - padding_x) / multiplier)
    new_y2 = int((coord[3] - padding_y) / multiplier)

    return [new_x1,new_y1,new_x2,new_y2]