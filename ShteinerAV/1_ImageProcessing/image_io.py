import cv2
import numpy as np
import os


def read_image(path_to_image: str) -> np.ndarray:
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            f"Не удалось прочитать изображение (возможно, файл поврежден или не существует): {path_to_image}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def write_image(path_to_image: str, image: np.ndarray) -> None:

    while os.path.exists(path_to_image) :
        os.remove(path_to_image)

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path_to_image, img_bgr)

    if not success:
        raise ValueError(f"Не удалось сохранить изображение по пути: {path_to_image}")
