import numpy as np
from abc import ABC, abstractmethod
import cv2

class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> 'ImageFilter':
        filters = {
            "resize": Resize,
            "grayscale": RGB2Grayscale,
            "antique": Antique,
            "fadecolor": FadeColor,
            "tape": Tape,
            "matte": Matte,
            "noize": Noize,
            "neon": Neon
        }

        filter_name = filter_type.lower()
        if filter_name not in filters:
            raise ValueError(f"Неизвестный тип фильтра: '{filter_type}'. "
                             f"Доступные: {list(filters.keys())}")

        return filters[filter_name](**kwargs)


class Resize(ImageFilter):
    def __init__(self, width:int , height:int):
        self.target_width = width
        self.target_height = height

    def apply_filter(self, image: np.ndarray) -> np.array:
        old_height, old_width = image.shape[:2]
        scale_x = old_width / self.target_width
        scale_y = old_height / self.target_height

        old_x = (np.arange(self.target_width) * scale_x).astype(int)
        old_y = (np.arange(self.target_height) * scale_y).astype(int)

        new_image = image[np.ix_(old_y, old_x)]

        return new_image

class RGB2Grayscale(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image.copy()

        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2126*r + 0.7152*g + 0.0722*b
        return gray.astype(np.uint8)

class Antique(ImageFilter):
    def __init__(self, intensivity: float=0.7, scale: float=10):
        self.intensivity = max(0.0, min(1.0, intensivity))
        self.scale = max(10, min(1000, scale))

    def add_film_grain(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        grain = np.random.poisson(img_float * self.scale) / self.scale

        if grain.ndim == 3:
            grain = grain.mean(axis=2)

        noisy_img = img_float - grain[..., np.newaxis] * 0.3
        return np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image_rgb = np.stack((image,)*3, axis=-1)
        else:
            image_rgb = image

        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ], dtype=np.float32)

        antique_image = np.dot(image_rgb.astype(np.float32), sepia_matrix.T)
        if self.intensivity < 1.0:
            result = image_rgb.astype(np.float32) * (1.0 - self.intensivity) + antique_image * self.intensivity
        else:
            result = antique_image

        result = np.clip(result, 0, 255).astype(np.uint8)
        result = self.add_film_grain(result)

        return result


class FadeColor(ImageFilter):
    def __init__(self, max_fade: float = 50.0):
        self.max_fade = max_fade

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32)
        f_min = self.max_fade
        f_max = 255.0 - self.max_fade

        scale = (f_max - f_min) / 255.0
        result = f_min + result * scale

        return np.clip(result, 0, 255).astype(np.uint8)

class Tape(ImageFilter):
    def __init__(self, mode: str = 'infrared', grain_amount: float = 0.3):
        self.mode = mode.lower()
        self.grain_amount = max(0.0, min(1.0, float(grain_amount)))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image_rgb = np.stack((image,)*3, axis=-1).astype(np.float32)
        else:
            image_rgb = image

        if self.mode == 'infrared':
            color_matrix = np.array([
                [0.1,  1.5, -0.2],
                [0.6,  0.2,  0.1],
                [-0.1, 0.1,  0.8]
            ], dtype=np.float32)
        elif self.mode == 'classic':
            color_matrix = np.array([
                [0.9,  0.1,  0.1],
                [0.0,  0.85, 0.15],
                [0.1,  0.1,  0.7]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Неизвестный параметр фильтра: '{self.mode}'. Доступны 'classic' и 'infraded'.")

        film_colored = np.dot(image_rgb, color_matrix.T)

        if self.grain_amount > 0.0:
            h, w, c = film_colored.shape
            std_dev = self.grain_amount * 25.0
            grain = np.random.normal(loc=0.0, scale=std_dev, size=(h, w, 1))

            luminance = (0.299 * film_colored[..., 0] + 0.587 * film_colored[..., 1] + 0.114 * film_colored[..., 2]) / 255.0
            grain_mask = 1.0 - 4.0 * (luminance - 0.5) ** 2
            grain_mask = np.clip(grain_mask[..., None], 0.1, 1.0)
            film_colored += grain * grain_mask

        result = film_colored
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

class Matte(ImageFilter):
    def __init__(self, scale: float = 0.9, feather: float = 0.15, fill_color: tuple = (255, 255, 255), density: float = 0.0):
        self.scale = max(0.1, min(1.0, float(scale)))
        self.feather = max(0.001, min(1.0, float(feather)))
        self.fill_color = np.array(fill_color, dtype=np.float32)
        self.density = max(0.001, min(10, density))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        center_y, center_x = h / 2.0, w / 2.0
        radius_x = center_x * self.scale
        radius_y = center_y * self.scale
        y_indices = np.arange(h, dtype=np.float32)
        x_indices = np.arange(w, dtype=np.float32)
        delta_x = (x_indices[None, :] - center_x) / radius_x
        delta_y = (y_indices[:, None] - center_y) / radius_y
        R = np.sqrt(delta_x ** 2 + delta_y ** 2)

        low_bound = 1.0 - self.feather
        high_bound = 1.0 + self.feather
        mask = (R - low_bound) / (high_bound - low_bound)
        mask = np.clip(mask, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = mask[..., None]

        blend = 4.0 * mask * (1.0 - mask)
        mask = np.clip(mask + blend * self.density, 0.0, 1.0)

        if len(image.shape) == 2:
            image_float = image.astype(np.float32)[..., None]
            fill_bg = np.array([self.fill_color.mean()], dtype=np.float32)
        else:
            image_float = image[..., :3].astype(np.float32)
            fill_bg = self.fill_color


        result_rgb = image_float * (1.0 - mask) + self.fill_color * mask
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        return result_rgb.squeeze() if len(image.shape) == 2 else result_rgb


class Noize(ImageFilter):
    def __init__(self, noise_amount: float = 0.004, num_scratches: int = 12, scratch_intensity: float = 0.7):
        self.noise_amount = max(0.0, min(0.1, float(noise_amount)))
        self.num_scratches = max(0, int(num_scratches))
        self.scratch_intensity = max(0.0, min(1.0, float(scratch_intensity)))

    def _draw_line(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int = 1):
        h, w = mask.shape[:2]
        num_points = int(np.hypot(x1 - x0, y1 - y0)) + 1

        if num_points <= 1:
            return

        t = np.linspace(0, 1, num_points)
        x_pts = x0 + t * (x1 - x0) + np.random.normal(0, 0.3, num_points)
        y_pts = y0 + t * (y1 - y0) + np.random.normal(0, 0.3, num_points)

        line_alpha = np.random.uniform(0.4, 1.0, size=num_points)

        radius = max(0.8, thickness / 1.5)

        for i in range(num_points):
            cx, cy = x_pts[i], y_pts[i]

            r_int = int(np.ceil(radius)) + 1
            x_min, x_max = max(0, int(cx - r_int)), min(w, int(cx + r_int + 1))
            y_min, y_max = max(0, int(cy - r_int)), min(h, int(cy + r_int + 1))

            if x_min >= x_max or y_min >= y_max:
                continue

            grid_y, grid_x = np.ogrid[y_min:y_max, x_min:x_max]
            dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

            falloff = np.clip(1.0 - (dist / (radius + 0.5)), 0.0, 1.0)

            mask[y_min:y_max, x_min:x_max] = np.maximum(
                mask[y_min:y_max, x_min:x_max],
                falloff * line_alpha[i]
            )

    def _generate_scratches_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.num_scratches):
            if np.random.rand() < 0.8:
                x0 = np.random.randint(0, w)
                y0 = np.random.randint(0, int(h * 0.3))
                x1 = x0 + np.random.randint(-15, 15)
                y1 = y0 + np.random.randint(int(h * 0.4), h)
            else:
                x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
                x1 = x0 + np.random.randint(-40, 40)
                y1 = y0 + np.random.randint(-40, 40)

            thickness = 1 if np.random.rand() > 0.2 else 2
            self._draw_line(mask, x0, y0, x1, y1, thickness=thickness)

        return mask

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        white_mask = np.zeros((h, w), dtype=np.float32)
        black_mask = np.zeros((h, w), dtype=np.float32)

        if self.noise_amount > 0:
            rand_matrix = np.random.rand(h, w)
            white_mask[rand_matrix < (self.noise_amount * 0.6)] = 1.0
            black_mask[(rand_matrix >= (self.noise_amount * 0.6)) & (rand_matrix < self.noise_amount)] = 1.0

        if self.num_scratches > 0 and self.scratch_intensity > 0:
            scratches = self._generate_scratches_mask(h, w)
            # np.maximum нужен, чтобы царапины не пересвечивали пыль, если они пересеклись
            white_mask = np.maximum(white_mask, scratches)

        def blur(mask_to_blur):
            if mask_to_blur.max() == 0:
                return mask_to_blur
            kernel = np.array([
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0]
            ], dtype=np.float32) / 6.0
            padded = np.pad(mask_to_blur, ((1, 1), (1, 1)), mode='edge')
            windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
            return np.sum(windows * kernel, axis=(-2, -1))

        white_mask = blur(white_mask)
        black_mask = blur(black_mask)

        w_alpha = np.clip(white_mask * self.scratch_intensity, 0.0, 1.0)[..., None]
        b_alpha = np.clip(black_mask, 0.0, 1.0)[..., None]

        result[..., :3] = result[..., :3] * (1.0 - w_alpha) + 255.0 * w_alpha
        result[..., :3] = result[..., :3] * (1.0 - b_alpha) + 0.0 * b_alpha

        return np.clip(result, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    def __init__(self, thickness:float = 2.0, intention: float = 0.3, glow_radius: float = 2, edges: float = 1.0):
        self.thickness = max(1.0, min(5.0, thickness))
        self.intention = max(0.0, min(1.0, intention))
        self.glow_radius = max(0.0, min(8.0, glow_radius))
        self.edges = max(0.0, min(1.0, edges))

    def get_gaussian_kernel(self, radius):
        if radius <= 0:
            return np.array([1.0], dtype=np.float32)
        sigma = max(radius / 2.0, 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / np.sum(kernel)

    def apply_gaussian_blur(self, image, radius):
        if radius <= 0:
            return image.copy()
        kernel = self.get_gaussian_kernel(int(radius))
        pad = len(kernel) // 2

        padded_x = np.pad(image, ((0, 0), (pad, pad)), mode='edge')
        temp = np.zeros_like(image, dtype=np.float32)
        for i, w in enumerate(kernel):
            temp += w * padded_x[:, i:i + image.shape[1]]

        padded_y = np.pad(temp, ((pad, pad), (0, 0)), mode='edge')
        out = np.zeros_like(image, dtype=np.float32)
        for i, w in enumerate(kernel):
            out += w * padded_y[i:i + image.shape[0], :]

        return out

    def sobel_filter(self, gray):
        Kx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)

        Ky = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)

        padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')

        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        gx = np.sum(windows * Kx, axis=(-2, -1))
        gy = np.sum(windows * Ky, axis=(-2, -1))

        return np.hypot(gx, gy)

    def dilate(self, mask, thickness):
        if thickness <= 1.0:
            return mask

        def dilate_step(m):
            p = np.pad(m, 1, mode='edge')

            windows = np.lib.stride_tricks.sliding_window_view(p, (3, 3))

            return np.max(windows, axis=(-2, -1))

        t = float(thickness)
        full_steps = int(t) - 1
        frac = t - int(t)

        res = mask.copy()
        for _ in range(full_steps):
            res = dilate_step(res)

        if frac > 0:
            next_step = dilate_step(res)
            res = res * (1.0 - frac) + next_step * frac

        return res

    def apply_filter(self, image):
        img_float = image.astype(np.float32)

        edges_channels = []
        for c in range(3):
            smooth_c = self.apply_gaussian_blur(img_float[:, :, c], radius=1)
            edges_channels.append(self.sobel_filter(smooth_c))

        edges = np.maximum.reduce(edges_channels)

        edge_threshold = 30 + 70*self.edges

        edges = np.where(edges < edge_threshold, 0.0, edges)

        if edges.max() > 0:
            edges = np.clip((edges - edge_threshold) * 2.5, 0, 255)

        white_line = self.dilate(edges, self.thickness)

        glow_mask = self.apply_gaussian_blur(white_line, radius=self.glow_radius)
        if glow_mask.max() > 0:
            glow_mask = np.clip(glow_mask * 2.5, 0, 255)

        dark_bg = img_float * (1.0 - self.intention)

        neon_layer = np.zeros_like(image, dtype=np.float32)
        neon_layer[:, :, 0] = glow_mask * 0.85
        neon_layer[:, :, 1] = glow_mask * 0.15
        neon_layer[:, :, 2] = glow_mask * 1.00

        white_layer = np.repeat(white_line[:, :, np.newaxis], 3, axis=2)

        result = dark_bg + neon_layer
        result = np.maximum(result, white_layer)

        return np.clip(result, 0, 255).astype(np.uint8)
