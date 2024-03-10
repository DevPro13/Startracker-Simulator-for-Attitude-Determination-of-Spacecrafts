from math import atan, tan
from decimal import Decimal

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Camera:
    def __init__(self, focal_length, x_center, y_center, x_resolution, y_resolution):
        self.focal_length = focal_length
        self.x_center = x_center
        self.y_center = y_center
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    def spatial_to_camera(self, vector):
        assert vector.x > 0  # can't handle things behind the camera

        focal_factor = self.focal_length / vector.x
        y_pixel = vector.y * focal_factor
        z_pixel = vector.z * focal_factor

        return Vec2(-y_pixel + self.x_center, -z_pixel + self.y_center)

    def camera_to_spatial(self, vector):
        assert self.in_sensor(vector)

        x_pixel = -vector.x + self.x_center
        y_pixel = -vector.y + self.y_center

        return Vec3(1, x_pixel / self.focal_length, y_pixel / self.focal_length)

    def in_sensor(self, vector):
        return 0 <= vector.x <= self.x_resolution and 0 <= vector.y <= self.y_resolution

    def fov(self):
        return self.focal_length_to_fov(self.focal_length, self.x_resolution, 1.0)

    @staticmethod
    def fov_to_focal_length(x_fov, x_resolution):
        return x_resolution / Decimal(2.0) / Decimal(tan(x_fov / 2))

    @staticmethod
    def focal_length_to_fov(focal_length, x_resolution, pixel_size):
        return atan(x_resolution / 2 * pixel_size / focal_length) * 2