from math import cos, sin, asin, atan2, sqrt, hypot, pi, acos

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def magnitude_sq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def magnitude(self):
        return hypot(hypot(self.x, self.y), self.z)

    def normalize(self):
        mag = self.magnitude()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def cross_product(self, other):
        return Vec3(self.y * other.z - self.z * other.y,
                    -(self.x * other.z - self.z * other.x),
                    self.x * other.y - self.y * other.x)

class Mat3:
    def __init__(self, x):
        self.x = x

    def at(self, i, j):
        return self.x[3 * i + j]

    def column(self, j):
        return Vec3(self.at(0, j), self.at(1, j), self.at(2, j))

    def row(self, i):
        return Vec3(self.at(i, 0), self.at(i, 1), self.at(i, 2))

    def __add__(self, other):
        return Mat3([self.at(0, 0) + other.at(0, 0), self.at(0, 1) + other.at(0, 1), self.at(0, 2) + other.at(0, 2),
                     self.at(1, 0) + other.at(1, 0), self.at(1, 1) + other.at(1, 1), self.at(1, 2) + other.at(1, 2),
                     self.at(2, 0) + other.at(2, 0), self.at(2, 1) + other.at(2, 1), self.at(2, 2) + other.at(2, 2)])

    def __mul__(self, other):
        return Mat3([
            self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0) + self.at(0, 2) * other.at(2, 0),
            self.at(0, 0) * other.at(0, 1) + self.at(0, 1) * other.at(1, 1) + self.at(0, 2) * other.at(2, 1),
            self.at(0, 0) * other.at(0, 2) + self.at(0, 1) * other.at(1, 2) + self.at(0, 2) * other.at(2, 2),
            self.at(1, 0) * other.at(0, 0) + self.at(1, 1) * other.at(1, 0) + self.at(1, 2) * other.at(2, 0),
            self.at(1, 0) * other.at(0, 1) + self.at(1, 1) * other.at(1, 1) + self.at(1, 2) * other.at(2, 1),
            self.at(1, 0) * other.at(0, 2) + self.at(1, 1) * other.at(1, 2) + self.at(1, 2) * other.at(2, 2),
            self.at(2, 0) * other.at(0, 0) + self.at(2, 1) * other.at(1, 0) + self.at(2, 2) * other.at(2, 0),
            self.at(2, 0) * other.at(0, 1) + self.at(2, 1) * other.at(1, 1) + self.at(2, 2) * other.at(2, 1),
            self.at(2, 0) * other.at(0, 2) + self.at(2, 1) * other.at(1, 2) + self.at(2, 2) * other.at(2, 2),
        ])

    def __rmul__(self, scalar):
        return Mat3([scalar * elem for elem in self.x])

    def transpose(self):
        return Mat3([self.at(0, 0), self.at(1, 0), self.at(2, 0),
                     self.at(0, 1), self.at(1, 1), self.at(2, 1),
                     self.at(0, 2), self.at(1, 2), self.at(2, 2)])

    def trace(self):
        return self.at(0, 0) + self.at(1, 1) + self.at(2, 2)

    def det(self):
        return (self.at(0, 0) * (self.at(1, 1) * self.at(2, 2) - self.at(2, 1) * self.at(1, 2))) - \
               (self.at(0, 1) * (self.at(1, 0) * self.at(2, 2) - self.at(2, 0) * self.at(1, 2))) + \
               (self.at(0, 2) * (self.at(1, 0) * self.at(2, 1) - self.at(2, 0) * self.at(1, 1)))

    def inverse(self):
        scalar = 1 / self.det()

        res = Mat3([
            self.at(1, 1) * self.at(2, 2) - self.at(1, 2) * self.at(2, 1), self.at(0, 2) * self.at(2, 1) - self.at(0, 1) * self.at(2, 2), self.at(0, 1) * self.at(1, 2) - self.at(0, 2) * self.at(1, 1),
            self.at(1, 2) * self.at(2, 0) - self.at(1, 0) * self.at(2, 2), self.at(0, 0) * self.at(2, 2) - self.at(0, 2) * self.at(2, 0), self.at(0, 2) * self.at(1, 0) - self.at(0, 0) * self.at(1, 2),
            self.at(1, 0) * self.at(2, 1) - self.at(1, 1) * self.at(2, 0), self.at(0, 1) * self.at(2, 0) - self.at(0, 0) * self.at(2, 1), self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0)
        ])

        return res * scalar

class Quaternion:
    def __init__(self, real, i, j, k):
        self.real = real
        self.i = i
        self.j = j
        self.k = k

    def __mul__(self, other):
        return Quaternion(
            self.real * other.real - self.i * other.i - self.j * other.j - self.k * other.k,
            self.real * other.i + other.real * self.i + self.j * other.k - self.k * other.j,
            self.real * other.j + other.real * self.j + self.k * other.i - self.i * other.k,
            self.real * other.k + other.real * self.k + self.i * other.j - self.j * other.i
        )

    def conjugate(self):
        return Quaternion(self.real, -self.i, -self.j, -self.k)

    def vector(self):
        return Vec3(self.i, self.j, self.k)

    def set_vector(self, vec):
        self.i = vec.x
        self.j = vec.y
        self.k = vec.z

    def rotate(self, vec):
        return (self * Quaternion(vec) * self.conjugate()).vector()

    def angle(self):
        if self.real <= -1:
            return 0
        return (0 if self.real >= 1 else acos(self.real)) * 2

    def smallest_angle(self):
        raw_angle = self.angle()
        return 2 * pi - raw_angle if raw_angle > pi else raw_angle

    def set_angle(self, new_angle):
        self.real = cos(new_angle / 2)
        self.set_vector(self.vector().normalize() * sin(new_angle / 2))

class EulerAngles:
    def __init__(self, ra, de, roll):
        self.ra = ra
        self.de = de
        self.roll = roll

def spherical_to_spatial(ra, de):
    return Vec3(
        cos(ra) * cos(de),
        sin(ra) * cos(de),
        sin(de)
    )

def spatial_to_spherical(vec):
    ra = atan2(vec.y, vec.x)
    ra += 2 * pi if ra < 0 else 0
    de = asin(vec.z)
    return ra, de

def rad_to_deg(rad):
    return rad * 180 / pi

def deg_to_rad(deg):
    return deg * pi / 180

def rad_to_arcsec(rad):
    return rad_to_deg(rad) * 3600

def arcsec_to_rad(arcsec):
    return deg_to_rad(arcsec / 3600)

def decimal_modulo(x, mod):
    result = x - mod * (x // mod)
    return result if result >= 0 else result + mod

def angle(vec1, vec2):
    return angle_unit(vec1.normalize(), vec2.normalize())

def angle_unit(vec1, vec2):
    dot = vec1 * vec2
    return 0 if dot >= 1 else pi - 0.0000001 if dot <= -1 else acos(dot)

class AttitudeType:
    QuaternionType = 0
    DCMType = 1

class Attitude:
    def __init__(self, quat=None, matrix=None):
        self.type = None
        self.quaternion = None
        self.dcm = None
        if quat is not None:
            self.type = AttitudeType.QuaternionType
            self.quaternion = quat
        elif matrix is not None:
            self.type = AttitudeType.DCMType
            self.dcm = matrix

    def get_quaternion(self):
        if self.type == AttitudeType.QuaternionType:
            return self.quaternion
        elif self.type == AttitudeType.DCMType:
            return dcm_to_quaternion(self.dcm)

    def get_dcm(self):
        if self.type == AttitudeType.DCMType:
            return self.dcm
        elif self.type == AttitudeType.QuaternionType:
            return quaternion_to_dcm(self.quaternion)

    def rotate(self, vec):
        if self.type == AttitudeType.DCMType:
            return self.dcm * vec
        elif self.type == AttitudeType.QuaternionType:
            return self.quaternion.rotate(vec)

    def to_spherical(self):
        if self.type == AttitudeType.DCMType:
            return self.get_quaternion().to_spherical()
        elif self.type == AttitudeType.QuaternionType:
            return self.quaternion.to_spherical()

    def is_known(self):
        return self.type == AttitudeType.DCMType or self.type == AttitudeType.QuaternionType

def quaternion_to_dcm(quat):
    x = quat.rotate(Vec3(1, 0, 0))
    y = quat.rotate(Vec3(0, 1, 0))
    z = quat.rotate(Vec3(0, 0, 1))
    return Mat3([x.x, y.x, z.x,
                 x.y, y.y, z.y,
                 x.z, y.z, z.z])

def dcm_to_quaternion(dcm):
    old_x_axis = Vec3(1, 0, 0)
    new_x_axis = dcm.column(0)
    x_align_axis = old_x_axis.cross_product(new_x_axis).normalize()
    x_align_angle = angle_unit(old_x_axis, new_x_axis)
    x_align = Quaternion(x_align_axis, x_align_angle)

    old_y_axis = x_align.rotate(Vec3(0, 1, 0))
    new_y_axis = dcm.column(1)
    rotate_clockwise = old_y_axis.cross_product(new_y_axis) * new_x_axis > 0
    y_align = Quaternion(Vec3(1, 0, 0), angle_unit(old_y_axis, new_y_axis) * (1 if rotate_clockwise else -1))

    return x_align * y_align

def serialize_vec3(vec):
    return [vec.x, vec.y, vec.z]

def deserialize_vec3(data):
    return Vec3(data[0], data[1], data[2])

def serialize_length_vec3():
    return 3

def serialize_context():
    pass

def deserialize_context():
    pass

def serialize_primitive(ser, val):
    pass

def deserialize_primitive(des):
    pass

def serialize_attitude(ser, attitude):
    ser.serialize_uint8(attitude.type)
    if attitude.type == AttitudeType.QuaternionType:
        ser.serialize_quaternion(attitude.quaternion)
    elif attitude.type == AttitudeType.DCMType:
        ser.serialize_mat3(attitude.dcm)

def deserialize_attitude(des):
    type_ = des.deserialize_uint8()
    if type_ == AttitudeType.QuaternionType:
        return Attitude(quat=des.deserialize_quaternion())
    elif type_ == AttitudeType.DCMType:
        return Attitude(matrix=des.deserialize_mat3())
