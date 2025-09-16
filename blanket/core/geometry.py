from __future__ import annotations

import numpy as np


class SO3:
    """
    This class represents an SO3 rotations internally represented by rotation matrix.
    """

    def __init__(self, rotation_matrix: np.ndarray | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""

        super().__init__()
        self.rot: np.ndarray = np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)

    @staticmethod
    def exp(rot_vector: np.ndarray) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""

        v = np.asarray(rot_vector)
        assert v.shape == (3,)

        angle = np.linalg.norm(v)
        return SO3.from_angle_axis(angle, v / angle) if angle != 0 else SO3()

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""

        angle, axis = self.to_angle_axis()
        return angle * axis if axis is not None else np.zeros((3,))

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""

        return SO3(self.rot @ other.rot)

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""

        return SO3(np.transpose(self.rot))

    def act(self, vector: np.ndarray) -> np.ndarray:
        """Rotate given vector by this transformation."""

        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""

        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x-axis."""

        return SO3.from_angle_axis(angle, np.asarray([1, 0, 0]))

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y-axis."""

        return SO3.from_angle_axis(angle, np.asarray([0, 1, 0]))

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z-axis."""

        return SO3.from_angle_axis(angle, np.asarray([0, 0, 1]))

    @staticmethod
    def from_quaternion(q: np.ndarray) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""

        return SO3.from_angle_axis(2 * np.arccos(q[3]), q[0:3] / np.linalg.norm(q[0:3]))

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""

        angle, axis = self.to_angle_axis()
        return np.asarray((*(axis * np.sin(angle / 2)), np.cos(angle / 2)))

    @staticmethod
    def from_angle_axis(angle: float, axis: np.ndarray) -> SO3:
        """Compute rotation from angle axis representation."""

        assert len(axis) == 3
        # assert np.linalg.norm(axis) == 1

        omega_matrix = np.asarray([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        return SO3(np.eye(3) + np.sin(angle) * omega_matrix + (1 - np.cos(angle)) * (omega_matrix @ omega_matrix))

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""

        if np.allclose(self.rot, np.eye(3)):
            angle = 0
            axis = np.asarray([1, 0, 0])

        elif np.trace(self.rot) == -1:
            angle = np.pi
            for index in range(2, -1, -1):
                if self.rot[index][index] != -1:
                    axis = np.asarray([self.rot[0][index], self.rot[1][index], self.rot[2][index]])
                    axis[index] += 1
                    axis /= np.sqrt(2 + 2 * self.rot[index][index])
                    break

        else:
            angle = np.arccos((np.trace(self.rot) - 1) / 2)
            skew_symmetric_matrix = (self.rot - np.transpose(self.rot)) / (2 * np.sin(angle))
            axis = np.asarray([skew_symmetric_matrix[2][1], skew_symmetric_matrix[0][2], skew_symmetric_matrix[1][0]])

        return angle, axis

    @staticmethod
    def from_euler_angles(angles: np.ndarray, seq: list[str] | str) -> SO3:
        """
        Compute rotation from euler angles defined by a given sequence.

        :param angles: is a three-dimensional array of angles
        :param seq: is a list of axes around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """

        if type(angles) is not np.ndarray:
            angles = np.asarray(angles)

        assert angles.shape == (len(seq),)

        final_rot = SO3()
        axes = "xyz"
        axis_vectors = (np.asarray([1, 0, 0]), np.asarray([0, 1, 0]), np.asarray([0, 0, 1]))

        for angle, axis in zip(angles, seq):
            if axis not in axes:
                raise ValueError(f'Unknown axis "{axis}" in seq:"{seq}" (accepting only x, y, z)')
            final_rot *= SO3.from_angle_axis(angle, axis_vectors[axes.index(axis)])

        return final_rot

    def to_euler_angles_zyx(self) -> np.ndarray:
        if self.rot[0, 0] == 0 and self.rot[1, 0] == 0:
            x_rot = np.arctan2(self.rot[0, 1], self.rot[1, 1])
            y_rot = np.pi / 2
            z_rot = 0

        else:
            try:
                x_rot = np.arctan2(self.rot[2, 1], self.rot[2, 2])
                y_rot = np.arctan2(-self.rot[2, 0], np.sqrt(self.rot[0, 0] ** 2 + self.rot[1, 0] ** 2))
                z_rot = np.arctan2(self.rot[1, 0], self.rot[0, 0])
            except ValueError:
                print(
                    f"Couldn't find Euler angles for matrix:\n{self}\n"
                    f"x_rot = np.arctan2({self.rot[2, 1]}, {self.rot[2, 2]})\n"
                    f"y_rot = np.arctan2({-self.rot[2, 0]}, {np.sqrt(self.rot[0, 0] ** 2 + self.rot[1, 0] ** 2)})\n"
                    f"z_rot = np.arctan2({self.rot[1, 0]}, {self.rot[0, 0]})"
                )
                x_rot, y_rot, z_rot = 0, 0, 0
            except TypeError:  # usually caused by complex valued rotation matrix
                # (imaginary parts are zero so just taking the real part is fine)
                print(
                    f"Couldn't find Euler angles for matrix:\n{self}\n"
                    f"x_rot = np.arctan2({self.rot[2, 1]}, {self.rot[2, 2]})\n"
                    f"y_rot = np.arctan2({-self.rot[2, 0]}, {np.sqrt(self.rot[0, 0] ** 2 + self.rot[1, 0] ** 2)})\n"
                    f"z_rot = np.arctan2({self.rot[1, 0]}, {self.rot[0, 0]})"
                )
                x_rot, y_rot, z_rot = 0, 0, 0

        return np.asarray([z_rot, y_rot, x_rot])

    def _get_axis(self, axis_index):
        return self.rot[:, axis_index]

    @property
    def x_axis(self):
        return self._get_axis(0)

    @property
    def y_axis(self):
        return self._get_axis(1)

    @property
    def z_axis(self):
        return self._get_axis(2)

    def rot_90_x(self):
        new_matrix = self.rot.copy()
        new_matrix[:, 1], new_matrix[:, 2] = new_matrix[:, 2], -new_matrix[:, 1]

        return SO3(new_matrix)

    def rot_90_y(self):
        new_matrix = self.rot.copy()
        new_matrix[:, 0], new_matrix[:, 2] = new_matrix[:, 2], -new_matrix[:, 0]

        return SO3(new_matrix)

    def rot_90_z(self):
        new_matrix = self.rot.copy()
        new_matrix[:, 0], new_matrix[:, 1] = new_matrix[:, 1], -new_matrix[:, 0]

        return SO3(new_matrix)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"SO3(rot={self.rot}"
