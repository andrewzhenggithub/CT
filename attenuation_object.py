"""Attenuation Object
=====================

Defines the abstract `AttenuationObject` class and provides basic
implementations of this class for some simple shapes.

"""
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import scipy.integrate


class AttenuationObject(ABC):
    """Abstract class for an attenuation object.

    An abstract class is a way of define what a set of objects should do,
    without explicitly implementing them all individually.

    """

    @abstractmethod
    def attenuation(self, x: float, y: float) -> float:
        """Return the attenuation factor μ at the point `(x, y)` in the
        original Cartesian coordinates.

        Note that this function is *abstract* meaning that the sub-class must
        implement this method.

        """

    def project_attenuation(
        self, theta: float, xi: float, eta_range: (float, float)
    ) -> float:
        """Return the integrated attenuation factor for a given xi and theta
        value.

        The integration is done along the specified range of eta values.

        """
        # Although we don't know what the underlying object is (as this is an
        # abstract class), we know that we will be able get the attenuation at
        # a given `(x, y)` coordinate.  Based on this alone, we can calculate
        # the projected attenuation.
        def integrand_eta(eta):
            x = xi * np.cos(theta) + eta * np.sin(theta)
            y = eta * np.cos(theta) -  xi * np.sin(theta)
            #this gives us the attenuation factor for a specific x and y in the
            #eta and xi coordinates
            return (self.attenuation(x , y))
        #Gives an integrand across all of eta
        return (sp.integrate.quad(integrand_eta,eta_range[0],eta_range[1])[0])

    def to_array(self, x, y):
        """Create a 2D array in the `(x, y)` coordinate of the attenuation.

        With the entries corresponding the the attenuation at the location.  No
        projection is done here, this function just creates a 2D 'image' of the
        projection over the specified range.

        """
        data = np.zeros((x.size, y.size), dtype=float)
        for idx in np.ndindex(data.shape):
            data[idx] = self.attenuation(x[idx[0]], y[idx[1]])

        return data


class Rectangle(AttenuationObject):
    """Implementat the `AttenuationObject` for a rectangle."""

    def __init__(
        self, x1: float, x2: float, y1: float, y2: float, attenuation_factor: float
    ):
        """Create a new Rectangle object bounded by `x1` and `x2` in the
        `x`-axis, and `y1` and `y2` in the `y`-axis.

        """
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        self.attenuation_factor = attenuation_factor

    def attenuation(self, x, y):
        if self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2:
            return self.attenuation_factor
        else:
            return 0

class Circle(AttenuationObject):
    """Implementat the `AttenuationObject` for a circle."""

    def __init__(self, x0, y0, r, attenuation_factor):
        """Create a new Circle object centered on `(x0, y0)` with radius `r`
        and the specified `attenuation_factor`."""
        self.x0 = x0
        self.y0 = y0
        self.r = abs(r)
        self.attenuation_factor = attenuation_factor

    def attenuation(self, x, y):

        if (np.sqrt((x - self.x0)**2 + (y - self.y0)**2)) <= self.r:
            return self.attenuation_factor
        else:
            return 0


class ObjectCollection(AttenuationObject):
    """Implementat the `AttenuationObject` for a collection of objects."""

    def __init__(self):
        self.objects = []

    def append(self, obj):
        """Add a new object to the collection."""
        if not isinstance(obj, AttenuationObject):
            raise RuntimeError("Object must be an AttenuationObject.")

        self.objects.append(obj)

    def attenuation(self, x, y):
        """Return the overall attenuation from all objects in the
        collection."""
        attenuation = 0
        for obj in self.objects:
            attenuation += obj.attenuation(x, y)
        return attenuation


class ImageObject(AttenuationObject):
    """Implement the `AttenuationObject` for images.

    The image is rescaled so that if fits within the `[-1, 1]^2`.  For a square
    image, this means that the coordinate `(-1, -1)` will correspond to the
    lower left corner, and the coordinate `(1, 1)` corresponds to the upper
    right corner.

    """

    def __init__(self, image):
        self.image = image.copy().convert(mode="L")
        self.width, self.height = self.image.size
        self.scale_factor = max(self.width, self.height) / 2

    def attenuation(self, x, y):
        # We scale the input (x, y) coordinates so the correspond to a pixel
        # index.
        x *= self.scale_factor
        y *= self.scale_factor

        x += self.width / 2
        y += self.height / 2

        # Get the pixel as the desired point if we're in the image, otherwise
        # return 0.
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.image.getpixel((x, y))
        return 0
