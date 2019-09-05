#!/usr/bin/env python3
"""Computed Tomography Lab
==========================

Main executable for the computed tomography lab
"""

import pathlib

import numpy as np

from attenuation_object import ObjectCollection, Rectangle, Circle, AttenuationObject
from misc import array_to_img
from projection import Projection




def main():
    """The main function in which everything you run should start."""
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()
#DAY 1
    print("DAY 1 \nSquare Trial")
    rectangle_1 = Rectangle(-1, 0, -1, 0, 1)
    rectangle_2 = Rectangle(0, 1, 0, 1, .5)
    circle_1 = Circle(.8, .1, .05, 1)
    #attenuation for a rectangle
    print('The attenuation at your point for the is:', rectangle_1.attenuation(7,0))
    collection = ObjectCollection()
    #collection.append(rectangle_1)
    #collection.append(rectangle_2)
    collection.append(circle_1)

#print(Projection.theta(theta_idx))

#DAY 2
    #attenutation of a circle because this would be useful for changing the
    #coordinates of eta and xi
    print('\nDAY 2 \nCircle Trial')

    print('The attenuation at (1.2 , 1.2):',circle_1.attenuation(1.2,1.2))
    print('Integrated attenuation for a given eta:', circle_1.project_attenuation(np.pi,0,(-2,2)))
    myproj = Projection([0,np.pi],100,[-2,2],100)
    myproj.add_object(collection,(-2,2))

    array_to_img(
        collection.to_array(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    ).save(output_dir / "myproj1(.8, .1, .05, 1).png")

    array_to_img(
        myproj.data
    ).save(output_dir / "sin(.8, .1, .05, 1).png")



if __name__ == "__main__":
    main()
