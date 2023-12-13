"""
TODO: Put comments for clarity.
Code is inspired by existing threads on github and so on.
"""

import math
import argparse
import numpy as np

from time import perf_counter

def axis_rotation(polycube, axes):
    for i in range(4):
        yield np.rot90(polycube, i, axes)

def all_rotations_3d(polycube):
    yield from axis_rotation(polycube, (1,2))

    yield from axis_rotation(np.rot90(polycube, 2, axes=(0,2)), (1,2))
    yield from axis_rotation(np.rot90(polycube, axes=(0,2)), (0,1))
    yield from axis_rotation(np.rot90(polycube, -1, axes=(0,2)), (0,1))

    yield from axis_rotation(np.rot90(polycube, axes=(0,1)), (0,2))
    yield from axis_rotation(np.rot90(polycube, -1, axes=(0,1)), (0,2))


def zero_edges_crop(cube):
    for i in range(cube.ndim):
        cube = np.swapaxes(cube, 0, i)

        while np.all(cube[0] == 0):
            cube = cube[1:]

        while np.all(cube[-1] == 0):
            cube = cube[:-1]

        cube = np.swapaxes(cube, 0, i)

    return cube


def cube_expansion(cube):
    cube = np.pad(cube, 1, "constant", constant_values=0)
    cube_output = np.array(cube)

    xs, ys, zs = cube.nonzero()

    cube_output[xs+1, ys, zs] = 1
    cube_output[xs-1, ys, zs] = 1
    cube_output[xs, ys+1, zs] = 1
    cube_output[xs, ys-1, zs] = 1
    cube_output[xs, ys, zs+1] = 1
    cube_output[xs, ys, zs-1] = 1

    exp = (cube_output ^ cube).nonzero()

    for (x, y, z) in zip(exp[0], exp[1], exp[2]):
        cube_new = np.array(cube)
        cube_new[x, y, z] = 1

        yield zero_edges_crop(cube_new)

def rle(polycube):
    r = []
    r.extend(polycube.shape)
    current = None
    value = 0

    for i in polycube.flat:
        if current is None:
            current = i
            value = 1
            pass
        elif current == i:
            value += 1
        elif current != i:
            r.append(value if current == 1 else -value)
            current = i
            value = 1

    r.append(value if current == 1 else -value)

    return tuple(r)

def cube_exists_rle(polycube, polycubes_rle):
    for cube_rotation in all_rotations_3d(polycube):
        if rle(cube_rotation) in polycubes_rle:
            return True
        
    return False


def polycubes_generation(n):
    if n < 1:
        return []
    elif n == 1:
        return [np.ones((1, 1, 1), dtype=np.byte)]
    elif n == 2:
        return [np.ones((2, 1, 1), dtype=np.byte)]
    
    polycubes = []
    polycubes_rle = set()

    cube_base = polycubes_generation(n-1)

    for idx, cube in enumerate(cube_base):
        for new_cube in cube_expansion(cube):
            if not cube_exists_rle(new_cube, polycubes_rle):
                polycubes.append(new_cube)
                polycubes_rle.add(rle(new_cube))

    return polycubes


def shape_rendering(shapes, n_cubes):
    import matplotlib.pyplot as plt

    n = len(shapes)
    dim = max(max(x.shape) for x in shapes)

    i = math.isqrt(n) + 1
    voxel_dim = dim * i
    voxel_array = np.zeros((voxel_dim + i, voxel_dim + i, dim), dtype=np.byte)

    for idx, shape in enumerate(shapes):
        x = (idx % i) * dim + (idx % i)
        y = (idx // i) * dim + (idx // i)

        s = shape.shape
        voxel_array[x : x+s[0], y : y+s[1], 0 : s[2]] = shape

    voxel_array = zero_edges_crop(voxel_array)
    colors = np.empty(voxel_array.shape, dtype=object)
    colors[:] = '#FFD65DC0'

    ax = plt.figure(figsize=(20,16), dpi=600).add_subplot(projection='3d')
    ax.voxels(voxel_array, facecolors=colors, edgecolor='k', linewidth=0.1)

    ax.set_xlim([0, voxel_array.shape[0]])
    ax.set_ylim([0, voxel_array.shape[1]])
    ax.set_zlim([0, voxel_array.shape[2]])
    plt.axis('off')

    ax.set_box_aspect((1, 1, voxel_array.shape[2]/voxel_array.shape[0]))
    plt.savefig(f"{n_cubes}_cube_polycubes", bbox_inches='tight', pad_inches=0)


def main():
    parser = argparse.ArgumentParser(
        prog='Polycube generator',
        description="""Returns the amount of polycubes depending on the number of cubes chosen. 
        Additionally, allows for graphical representation."""
    )

    parser.add_argument('n', metavar='N', type=int,
                        help='The number of cubes for construction of polycube.')

    parser.add_argument('--image', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    n = args.n
    render = args.image if args.image is not None else False

    time_start = perf_counter()
    cubes_all = list(polycubes_generation(n))
    time_stop = perf_counter()

    print(f"I have found {len(cubes_all)} unique polycubes made out of {n} cubes or less.")
    print(f"I took me {round(time_stop - time_start, 3)}s.")

    if render == True:
        shape_rendering(cubes_all, n)


if __name__ == "__main__":
    main()
