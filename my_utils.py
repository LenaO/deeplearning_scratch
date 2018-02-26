from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import re
import os
import glob
import logging
import logging.handlers
import errno
from test_params import section_file

# flatten a list of lists
def iter_flatten(iterable):
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def flatten(l):
    return [e for e in iter_flatten(l)]



def init_logging(level=logging.INFO, logfile=None):
    """Convenience function to initialize logging.

    Should only be called once for each program.

    Args:
        level: logging level (default: logging.INFO).
        logfile (string, optional): if specified, the output is logged to this file.
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format='%(asctime)s %(name)s [%(levelname)s]:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if logfile is not None:
        fh = logging.handlers.RotatingFileHandler(logfile)
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(fh)


def get_logger(name, rank=None):
    """Returns a logger with an optional prefix '[Rank rank]'.

    Can be called in every module that wants to do logging.

    Args:
        name (string): name of the logger, typically ``__name__``.
        rank (int, optional): rank of the process requesting the logger. If specified, `[Rank rank]` is prefixed to every log message.

    Returns:
        logging.Logger: logger instance that can be used to log messages, using the ``debug(), info(), warning()`` and ``error()`` functions.
    """
    class MPILogger(logging.LoggerAdapter):
        def __init__(self, prefix, logger):
            super(MPILogger, self).__init__(logger, {})
            self.prefix = prefix

        def process(self, msg, kwargs):
            return '[Rank %s] %s' % (self.prefix, msg), kwargs

    logger = logging.getLogger(name)
    if rank is not None:
        logger = MPILogger(rank, logger)
    return logger

def uint8(img):
    img = np.array(img)
    if img.dtype == np.uint8:
        return img

    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


def dtype_limits(dtype):
    """Returns the min and max values for a given dtype.

    Supports numpy dtypes (integer or float)

    Args:
        dtype: numpy dtype

    Returns:
        tuple: min and max value of the dtype.
    """

    dtype = np.dtype(dtype)

    if issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
    elif issubclass(dtype.type, np.floating):
        info = np.finfo(dtype)
    return info.min, info.max

def isstring(obj):
    """ Checks if the given object is a string. Python 2 and 3 compatible"""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

def coord_to_mesh(coord, section, spacing, offset=[0,0], _type='obj'):
    """ Convert 2d coordinate of section number in mesh mm coordinate. Spacing and offset are defined for the coordinate in the pixel volume.
    _type tells us about type of mesh (nii (OLD) and obj (NEW))"""
    if _type == 'nii':
        # mesh has other coordinate system (invert all axes, then transpose (0,2,1))
        res_section = -1*(-70+(int(section)-1)*0.02)
        res_coord_x, res_coord_z = -1*(np.array(coord) * spacing + offset)
    elif _type == 'obj':
        # invert z axis
        res_section = (-70+(int(section)-1)*0.02)
        res_coord_x = coord[0]*spacing - 70.5666
        res_coord_z = -1*coord[1]*spacing - 58.6777 + 121.
    return [res_coord_x, res_section, res_coord_z]


def mesh_to_coord(mesh_coord, spacing):
    """ Convert mesh coordinate to pixel-volume coordinate. ONLY works for NEW mesh (obj file) """
    section = int(round((mesh_coord[1]+70)/0.02+1))
    coord_x = (1*mesh_coord[0] + 70.5666)/spacing
    coord_z = (121. - 1*mesh_coord[2] - 58.6777)/spacing
    return section, [coord_x, coord_z]

def get_sections_for_coords(section_file=section_file, include_labels=['good', 'excellent'], split=None, spacing=1., offset=0.):
    available_sections = get_available_sections(section_file, include_labels, split)
    rounded_coords = [int(round(section_to_coord(sec, spacing, offset))) for sec in available_sections]
    sections = {coord:list(np.array(available_sections)[np.array(rounded_coords)==coord]) for coord in np.unique(rounded_coords)}
    return sections

def get_direction(points1, points2):
    """ Calculates the absolute normalize directions beween points1 and points2/
    Args:
        points1, points2: array-like of shape (num_examples, dims)
    """
    d = np.abs(np.asarray(points1) - np.asarray(points2))
    d = d / np.linalg.norm(d, axis=-1)[:,None]
    return d


def get_sides(points):
    sides = []
    for pt in points:
        sides.append(pt['side'])
    return np.array(sides)


