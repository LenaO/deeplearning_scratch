from __future__ import print_function, division, absolute_import
import os
import math
import numpy as np
import h5py
import pytiff
import skimage.transform
import scipy.ndimage
from my_utils import get_logger, dtype_limits, isstring

from bounding_box import BoundingBox
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("image: Module cv2 is not available, using scikit-image as a fallback")
    CV2_AVAILABLE = False



# modes for opening an Image
READONLY = 'r'
READWRITE = 'r+'

# interpolation strategies for rescaling
NEAREST = 0
LINEAR = 1
CUBIC = 3

def resize(img, shape, interpolation=CUBIC, antialias=True):
    """Resize the image to specified shape using the given interpolation.
    If antialias is defined, a gauss filter will smooth the image before downsizing.
    If interplation is NEAREST, antialiassing is turned of.

    Args:
        img (array-like): image to resize.
        shape (tuple): shape of the resized image.
        interpolation (int): interpolation strategy.
        antialias (bool): smooth image before downsizing.
    """
    # check valid size of image
    if min(img.shape) == 0:
        log.warn("Not possible to resize image of shape {}".format(img.shape))
        return img
    # take care of rgb images and 2dim shapes
    if len(img.shape) == 3 and len(shape) == 2:
        shape = (shape[0], shape[1], img.shape[2])
    arr = img
    if antialias and interpolation != NEAREST:
        # if target shape is larger than image shape, sigma will be 0 (no filtering)
        filter_width = np.round(np.asarray(img.shape, dtype=float) / np.asarray(shape) * 2).astype(int)
        sigma = (filter_width-1)/2. - 0.5
        arr = scipy.ndimage.filters.gaussian_filter(img, sigma=sigma, truncate=1)

    if CV2_AVAILABLE:
        interpolation_codes = {NEAREST: cv2.INTER_NEAREST, LINEAR: cv2.INTER_LINEAR, CUBIC: cv2.INTER_CUBIC}
        res = cv2.resize(src=arr, dsize=(shape[1], shape[0]), interpolation=interpolation_codes[interpolation])
    else:
        res = skimage.transform.resize(arr, shape[:2], order=interpolation, preserve_range=True, mode="constant")

    return res.astype(img.dtype)




def open_image(filename, page=0):
    """ Open image (lazy, i.e. not reading the whole image if not necessary).

    Args:
        filename: image filename.
        page (str or int): page of tiff file or internal path to dataset in HDF5 file.

    Returns:
        h5py._hl.dataset.Dataset, pytiff._pytiff.Tiff or numpy.ndarray
            containing image in row x col format
    """
    print("open Image {}".format(filename))
    if filename.endswith('.tif'):
        # open image with pytiff
        img = pytiff.Tiff(filename)
        img.set_page(int(page))
    elif filename.endswith('.h5') or filename.endswith('.hdf5'):
        # open image with h5py
        f = h5py.File(filename, 'r')
        if page == 0:
            page = "/"  # set default page for hdf5 files
        img = f[page]
    elif filename.endswith('.nii'):
        import nibabel as nib
        img = np.transpose(nib.load(filename).get_data())
        # Reshape image, since some nii files contain 4 dimension
        shape = img.shape
        if len(shape) > 2:
            # After transposing, we have to adjust the dimensions
            img = img.reshape(shape[2], shape[3])
    else:
        img = skimage.io.imread(filename)  #, as_grey=True)
    # if rgb, convert to grayscale
    #if len(img.shape) > 2:
    #    img = skimage.color.rgb2gray(img)
    return img


def get_maxintensity(filename, attributePath=None):
    """Get maximal pixel intensity of given image.

    Args:
        filename (str): Image filename.
        attributePath (str): Path to the maxintensity attribute in a hdf5 file.
    Returns:
        float: maximal pixel intensity of given image.
    """
    close_file = False
    if isinstance(filename, str):
        img = open_image(filename, page=0)
        if isinstance(img, pytiff._pytiff.Tiff) or isinstance(img, h5py._hl.dataset.Dataset):
            close_file = True
    else:
        img = filename
    if isinstance(img, pytiff._pytiff.Tiff):
        maxintensity = 2.**img.n_bits
    elif isinstance(img, h5py._hl.dataset.Dataset):
        maxintensity = img[attributePath].attrs["maxintensity"]
    else:
        maxvalues = np.array([2**b-1 for b in [8,16,32,64]])
        imgmax = np.amax(img)
        maxintensity = maxvalues[np.where(maxvalues - imgmax >= 0)[0][0]]
        if imgmax < 1:
            maxintensity = 1
    if close_file:
        img.close()
    return maxintensity



class Image(object):
    """Basic Image class. Implemented by Hdf5Image, TiffImage and MemoryImage.

    It has some basic functionality on it's own, but it does not make
    a lot of sense to use this class directly.
    """
    def __init__(self, spacing, size, scales, maxintensity=255):
        """
        Args:
            spacing (number): spacing of the image (in um).
            size (2-tuple): (h,w) size of the image in the current spacing.
            scales (array-like): available scales of the image.
            maxintensity (int): maxintensity of the image (e.g. 127). Will be rounded up to the next power of 2 - 1.
        """
        assert len(size) == 2
        if spacing is None:
            self._spacing = 1.
        else:
            self._spacing = float(spacing)
        self._scales = np.asarray(scales)
        self._size = size
        if maxintensity == dtype_limits('float64')[1]:
            # avoid overflow when calcuating maxintensity
            self._maxintensity = maxintensity
        else:
            self._maxintensity = 2**math.ceil(math.log(maxintensity+1,2))-1

    def __repr__(self):
        return 'Image(spacing={}, size={}, scales={}, maxintensity={})'.format(
            self._spacing, self._scales, self._size, self.maxintensity)

    @staticmethod
    def get_image(filename, prefix='/', spacing=None, maxintensity=None):
        """ Create an instance of Hdf5Image, TiffImage or MemoryImage.

        Args:
            filename (string or array-like): filename of the image, or already opened image.
            prefix (string, optional): if the file is an hdf5, prefix points to the image
                inside the hdf5.
            spacing (int, optional): spacing of the image/largest pyramid layer.
        """
        # Check for basestring instead of str, because creating a MemoryImage from unicode is not a good idea...
        if isstring(filename):
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                return Hdf5Image(filename, prefix=prefix, spacing=spacing, maxintensity=maxintensity)
            elif filename.endswith('.tif') or filename.endswith('.tiff'):
                return TiffImage(filename, spacing=spacing, maxintensity=maxintensity)
            else:
                img = open_image(filename)
                return MemoryImage(img, spacing=spacing, maxintensity=maxintensity)
 
    @property
    def scales(self):
        """Scales that are available for this image."""
        return np.copy(self._scales)

    @property
    def spacing(self):
        """The spacing of the image (in um)."""
        return self._spacing

    @property
    def maxintensity(self):
        """The maxintensity of the image (rounded up to next power of 2)."""
        return self._maxintensity

    def get_size(self, spacing=1):
        """Return the size of the image give the specified spacing.

        Args:
            spacing (int, optional): spacing used to calculate the size (in um).
        """
        size = tuple(map(lambda s: int(s*self._spacing/float(spacing)), self._size))
        return size

    def _get_available_scale(self, spacing):
        """Return the scale level of the pyramid which has a spacing closest to the required
        spacing (is smaller if possible)"""
        factor = self.spacing_to_scale(spacing)
        available_scales = self._scales[self._scales >= factor]
        if len(available_scales) > 0:
            scale = min(available_scales)
        else:
            # there are no larger scales (need to upscale)
            scale = max(self.scales)
        return scale

    def spacing_to_scale(self, spacing):
        return self._spacing / float(spacing)

    def scale_to_spacing(self, scale):
        return self._spacing / float(scale)


    #TODO problems if blocksize * scale is not a natural number.
    @staticmethod
    def crop_from_array(arr, bbox, arr_spacing=1., wanted_spacing=1., interpolation=CUBIC, dtype=None, blocksize=8000):
        """Crop bbox from arr and rescale to wanted_spacing.

        Assume arr has spacing arr_spacing. Respect bbox.spacing
        Args:
            arr (arry-like): image to crop from with spacing arr_spacing.
            bbox (BoundingBox): defines the crop area with spacing bbox.spacing.
            arr_spacing (float): spacing of arr.
            wanted_spacing (float): spacing of the returned crop.
            interpolation (int): interpolation used for eventual rescaling.
                From skimage.transform.rescale
            dtype (numpy dtype): integer dtype of the resulting crop.
        Returns:
            (np.ndarray): cropped image
        """
        arr_spacing = float(arr_spacing)
        wanted_spacing = float(wanted_spacing)

        if bbox is None:
            bbox = BoundingBox.from_corners((0, 0), (arr.shape[1], arr.shape[0]), spacing=arr_spacing)
        else:
            bbox = BoundingBox.copy(bbox)
            bbox.set_to_spacing(arr_spacing)
        assert bbox.spacing == arr_spacing
        output_size = tuple([int(math.floor(s * arr_spacing/wanted_spacing)) for s in bbox.get_hw(round_values=False)])
        if len(arr.shape) > 2:
            # Allow cropping from arrays with more than 2 dimensions
            output_size = output_size + arr.shape[2:]
        if dtype is None:
            result_crop = np.zeros(output_size, dtype=arr.dtype)
        else:
            result_crop = np.zeros(output_size, dtype=dtype)
        if not (blocksize * arr_spacing/wanted_spacing).is_integer():
            log.warning("blocksize * scale is not a natural number:{}. This may lead to problems in blockwise resizing.".format(blocksize * arr_spacing/wanted_spacing))
        for box in blocks(bbox,(blocksize,blocksize)):
            # box is part of bbox, has to be moved by vector of upper left corner
            box.set_center(np.array(box.center)+np.array(bbox.get_corners(round_values=False)[0]))
            output_size = tuple([int(math.floor(s * (arr_spacing/wanted_spacing))) for s in box.get_hw(round_values=False)])
            if min(output_size) == 0: continue
            #pyextrae.eventandcounters(5000, 20)
            crop = np.array(box.crop_from_array(arr))
            #pyextrae.eventandcounters(5000, 0)
            if arr_spacing != wanted_spacing:
                # rescale the crop to fit to output_size
                #log.debug("crop_from_array is reshaping from {} to {}".format(crop.shape, output_size))
                crop = resize(crop, output_size, interpolation=interpolation, antialias=True)
                assert crop.shape[:2] == output_size, "{} {}".format(crop.shape,output_size)

            if dtype is not None:
                crop = crop.astype(np.float64) / get_maxintensity(arr) * dtype_limits(dtype)[1]
                crop = crop.astype(dtype)
            # move box back to origin and insert crop into result_crop
            box.set_center(np.array(box.center)-np.array(bbox.get_corners(round_values=False)[0]))
            box.rescale(arr_spacing/wanted_spacing)
            box.set_hw(box.get_hw()[0],box.get_hw()[1])
            box.write_to_array(result_crop,crop)
        return result_crop


class Hdf5Image(Image):
    """Represents an hdf5 image file.

    Manages pyramid scales and enables cropping from the image using an arbitrary
    spacing. Creation of a new pyramid is also possible.

    The image is identified by either a pyramid (largest scale in h5[prefix +
    '/pyramid/00']) or by a dataset in h5[prefix].

    If the image has a pyramid structure, the spacing can be read from the attribute
    'spacing' in prefix+'/pyramid'. Additionally, an attribute 'scales' is expected.
    If the image is given by a dataset, the location of the spacing attribute is
    in prefix.

    The attribute 'maxintensity' is also read if present.

    Usage example:
    ```python
        filename = 'path/to/file.hdf5'
        img = Hdf5Image(filename, mode=READWRITE)

        # create pyramid (path /pyramid/00 ...)
        pyramid_img = img.create_pyramid('/')

        # get crop
        bbox = BoundingBox((100,100), 10, 10, spacing=1)
        crop = img.get_crop(bbox, spacing=2)
    ```
    """

    def __init__(self, filename, prefix='/', spacing=None, mode=READONLY, maxintensity=None):
        """Initialize Hdf5Image.

        Args:
            filename (string): path to hdf5 file.
            prefix (string, optional): path to the image dataset inside the hdf5.
                Can either point to a group containing 'pyramid/00', etc., or
                an image dataset.
            spacing (number, optional): the spacing of the image. Normally read from
                the 'spacing' attribute in the hdf5 file.
            mode (string): open the file for reading or for writing?
            maxintensity (int): maxintensity of the image (e.g. 127). If not specified,
                is set to max value of the image dtype.
        """

        f = h5py.File(filename, mode)
        dset = f[prefix]
        attrs = dset.attrs
        if isinstance(dset, h5py.Group) and 'pyramid' in dset.keys():
            dset = dset['pyramid']
            scales = dset.attrs['scales']
            levels = [dset[lvl] for lvl in sorted(dset.keys())]
        elif isinstance(dset, h5py.Dataset):
            scales = [1.]
            levels = [dset]
            attrs = dset.attrs
        else:
            raise ValueError("No pyramid or dataset found in file {} at prefix {}".format(filename, prefix))
        if 'spacing' in dset.attrs:
            sp = dset.attrs['spacing']
            if spacing is not None and sp != spacing:
                log.info('specified spacing of {0} on initialization, but spacing set in hdf5 is {1}. \
                Using spacing {1}'.format(spacing, sp))
            spacing = sp

        # TODO temporary workarounds for HANNAH
        elif 'scale0' in dset.attrs:
            spacing = 1./dset.attrs['scale0']
            log.debug('found depreceated scale0 in file {} (setting spacing to {})'.format(filename, spacing))
        elif 'scale' in dset.attrs:
            spacing = 1./dset.attrs['scale']
            log.debug('found depreceated scale in file {} (setting spacing to {})'.format(filename, spacing))

        if spacing is None:
            spacing = 1.
        if 'maxintensity' in dset.attrs:
            maxint = dset.attrs['maxintensity']
            if maxintensity is not None and maxint != maxintensity:
                log.info('specified maxintensity of {0} on initialization, but maxintensity set in hdf5 is {1}. \
                Using maxitensity {1}'.format(maxintensity, maxint))
            maxintensity = maxint
        if maxintensity is None:
            maxintensity = dtype_limits(levels[0].dtype)[1]
        size = levels[0].shape[:2]

        super(Hdf5Image, self).__init__(spacing, size, scales, maxintensity)
        self._levels = levels
        self._file_handle = f
        self._mode = mode
        self._dset = dset
        self.attrs = attrs
        self._prefix = prefix

    def __repr__(self):
        return 'Hdf5Image("{}", prefix="{}", spacing={}, mode="{}", maxintensity={})'.format(
            self._file_handle.file.filename, self._prefix, self.spacing, self._mode,
            self.maxintensity)

    @property
    def dataset(self):
        """Dataset or group containing the image"""
        return self._dset

    def get_level(self, spacing):
        """Return h5py.Dataset containing the image at the specified spacing.
        If this spacing does not exists, returns None.

        Args:
            spacing (number): the spacing of the wanted level.
        """
        scale = self.spacing_to_scale(spacing)
        if scale in self.scales:
            lvl = self._levels[list(self.scales).index(scale)]
            return lvl
        return None

    def get_crop(self, bbox, spacing, dtype=None, interpolation=CUBIC):
        """Crop bbox from image.

        If a dtype is given, the returned values are divided by the maxintensity and scaled
        to the dtype.

        Args:
            bbox (BoundingBox): box with spacing box.spacing. If none, the whole image is cropped.
            spacing (number): spacing that the resulting crop should have.
            dtype (numpy dtype): integer dtype of the resulting crop.
            interpolation (int): interpolation method for eventual rescale of the crop.
        Returns:
            (np.ndarray): cropped image.
        """
        scale = self._get_available_scale(spacing)
        img = self._levels[list(self._scales).index(scale)]

        img_spacing = self.scale_to_spacing(scale)
        crop = Image.crop_from_array(img, bbox, img_spacing, spacing, interpolation)
        if dtype is not None:
            crop = crop.astype(np.float64) / self.maxintensity * dtype_limits(dtype)[1]
            crop = crop.astype(dtype)
        return crop


class TiffImage(Image):
    """Represents an tiff image file (using pytiff).

    Manages pyramid scales and enables cropping from the image using an arbitrary
    spacing.
    For each pyramid level, a separate file is opened to the correct page, allowing for concurrent
    reads of different levels using the same TiffImage.
    """

    def __init__(self, filename, spacing=1, mode=READONLY, maxintensity=None):
        """Initialize TiffImage.
        TiffImage assumes that the pyramids are calculated with scales that are powers of two!
        E.g. 0.5, 0.25, etc.

        Args:
            filename (string): path to hdf5 file.
            spacing (number, optional): the spacing of the image. Default is 1.
            mode (string): open the file for reading or for writing?
            maxintensity (int): maxintensity of the image (e.g. 127). If not specified,
                is set to max value of the image dtype.
        """
        f = pytiff.Tiff(filename, mode)
        # get scales and levels array
        shapes = []
        pages = []
        levels = []
        for i in range(f.number_of_pages):
            f.set_page(i)
            if f.shape not in shapes:
                shapes.append(f.shape)
                pages.append(i)
            else:
                ind = shapes.index(f.shape)
                if f.is_tiled():
                    # prefer to use the tiled page
                    pages[ind] = i
        f.set_page(0)
        for page in pages:
            img = pytiff.Tiff(filename, mode)
            img.set_page(page)
            levels.append(img)
        scales = [shape[0] / float(shapes[0][0]) for shape in shapes]
        #log.debug("TiffImage - Scales before rounding: %s", scales)
        scales = [1/2**round(math.log(1/scale, 2)) for scale in scales]
        #log.debug("TiffImage - Scales after rounding: %s", scales)
        if maxintensity is None:
            maxintensity = dtype_limits(f.dtype)[1]
        size = shapes[0][:2]
        if spacing is None:
            spacing = 1

        super(TiffImage, self).__init__(spacing, size, scales, maxintensity)
        self._file_handle = f
        self._mode = mode
        self._pages = pages
        self._levels = levels

    def __repr__(self):
        return 'TiffImage("{}", spacing={}, mode="{}", maxintensity={})'.format(
            self._file_handle.filename,
            self.spacing,
            self._mode,
            self.maxintensity)

    def get_level(self, spacing):
        """Return pytiff.Tiff image opened at the specified spacing.
        If this spacing does not exists, returns None

        Args:
            spacing (number): the spacing of the wanted level.
        """
        scale = self.spacing_to_scale(spacing)
        if scale in self.scales:
            lvl = pytiff.Tiff(self._file_handle.filename, READONLY)
            lvl.set_page(self._pages[list(self.scales).index(scale)])
            return lvl
        return None

    def get_crop(self, bbox, spacing, dtype=None, interpolation=CUBIC):
        """Crop bbox from image.

        If a dtype is given, the returned values are divided by the maxintensity and scaled
        to the dtype.

        Args:
            bbox (BoundingBox): box with spacing box.spacing. If none, the whole image is cropped.
            spacing (number): spacing that the resulting crop should have.
            dtype (numpy dtype): integer dtype of the resulting crop.
            interpolation (string): interpolation method for eventual rescale of the crop.
        Returns:
            (np.ndarray): cropped image.
        """
        scale = self._get_available_scale(spacing)
        lvl = self._levels[list(self.scales).index(scale)]

        img_spacing = self.scale_to_spacing(scale)

        crop = Image.crop_from_array(lvl, bbox, img_spacing, spacing, interpolation)
        if dtype is not None:
            crop = crop.astype(np.float64) / self.maxintensity * dtype_limits(dtype)[1]
            crop = crop.astype(dtype)
        return crop

def blocks(data, size):
    """A generator yielding sliding bounding boxes with a given size over given data.

    Args:
        data : provides a shape attribute. E.g. a hdf5 dataset or a (py)tiff image
        size : sets the size of the generated bounding boxes.

    Returns:
        BoundingBox : A bounding box from the given data.
    """
    size = np.array(size)
    shape = np.array(data.shape)
    n_rows, n_cols = np.ceil(shape[:2]/size).astype(int)
    for r in range(n_rows):
        for c in range(n_cols):
            # swap row and column, x = column, y = row
            upper_left = (c*size[1], r * size[0])
            lower_right = (min((c+1)*size[1], shape[1]), min((r+1)*size[0], shape[0]))
            yield BoundingBox.from_corners(upper_left, lower_right)



class MemoryImage(Image):
    """Represents an image from memory.

    Supports creation of pyramids, extraction of crops and saving as tiff or hdf5 image.
    """

    def __init__(self, arr, spacing=1., maxintensity=None):
        """ Initialize MemoryImage.

        Args:
            arr (array-like): the image.
            spacing (float): spacing of the image.
            maxintensity (int): maxintensity of pixel values in the image.
        """
        levels = [arr]
        scales = [1.]
        if maxintensity is None:
            maxintensity = dtype_limits(levels[0].dtype)[1]

        size = levels[0].shape[:2]
        super(MemoryImage, self).__init__(spacing, size, scales, maxintensity)
        self._levels = levels

    def __repr__(self):
        return 'MemoryImage(arr={}, spacing={}, maxintensity={})'.format(
            self._levels[0], self.spacing, self.maxintensity)

    def get_crop(self, bbox, spacing, dtype=None, interpolation=CUBIC):
        """Crop bbox from image.

        If a dtype is given, the returned values are divided by the maxintensity and scaled
        to the dtype.

        Args:
            bbox (BoundingBox): box with spacing box.spacing. If none, the whole image is cropped.
            spacing (number): spacing that the resulting crop should have.
            dtype (numpy dtype): integer dtype of the resulting crop.
            interpolation (string): interpolation method for eventual rescale of the crop.
        Returns:
            (np.ndarray): cropped image.
        """
        scale = self._get_available_scale(spacing)
        img = self._levels[list(self.scales).index(scale)]
        img_spacing = self.scale_to_spacing(scale)
        crop = Image.crop_from_array(img, bbox, img_spacing, spacing, interpolation, dtype)
        return crop



def crop_from_image(bbox, f, scale=1., prefix='/', interpolation=CUBIC, dtype=None):
    """Crop an array from an image using the bbox as border.

    Usage example:
    ```
    f = '/data/BDA/personal/hspitzer/braincollection/B01/B01_0316_Pyramid.hdf5'
    bbox = bb.BoundingBox((10000,10000), h=10000, w=10000)
    crop = im.crop_from_image(bbox, f, scale=1/32.)
    ```

    Args:
        bbox (BoundingBox): box on scale 1. If none, the whole image is cropped
        f (string): filename of the image or opened image (tiff, h5py, or numpy array)
        scale (float): scale of the crop relative to largest image available in the pyramid
        prefix (string): for hdf5 images: prefix to the image dataset or the pyramid
    """
    # if f is already opened tiff or hdf5, want to make use of the (possible) pyramid
    if isinstance(f, h5py.Dataset) or isinstance(f, h5py.Group):
        prefix = f.name
        fname = f.file.filename
        img = Hdf5Image(fname, prefix)
    elif isinstance(f, pytiff.Tiff):
        fname = f.filename
        img = TiffImage(fname)
    else:
        # f is string, or numpy array
        img = Image.get_image(f, prefix)

    spacing = img.scale_to_spacing(scale)
    crop = img.get_crop(bbox, spacing, interpolation=interpolation, dtype=dtype)
    return crop
