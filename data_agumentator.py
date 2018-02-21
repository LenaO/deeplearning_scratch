import math
import numpy as np
import skimage.transform
import scipy.ndimage
import numbers
# Try to import OpenCV and use scikit-image as a fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("data_augmentator: Module cv2 is not available, using scikit-image as a fallback")
    CV2_AVAILABLE = False

default_params = {
    'rotation': 0,
    'mirror': False,
    'shear_x': 0,
    'shear_y': 0,
    'deform': None,
    'gamma': 1
}


class DataAugmentator:
    def __init__(self, mirroring=False, rotation=False, shearing=False, shearing_range=None, deformation=False, deformation_field_size=None, deformation_grid_size=3, deformation_magnitude=10, gamma=False, gamma_magnitude=2):
        """ Calculate and apply random affine transformation and elastic deformation to images """
        if deformation:
            assert deformation_field_size is not None, "for calculating deformations, a deformation field size needs to be given!"
        if shearing_range is None:
            shearing_range = (-math.pi/8., math.pi/8.)

        for key, default_value in default_params.iteritems():
            setattr(self, key, default_value)
        if rotation:
            self.rotation = np.random.uniform(low=0, high=2*math.pi)
        if mirroring:
            self.mirror = np.random.choice([True, False])
        if shearing:
            self.shear_x = np.random.uniform(low=shearing_range[0], high=shearing_range[1])
            self.shear_y = np.random.uniform(low=shearing_range[0], high=shearing_range[1])
        if deformation:
            self.deform = DataAugmentator.get_random_deformation_field(deformation_field_size, deformation_grid_size, deformation_magnitude)
        if gamma:
            self.gamma = np.random.uniform(low=1./gamma_magnitude, high=gamma_magnitude)

    @classmethod
    def init_from_params(cls, params):
        """ Give rotation in radians """
        self = cls()
        for key, default_value in default_params.iteritems():
            setattr(self, key, params.get(key, default_value))
        return self

    def get_params(self):
        params = {}
        for key in default_params.keys():
            params[key] = getattr(self, key)
        return params

    @staticmethod
    def get_random_deformation_field(size, grid_size, sigma):
        ''' return deform_x, deform_y of shape (size, size)'''
        grid_displacement_x = np.random.normal(scale=sigma, size=(grid_size**2))
        grid_displacement_y = np.random.normal(scale=sigma, size=(grid_size**2))
        xx,yy = np.mgrid[0:size,0:size]

        deform_x = scipy.ndimage.zoom(grid_displacement_x.reshape((grid_size, grid_size)), zoom=size/float(grid_size), order=3)
        deform_y = scipy.ndimage.zoom(grid_displacement_y.reshape((grid_size, grid_size)), zoom=size/float(grid_size), order=3)
        return (deform_x, deform_y)

    def get_affine_transformation(self, shape):
        """ return affine transformation for images of shape shape """
        shift_y, shift_x = np.array(shape) / 2.
        #rotation and then shear
        matrix_shear_x = np.matrix([[1,math.tan(self.shear_x),0],[0,1,0],[0,0,1]])
        matrix_shear_y = np.matrix([[1,0,0],[math.tan(self.shear_y),1,0],[0,0,1]])
        matrix_rotation = np.matrix([[math.cos(self.rotation), math.sin(self.rotation), 0],[-math.sin(self.rotation), math.cos(self.rotation), 0],[0,0,1]])
        matrix_mirror = np.matrix([[1,0,0],[0,-1 if self.mirror else 1, 0],[0,0,1]])
        matrix = matrix_shear_y * matrix_shear_x * matrix_rotation * matrix_mirror
        transformation = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y]) + skimage.transform.AffineTransform(matrix=matrix) + skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        return transformation

    def apply(self, img, scale=1., order=1, graylevel_augmentation=True):
        """ apply deformation to img of shape (h, w).

        If img.shape * scale < deform.shape, the deformation field is cropped around the center
        to fit the image
        Args:
            img (array-like): shape (h,w)
            scale (float): factor by which to scale the deformation field before applying it """
        img = self.apply_affine_transformation(img, order=order)
        img = self.apply_elastic_deformation(img, scale=scale, order=order)
        if graylevel_augmentation:
            img = img ** self.gamma
        return img

    def apply_list(self, images, scales, order=1, graylevel_augmentation=None):
        """ apply deformation to list of images, with shape (channels, h, w),
        and with scales 'scales' """
        if isinstance(order, numbers.Number):
            order = [[order for i in range(len(images[0]))] for j in range(len(scales))]
        if graylevel_augmentation is None:
            graylevel_augmentation = [[True for i in range(len(images[0]))] for j in range(len(scales))]
        transformed_images = []
        for num in range(len(images)):
            img = images[num]
            scale = scales[num]
            img_transformed = np.zeros_like(img)
            for i,channel in enumerate(img):
                img_transformed[i] = self.apply(channel, scale, order=order[num][i], graylevel_augmentation=graylevel_augmentation[num][i])
            transformed_images.append(img_transformed)
        return transformed_images

    def apply_affine_transformation(self, img, order=1):
        transformation = self.get_affine_transformation(img.shape[:2])
        if CV2_AVAILABLE:
            interpolation_codes = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_CUBIC, 3: cv2.INTER_CUBIC}
            mat = cv2.invertAffineTransform(transformation.params[:2])
            transformed_image = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]), flags=interpolation_codes[order])
        else:
            transformed_image = skimage.transform.warp(img, transformation, preserve_range=True, order=order)
        return transformed_image

    def apply_elastic_deformation(self, img, scale, order=1):
        """ Apply self.deform to img.
        Scale deform by scale. If img is smaller than deform, deform is cropped around the center
        Args:
            img (arry like): input image
            scale (float): scale of the deformation field """
        if self.deform is None:
            return img

        assert img.shape[0] == img.shape[1], "non quadratic images not supported!"
        # scale deform according to scale
        deform_x, deform_y = map(lambda d: skimage.transform.rescale(d, scale, preserve_range=True, order=order), list(self.deform))
        assert deform_x.shape[0] >= img.shape[0] and deform_x.shape[1] >= img.shape[1], "deformation field must be >= than the image"

        # crop deform to fit img
        pad_l = int((deform_x.shape[0] - img.shape[0]) / 2.)
        pad_r = -1 * (deform_x.shape[0] - img.shape[0] - pad_l)
        if pad_r == 0:
            pad_r = None
        deform_x = deform_x[pad_l:pad_r,pad_l:pad_r]
        deform_y = deform_y[pad_l:pad_r,pad_l:pad_r]

        # coordinates to evaluate at: xx+deform_x and yy+deform_y -> these are the new values for xx and yy
        xx,yy = np.mgrid[0:img.shape[0],0:img.shape[1]]
        img_deformed = scipy.ndimage.map_coordinates(img, [xx+deform_x, yy+deform_y], mode='reflect', order=order)
        return img_deformed


class DeterministicDataAugmentator:
    """ for one crop, give list of crops that are the augmented versions.
        mirroring: crop and mirrored crop
        rotation: crop rotated by (0,90,180,270) degrees

        Used during prediction, when net was trained with augmented data """

    def __init__(self, mirroring=False, rotation=False):
        self.mirroring = mirroring
        self.rotation = rotation

    def augment_batch(self, batch):
        """ Apply deterministic data augmentation.
        Args:
            batch (array-like): shape (batch_size, channels, h, w)
        Returns:
            augmented_batch (list): list of augmented versions of batch """
        augmented_batch = [batch,]
        if self.mirroring:
            augmented_batch.append(self.mirror_batch(batch))
        if self.rotation:
            for k in (1,2,3):
                augmented_batch.append(self.rotate_batch(batch, k))
            if self.mirroring:
                for k in (1,2,3):
                    augmented_batch.append(self.rotate_batch(self.mirror_batch(batch), k))
        return augmented_batch

    def reduce_batch(self, augmented_batch):
        """ Reverses deterministic data augmentation and returns average of de-augmented batches
        Args:
            augmented_images (list): list of augmented versions of img of shape (h,w,*) """
        expected_len = (1 + self.mirroring) * (4 if self.rotation else 1)
        assert len(augmented_batch) == expected_len, "augmented_batch does not fit to current parameters"
        # reverse augment_batch
        res = []
        index = 0  # pointer to position in augmented_batch
        # the original image
        res.append(augmented_batch[index])
        index += 1
        # potentially mirrored image
        if self.mirroring:
            res.append(self.mirror_batch(augmented_batch[index]))
            index += 1
        if self.rotation:
            # rotated images
            for k in (3,2,1):
                res.append(self.rotate_batch(augmented_batch[index], k))
                index += 1
            if self.mirroring:
                # mirrored and rotated images
                for k in (3,2,1):
                    res.append(self.mirror_batch(self.rotate_batch(augmented_batch[index], k)))
                    index += 1
        res = np.array(res)
        res = res.mean(axis=0)
        return res

    def rotate_batch(self, batch, k):
        return np.transpose(np.rot90(np.transpose(batch, axes=(2,3,0,1)), k=k), axes=(2,3,0,1))

    def mirror_batch(self, batch):
        return np.transpose(np.flipud(np.transpose(batch, axes=(2,3,0,1))), axes=(2,3,0,1))


class DataTransformator:
    def __init__(self, rotation=0, mirror=False):
        """ Transform crop of shape (...,h,w).
        Args:
            rotation (float): rotation angle in radians.
            mirror (bool): mirror the crop
        """
        assert mirror is False, "mirror True is not implemented"
        self.rotation = -1*math.degrees(rotation)
        self.mirror = mirror

    def apply(self, crop, order=1):
        axes = (len(crop.shape)-2, len(crop.shape)-1)
        return scipy.ndimage.rotate(crop, self.rotation, axes=axes, order=order)

    def apply_list(self, crops, order=1):
        res = []
        for crop in crops:
            res.append(self.apply(crop, order))
        return res
