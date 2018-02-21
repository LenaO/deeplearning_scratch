from __future__ import print_function, division, absolute_import
import numpy as np
import math
from my_utils import get_logger

__all__ = ["BoundingBox"]

logger = get_logger(__name__)

class BoundingBox:

    def __init__(self, center, w, h, **kwargs):
        """A BoundingBox is a representation for a rectangle.

        It can be scaled, painted to an array and used to crop from an array.
        Internally, it knows a spacing, so if it is resized, the spacing is updated accordingly.

        Args:
            center (tuple): (x,y) coordinated of center.
            w (number): width of box
            h (number): height of box
            kwargs:
                spacing (float): spacing of the bounding box coordinates.
        """
        assert len(center) == 2
        self.center = (float(center[0]), float(center[1]))
        self.w = float(w)
        self.h = float(h)
        self.spacing = 1.
        if 'spacing' in kwargs:
            self.spacing = float(kwargs['spacing'])

    @classmethod
    def from_corners(cls, pt1, pt2, **kwargs):
        """Create BoundingBox defined by rectangle between pt1 and pt2.

        Args:
            pt1 (tuple): (x,y) coordinates of one corner.
            pt2 (tuple): (x,y) coordinates of the opposite corner.
            kwargs: see documentation for BoundingBox.__init__

        Returns:
            (BoundingBox instance)
        """
        assert len(pt1) == 2
        assert len(pt2) == 2
        xmin = min(pt1[0],pt2[0])
        ymin = min(pt1[1],pt2[1])
        w = abs(pt1[0]-pt2[0])
        h = abs(pt1[1]-pt2[1])
        x = xmin + w/2.
        y = ymin + h/2.
        return cls((x,y), w, h, **kwargs)

    @classmethod
    def from_contour(cls, contour, offset, **kwargs):
        """Create BoundingBox defined by rectangle between pt1 and pt2.

        Args:
            contour (array_like): (x,y) coordinates of the contour.
            offset (int): Offset added around the contour.
            kwargs: see documentation for BoundingBox.__init__

        Returns:
            (BoundingBox instance)
        """
        xmin,ymin = contour.min(0)-offset
        xmax,ymax = contour.max(0)+offset
        width = xmax-xmin
        height = ymax-ymin
        x = xmin + width/2.
        y = ymin + height/2.
        return cls((x,y), width, height, **kwargs)

    @classmethod
    def copy(cls, box):
        """Copy constructor for BoundingBox.

        Args:
            box (BoundingBox): box to copy
        Returns:
            (boundingBox instance)
        """
        new_box = cls(center=box.center, w=box.w, h=box.h, spacing=box.spacing)
        return new_box

    @classmethod
    def from_json(cls, bbox_dict):
        """Restore BoundingBox object from dictionary 
        
        Args:
            bbox_dict: dictionary containing keys 'center', 'h', 'w', 'spacing'
                (result of BoundingBox.to_json())
        Returns:
            (BoundingBox instance)
        """
        c = bbox_dict.get('center', [0,0])
        h = bbox_dict.get('h', 1)
        w = bbox_dict.get('w', 1)
        spacing = bbox_dict.get('spacing', 1.)
        return cls(center=c, w=w, h=h, spacing=spacing)


    def __str__(self):
        return 'BoundingBox - (center: {}, h: {}, w: {}, spacing: {})'.format(
            self.get_center(),
            self.get_hw()[0],
            self.get_hw()[1],
            self.spacing)

    def __repr__(self):
        return 'brainmap.image.BoundingBox({}, {}, {}, spacing={})'.format(
            self.center, self.h, self.w, self.spacing)

    def to_json(self):
        """Encode BoundingBox object as json dictionary
        Returns:
            dictionary containing keys 'center', 'h', 'w', 'spacing'
        """
        return {'center': self.center, 'h': self.h, 'w':self.w, 'spacing':self.spacing}

    # getter
    def get_center(self):
        """Get rounded down center coordinates.

        Returns:
            (tuple of ints): (x,y) coordinates of center
        """
        return (int(math.floor(self.center[0])), int(math.floor(self.center[1])))

    def get_hw(self, round_values=True):
        """Get rounded down height and width of the box.

        Return:
            (tuple of ints): (height, width)
        """
        if not round_values:
            return (self.h, self.w)
        return (int(math.floor(self.h)), int(math.floor(self.w)))

    @property
    def shape(self):
        return self.get_hw(round_values=False)

    def get_corners(self, round_values=True):
        """Get rounded down corners.

        Returns:
            (tuple): (lower-left, upper-left, lower-right, upper-right)
            where notation is according to a mathematical coordinate system (origin is bottom left,
            so "lower left" corner is nearest to origin and not upper left as in most images...).
            Each of these points is a list of length 2 containing x and y coordinates.
        """
        ll = np.array(self.center) + np.array((-self.w/2, -self.h/2))
        ul = np.array(self.center) + np.array((-self.w/2, self.h/2))
        lr = np.array(self.center) + np.array((self.w/2, -self.h/2))
        ur = np.array(self.center) + np.array((self.w/2, self.h/2))
        
        if not round_values:
            return [ll, lr, ul, ur]
        return list(map(lambda l:list(map(lambda el:int(math.floor(el)), l)), (ll, lr, ul, ur)))

    # setter and other BoundingBox modification methods (working inplace)
    def set_center(self, center):
        """Set the center of the bounding box

        Args:
            center (tuple): (x,y) new coordinated of center
        """
        self.center = (float(center[0]), float(center[1]))

    def set_hw(self, h, w):
        """Set height and width of box.
        """
        self.h = float(h)
        self.w = float(w)

    def rescale(self, scale):
        """Scale the box by the given scale (inplace).
        Adapts the spacing of the box

        Args:
            scale (float): scale factor
        """
        self.center = (self.center[0]*scale, self.center[1]*scale)
        self.h = self.h * scale
        self.w = self.w * scale
        self.spacing = self.spacing / scale

    def set_to_spacing(self, spacing):
        """Scale the box such that the defined spacing is reached.

        Args:
            spacing (number): wanted spacing
        """
        self.rescale(self.spacing/float(spacing))

    def pad(self, pad):
        """Pad the bounding box equally on all sides with pad (inplace).

        Args:
            pad: padding
        Returns:
            status (bool): False, if height or width of the resulting box are < 0
        """
        self.set_hw(self.h+2*pad, self.w+2*pad)
        return (self.h > 0 and self.w > 0)

    # methods computing stuff with bounding boxes
    def fits_in_shape(self, shape):
        """Check if bounding box fits in given shape.

        Args:
            shape (array_like): [height, width] Check if bounding box fits in this shape.

        Returns:
            bool: True if bounding box fits completely into shape, else False.
        """
        corners = np.array(self.get_corners())
        if (corners[:,0] < 0).any() or (corners[:,0] > shape[1]).any():
            return False
        if (corners[:,1] < 0).any() or (corners[:,1] > shape[0]).any():
            return False
        return True

    def contains_point(self, point):
        """Check if point is contained in the bounding box (assuming rounded down corner coordinates).

        Args:
            point (tuple): (x,y) Coordinates of the point.
        Returns:
            (bool): True if the point is inside the box, else False.
        """
        ll, _, _, ur = self.get_corners()
        if (ll[0] <= point[0]) and (ll[1] <= point[1]) and (ur[0] > point[0]) and (ur[1] > point[1]):
            return True
        return False

    def contains_box(self, box):
        """Check if the bounding box completely surrounds box
        Args:
            box (BoundingBox)
        Returns:
            (bool): True if box is inside self, else False
        """
        for pt in box.get_corners():
            if not self.contains_point(pt):
                return False
        return True

    def get_overlap_with(self, box):
        """Returns the overlap area between self and box.

        Args:
            box (BoundingBox): box to calculate the overlap with.
        Returns:
            BoundingBox containing the overlap or None, if there was no overlap
        """
        ll, lr, ul, ur = box.get_corners()
        bx0 = ll[0]
        bx1 = lr[0]
        by0 = ll[1]
        by1 = ul[1]

        ll, lr, ul, ur = self.get_corners()
        x0 = max(ll[0], bx0)
        x1 = min(lr[0], bx1)
        y0 = max(ll[1], by0)
        y1 = min(ul[1], by1)
        if x0 > x1 or y0 > y1:
            return None
        else:
            return BoundingBox.from_corners((x0, y0), (x1, y1))

    # cropping and writing methods
    def crop_from_array(self, arr):
        """Crop the box from an array.

        Args:
            arr (array-like): array from where the box should be cropped. Has to have at least two dimensions

        Returns:
            (array-like): crop with shape == self.get_hw()
        """
        ll, _, _, ur = self.get_corners()
        x0 = max(0, min(ll[0], arr.shape[1]))
        y0 = max(0, min(ll[1], arr.shape[0]))
        x1 = min(arr.shape[1], max(0, ur[0]))
        y1 = min(arr.shape[0], max(0, ur[1]))
        crop = arr[y0:y1,x0:x1]
        # ensure correct size of crop by padding overflows with 0
        pad_x0 = -1*min(0,ll[0])
        pad_y0 = -1*min(0,ll[1])
        pad_x1 = max(self.get_hw()[1]-crop.shape[1]-pad_x0,0)
        pad_y1 = max(self.get_hw()[0]-crop.shape[0]-pad_y0,0)
        if pad_x0 > 0 or pad_x1 > 0 or pad_y0 > 0 or pad_y1 > 0:
            pad = (pad_y0, pad_y1), (pad_x0, pad_x1)
            if len(crop.shape) > 2:
                # For arrays with more than two dimensions, add zero padding for additional dimensions
                pad += ((0, 0), ) * (len(crop.shape) - 2)

            crop = np.pad(crop, pad, 'constant')

        # if crop is now too large (due to rounding), cut at the correct size
        if crop.shape[0] > self.get_hw()[0]:
            crop = crop[:self.get_hw()[0]]
        if crop.shape[1] > self.get_hw()[1]:
            crop = crop[:,:self.get_hw()[1]]
        return crop

    def write_to_array(self, arr, crop):
        """Write given crop to arr using the bounding box as coordinates.

        Args:
            arr (array-like): array to which the crop should be written.
            crop (array-like): crop to write. Should have the same dimensions as the bbox.
        """
        assert len(arr.shape) == len(crop.shape), "dimension mismatch: arr {} - crop {}".format(arr.shape, crop.shape)
        ll, _, _, ur = self.get_corners()
        x0 = max(0,ll[0])
        y0 = max(0,ll[1])
        x1 = min(arr.shape[1], ur[0], crop.shape[1]+x0)
        y1 = min(arr.shape[0], ur[1], crop.shape[0]+y0)
        if (y1-y0,x1-x0) != crop.shape[:2]:
            logger.warn("write_to_array: Not writing the entire crop: {} vs {}".format(crop.shape, (y1-y0, x1-x0)))
        arr[y0:y1,x0:x1] = crop[0:y1-y0,0:x1-x0]

    #TODO: kann es sein, dass bei width=1 und fill = False nichts eingezeichnet wird?
    def draw_on_array(self, arr, color=0, width=1, fill=False):
        """Draw the box on array.

        Args:
            arr (array-like).
            color (optional): color to draw on the array, default 0.
            width (int, optional): width of painted box, if it is not filled. Default is 1.
            fill (bool, optional): if true, the box is filled. Default is false.

        Returns:
            (array-like): array with box drawn on top.
        """
        import skimage.draw
        if fill:
            # define polygon as corners of bounding box
            ll,lr,ul,ur=self.get_corners()
            y = np.array([ll[1], lr[1], ur[1], ul[1], ll[1]])
            x = np.array([ll[0], lr[0], ur[0], ul[0], ll[0]])
        else:
            # define polygon as corners of larger bounding box followed by corners of smaller bounding box.
            pad = width - 1

            box_large = BoundingBox.copy(self)
            box_large.set_hw(self.h+pad, self.w+pad)
            ll,lr,ul,ur=box_large.get_corners()

            box_small = BoundingBox.copy(self)
            box_small.set_hw(self.h-pad, self.w-pad)
            ll1,lr1,ul1,ur1=box_small.get_corners()
            y = np.array([ll[1], lr[1], ur[1], ul[1], ll[1], ll1[1], lr1[1], ur1[1], ul1[1], ll1[1], ll[1]])
            x = np.array([ll[0], lr[0], ur[0], ul[0], ll[0], ll1[0], lr1[0], ur1[0], ul1[0], ll1[0], ll[0]])

        # draw filled polygon on array
        coords = skimage.draw.polygon(y,x, shape=arr.shape)
        arr[coords] = color

        return arr

    def transform(self, brainid, section_from, section_to, spacing_out=None, maxsize=None, fixedsize=None, reg_version=None, trans_type="rigid"):
        """ Transforms bbox into another section of the same brain.

        Args:
            brainid (str): Brain ID where sections are from.
            section_from (int): Section ID of section where boundingbox comes from.
            section_to (int): Section ID to transform boundingbox to.
            spacing_out (array-like or float, optional): Spacing of resulting bounding box (default: spacing of input bbox).
            maxsize (array-like, optional): Maximal size of resulting bbox in x and y direction (has lower priority than fixedsize).
            fixedsize (array-like, optional): Fixed size of resulting bbox in x and y direction (has higher priority than maxsize).
            reg_version (str, optional): Version of preregistration (default: None).
            trans_type (str, optional): rigid or affine (default: rigid).

        Returns:
            new transformed BoundingBox object
        """
        assert type(section_from) == int
        assert type(section_to) == int
        from brainmap.transformation import transformation
        if spacing_out is None:
            spacing_out = self.spacing
        trans_list = transformation.TransformationList.from_gpfs(brainid, [min(section_from,section_to),max(section_from,section_to)], reg_version=reg_version,trans_type=trans_type)
        if section_from not in trans_list.transformations.keys():
            logger.warning("could not find transform parameter file for source section %i"%section_from)
            return None
        if section_to not in trans_list.transformations.keys():
            logger.warning("could not find transform parameter file for target section %i"%section_to)
            return None
        trans_list_to_base = trans_list.get_transformations_to_base(section_to)
        bb = trans_list_to_base[section_from].apply_to_bbox(self, spacing_out, maxsize=maxsize, fixedsize=fixedsize)
        return bb
