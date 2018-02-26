from __future__ import print_function
import numpy as np
import h5py
import glob
import os
from collections import deque
from mpi4py import MPI
import numbers
from image import NEAREST, CUBIC
from bounding_box import BoundingBox

from test_params import section_file, prob_file, boxes_file_path, image_path, meta_path
from boxesdb import BoxesDB, SeedBoxesDB
import json
import math

from copy import deepcopy
import types
import shutil
from point_sampler import PointSampler
from image import Image
import data_agumentator as da
import test_params as test
from my_utils import get_logger


class file_cache:

    def __init__(self, name, path = "./"):
        self.name = name
        self.path = os.path.join(path, "."+self.name)
        self._dict = {}
        print(self.path)
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except OSError as e:
                 if e.errno != errno.EEXIST:
                    raise

    def write_np_array(self, key, data):
        new_path = os.path.join(self.path, str(key))
        #pyextrae.eventandcounters(6000,5)
        np.save(new_path, data)
        #pyextrae.eventandcounters(6000, 0)

    def read_np_array(self, key):
        key = str(key)+".npy"
        new_path = os.path.join(self.path, key)
        if not os.path.exists(new_path):
            return None
        else:
            pyextrae.eventandcounters(6000, 10)
            crop = np.load(new_path)
            pyextrae.eventandcounters(6000, 0)
            return crop
    def has(self, key):
        key = str(key)+".npy"
        new_path = os.path.join(self.path, key)
        return  os.path.exists(new_path)

    def keys(self):
        return self._dict.keys()

    def __getitem__(self, index):
        if index in  self._dict.keys():
            return self._dict[index]
        return self.read_np_array(index)

    def __setitem__(self, index, value):
        if isinstance(value, (np.ndarray, np.generic)):
            self.write_np_array(index, value)
        elif isinstance(value, dict):
            new_name = str( index)
            self._dict[index]=file_cache(new_name, self.path)

    def cleanup(self):
        shutil.rmtree(self.path)


batch_iterator_params = {
    # --------------------------------------------------------------------------------------------------------------
    # General parameters
    # --------------------------------------------------------------------------------------------------------------
    "batch_size": 10,  # Batch size
    "mode": "regression",  # mode of operation (for siameseBI should always be regession)
    "labels": ["geo_dist"],  # labels to train on. For siameseBI this can be distance and direction (defined in const.labels)
    "section_file": section_file,  # dictionary of available section numbers and meta info (quality, split)
    "include_labels": ["good", "excellent"],
    "split": "train",  # Defines the mode to work on. Can be "train" or "test"
    "probability_volume": prob_file.format("train"),
    "boxes_file": boxes_file_path.format("test_samples_500.sqlite"),
    "mask_background": False,  # Mask out the background in the input crops
    "deterministic": True,  # always output the same order of crops (does not work with augmentation)

    "coord_type": "inflated",  # coordinates to train on ("inflated", "inflated_left" (map right coords to left))
    # For PointSampler
    "point_sampler_sample_size": 1000,  # number of points that are sampled at once by PointSampler. Higher sample sizes are more efficient.
    "approximate_distance": True,  # if the calculated distance between points should only be approximated
    "approximate_distance_level": 1,  # higher numbers indicate more precise approximate distances
    "use_threads_for_distance": True,
    "num_threads_for_distance": 1,

    # --------------------------------------------------------------------------------------------------------------
    # Parameters defining input channels
    # --------------------------------------------------------------------------------------------------------------
    "input_size": [2019, 2019],  # Size of network input (h=w) !!! size is given relative to input_spacing !!!
    "input_spacing": [2, 2],  # Spacing of the input in um/px
    "input_channels": [("gray",), ("gray",)],  # List of input channels
    "input_channel_params": {  # Parameters for each input channel
        "gray": {  # Parameters for grayscale input channel
            "fname": image_path,  # Filename
            "dataset": "/",
            "normalize": True,  # Normalize grayscale values
            "mean": True, "std": True,  # Perform mean standardization
            "graylevel_augmentation": False,  # Augment grayscale values
            "mask_background": True,  # Mask out background
            "interpolation": CUBIC,  # interpolation strategy when resizing
        },
        "bok": {
            'fname': meta_path,
            'dataset': '/bok',
            'normalize': True,
            'mean': False, 'std': False,
            'graylevel_augmentation': False,
            'mask_background': True,
            'interpolation': NEAREST,
        },
        'laplace': {
            'fname': meta_path,
            'dataset': '/laplace',
            'normalize': True,
            'mean': False, 'std': False,
            'graylevel_augmentation': False,
            'mask_background': False,
            'interpolation': CUBIC,
        },
        "mask": {
            "fname": meta_path,
            "dataset": "/mask",
            "normalize": False,
            "mean": False, 'std': False,
            "mask_background": False,
            "interpolation": NEAREST
        },
    },  # TODO what about pmaps inputs??

    # --------------------------------------------------------------------------------------------------------------
    # Parameters for data augmentation
    # --------------------------------------------------------------------------------------------------------------
    "mean": None,  # Mean for mean standardisation
    "std": None,  # Standard deviation for standardisation
    "random_mirroring": False,  # Randomly mirror crops
    "random_rotation": False,  # Randomly rotate crops
    "random_shearing": False,  # Randomly shear crops
    "random_shearing_range": (-math.pi / 8., math.pi / 8.),  # Determines range in which can be sheared
    "random_elastic_deformation": False,  # Randomly deform crops
    "random_elastic_deformation_grid_size": 3,  # Gridsize for deformation
    "random_elastic_deformation_magnitude": 10,  # Extend of deformation
    "random_gamma": False,  # Manipulate gamma values
    "random_gamma_magnitude": 2,  # Extend of gamma manipulation
    "laplace_rotation": False,  # Do Laplace rotation

    # --------------------------------------------------------------------------------------------------------------
    # Default values for non-relevant parameters (for BraicollectionBatchIterator)
    # --------------------------------------------------------------------------------------------------------------
    "slices": {},
    "boxes_folder": None,
    "default_label": None,
    "label_size": 1,
    "label_spacing": 1
}

class BasicBatchIterator(object):

    """ Base class for Batch Iterators.
    New Iterators should inherit from this class
    """
    def __init__(self, params):
        self.log = get_logger(self.__class__.__name__, rank=MPI.COMM_WORLD.Get_rank())
        # set params
        for key, default_value in batch_iterator_params.iteritems():
            setattr(self, key, params.get(key, default_value))
            self.log.info("Setting {} to {:.200}".format(key, '{}'.format(getattr(self, key))))

        # deal with different means: numbers or arrays
        # put in list with length num_inputs, because for each input a different mean is possible
        if self.mean is not None:
            if isinstance(self.mean, numbers.Number) or isinstance(self.mean, basestring):
                self.mean = [(self.mean,),]
            self.mean = [tuple([np.load(m) if isinstance(m, basestring) else m for m in ms]) for ms in self.mean]
            self.log.info('Uses mean')
        if self.std is not None:
            if isinstance(self.std, numbers.Number) or isinstance(self.std, basestring):
                self.std = [(self.std,),]
            self.std = [tuple([np.load(m) if isinstance(m, basestring) else m for m in ms]) for ms in self.std]
            self.log.info('Uses std')

        # put input_size and input_spacing in array:
        if isinstance(self.input_size, numbers.Number):
            self.input_size = [self.input_size,]
        if isinstance(self.input_spacing, numbers.Number):
            self.input_scale = [self.input_spacing,]
        if isinstance(self.input_channels, tuple):
            self.input_channels = [self.input_channels,]

        # set num_inputs
        self.num_inputs = len(self.input_size)

        # calculate size of boxes (sqrt(3) larger than biggest size when augmenting data
        factor = 1.
        if self.random_rotation or self.laplace_rotation:
            factor = math.sqrt(2)
        if self.random_shearing and (self.random_rotation or self.laplace_rotation):
            factor = math.sqrt(3)
        if self.random_shearing:  # TODO not quite sure what should be here...
            factor = math.sqrt(3)
        self.padded_input_size = map(lambda s: int(math.ceil(s*factor)), self.input_size)
        self.padded_label_size = int(math.ceil(self.label_size*factor))
        self.log.info('Setting padded_input_size to {}'.format(self.padded_input_size))
        self.log.info('Setting padded_label_size to {}'.format(self.padded_label_size))

    def get_datum(self, **args):
        """ Returns one data point.
        This should already be preprocessed and transformed, i.e. mean substraction done and label in correct format

        Returns:
            dict with keys:
                inputs: list of inputs for the net (length self.num_crops),
                label: either number, or array (in case of segmentation), or a dictionary containing the bbox
                (in case of GridBatchIterator)
        """
        raise NotImplementedError

    def generate_batch(self):
        labels = []
        inputs = [[] for i in range(self.num_inputs)]
        for i in range(self.batch_size):
            try:
                datum = self.get_datum()
            except StopIteration as e:
                break

            # go through all the inputs and append to respective lists
            for num in range(self.num_inputs):
                inputs[num].append(datum['inputs'][num])
            labels.append(datum['label'])

        if len(labels) == 0:
            raise StopIteration
        if isinstance(labels[0], numbers.Number) or isinstance(labels[0], np.ndarray):

            self.log.debug('LABELS are numbers - converting them to np.array')
            try:
                labels = np.array(labels, dtype='int32')
            except Exception as e:
                self.log.error('Exception ocurred:')
                self.log.error(e)
                self.log.error('labels {}'.format(labels))
                self.log.error('len(labels) {}'.format(len(labels)))
                self.log.error(' '.join([str(l.shape) for l in labels]))
                return self.generate_batch()
        try:
            inputs = [np.array(batch).astype('float32') for batch in inputs]
        except Exception as e:
            #sometimes exceptions happen here because of nan values in datum?? still debugging...
            self.log.error('Exception ocurred:')
            self.log.error(e)
            self.log.error('len(inputs): {}'.format(len(inputs)))
            self.log.error('len(inputs[0]) {}'.format(len(inputs[0])))
            self.log.error(' '.join([str(b.shape) for b in inputs[0]]))
            self.log.error('inputs[0][0].shape {}'.format(inputs[0][0].shape))
            # solution: just try again
            return self.generate_batch()

        return (inputs, labels)

    def __iter__(self):
        return self

    def next(self):
        batch = self.generate_batch()
        return batch

    def stop(self):
        pass

    # -- Formatting functions
    def format_label_array(self, label_arr, label_list=None, default_label='bg', cortex_segmentation=None):
        """ Takes label_arr with labels coded according to label_list and transforms them
        according to self.labels.
        default_label determines the fallback label if in label_arr some label is present
        that is not in self.labels. """
        assert self.mode == 'segmentation', 'this is only needed for segmentation mode'
        assert default_label in self.labels, 'default_label is not in self.labels'
        if label_list is None:
            label_list = test.get_labels_array()
        return utils.format_label_array(label_arr, label_list, self.labels, default_label, cortex_segmentation, dtype=np.int32)

    def get_crop(self, box, channel, spacing, size, fname=None, dset=None, interpolation=None):
        """ Used by all function that want to crop something from a specific channel. Can be overwritten, if the data comes from somewhere else (Hdf5Iter).
        Size given on spacing spcaing"""
        #print('get_crop', box, channel, spacing, size)
        if channel == 'CONST':
            # return dummy data
            return np.zeros((size, size), dtype='uint8')
        if fname is None:
            fname = self.input_channel_params[channel]['fname']
        if dset is None:
            dset = self.input_channel_params[channel]['dataset']
        if interpolation is None:
            interpolation = self.input_channel_params[channel]['interpolation']
        bbox = BoundingBox.copy(box['bbox'])
        bbox.set_to_spacing(spacing)
        bbox.set_hw(size, size)

        # TODO is this necessary? - yes, for int16 images this is necessary!
        dtype = None
        if channel == 'gray':
            dtype = 'uint8'
        crop = Image.get_image(fname, prefix=dset).get_crop(bbox, spacing=spacing, dtype=dtype, interpolation=interpolation)
        return crop

    def get_crops_for_box(self, box):
        if isinstance(box, dict):
            box = [box for _ in range(self.num_inputs)]
        # get the crops in different scales and sizes
        crops = [[] for i in range(self.num_inputs)]
        for num in range(self.num_inputs):
            sp = self.input_spacing[num]
            si = self.padded_input_size[num]
            for input_channel in self.input_channels[num]:
                crop = self.get_crop(box[num], input_channel, spacing=sp, size=si)
                crops[num].append(crop)
        crops = [np.array(c) for c in crops]
        return crops

    def get_laplace_rotation_angle(self, box):
        # size of laplace box is 25x25 on spacing 16, because want to rotate crop according to direction of cortex in center of image
        dx = self.get_crop(box, 'laplace', spacing=16, size=25, dset='/original/dx')
        dy = self.get_crop(box, 'laplace', spacing=16, size=25, dset='/original/dy')
        rotation = np.arctan2(dy.mean(), dx.mean())
        # want reverse angle to "derotate" the crop
        return -1*rotation

    # -- Data preparation functions
    def normalize_crops(self, crops):
        """ Applies mean and std to crops.
        Crops are list of crops corresponding to the different inputs in the net.
        Each crop has shape (channels, height, width)"""
        res = []
        for num, crop in enumerate(crops):
            res_crop = []
            for i, channel in enumerate(crop):
                channel = channel.astype('float32')
                if self.input_channel_params[self.input_channels[num][i]]['normalize']:
                    channel = (channel / 255.)
                res_crop.append(channel)

            res.append(np.array(res_crop, dtype='float32'))
        return res

    def mean_substraction(self, crops):
        """ Applies mean and std to crops.
        Crops are list of crops corresponding to the different inputs in the net.
        Each crop has shape (channels, height, width)"""
        res = []
        for num, crop in enumerate(crops):
            res_crop = []
            for i, channel in enumerate(crop):
                if self.input_channel_params[self.input_channels[num][i]]['mean'] and self.mean is not None:
                    channel = channel - self.mean[num][i]
                if self.input_channel_params[self.input_channels[num][i]]['std'] and self.std is not None:
                    channel = channel / self.std[num][i]
                res_crop.append(channel)

            res.append(np.array(res_crop, dtype='float32'))
        return res

    def get_data_augmentation_params(self, crops):
        scales_for_da = [min(self.input_spacing)/float(s) for s in self.input_spacing]
        size_for_da = max([crop.shape[1]/sc for crop, sc in zip(crops, scales_for_da)])
        augmentator = da.DataAugmentator(
            mirroring=self.random_mirroring,
            rotation=self.random_rotation,
            shearing=self.random_shearing,
            shearing_range=self.random_shearing_range,
            deformation=self.random_elastic_deformation,
            deformation_field_size=size_for_da,
            deformation_grid_size=self.random_elastic_deformation_grid_size,
            deformation_magnitude=self.random_elastic_deformation_magnitude,
            gamma=self.random_gamma,
            gamma_magnitude=self.random_gamma_magnitude)
        return augmentator.get_params()

    def apply_data_augmentation_to_crops(self, params, crops):
        """ crops should have values between o and 1 (mean_substraction AFTER data_augmentation)"""
        augmentator = da.DataAugmentator.init_from_params(params)
        scales_for_da = [min(self.input_spacing)/float(s) for s in self.input_spacing]
        graylevel_augmentation = [[self.input_channel_params[channel]['graylevel_augmentation'] for channel in self.input_channels[num]] for num in range(self.num_inputs)]
        order = [[self.input_channel_params[channel]['interpolation'] for channel in self.input_channels[num]] for num in range(self.num_inputs)]
        crops = augmentator.apply_list(crops, scales_for_da, order=order, graylevel_augmentation=graylevel_augmentation)
        return crops

    def apply_data_augmentation_to_label(self, params, label):
        augmentator = da.DataAugmentator.init_from_params(params)
        label = augmentator.apply(label, min(self.input_spacing)/self.label_spacing, order=0, graylevel_augmentation=False)
        return label

    def unpad_crops(self,crops):
        """ crops has shape (channels, h, w) """
        res_crop = []
        for size, crop in zip(self.input_size, crops):
            pad_r = -1 * int((crop.shape[1] - size) / 2.)
            pad_l = int(crop.shape[1] - size + pad_r)
            if pad_r == 0:
                pad_r = None
            res_crop.append(crop[:,pad_l:pad_r,pad_l:pad_r])
        return res_crop

    def unpad_label(self, label):
        """ label has shape (h,w) """
        size = self.label_size
        pad_r = -1 * int((label.shape[0] - size) / 2.)
        pad_l = int(label.shape[0] - size + pad_r)
        if pad_r == 0:
            pad_r = None
        res_label = label[pad_l:pad_r, pad_l:pad_r]
        return res_label

    def apply_mask_background(self, crops, box):
        # apply after unpadding!
        if not self.mask_background:
            return crops
        if isinstance(box, dict):
            box = [box for _ in range(self.num_inputs)]

        # get background for the different scales and sizes
        for num in range(self.num_inputs):
            sp = self.input_spacing[num]
            si = self.padded_input_size[num]
            bg = self.get_crop(box[num], 'bg_wm_cor', spacing=sp, size=si)
            for i, ch in enumerate(self.input_channels[num]):
                if self.input_channel_params[ch]['mask_background']:
                    val = 0
                    if self.input_channel_params[self.input_channels[num][i]]['mean'] and self.mean is not None:
                        val = self.mean[num][i]
                    crop = crops[num][i]
                    crop[bg != 1] = val
                    crops[num][i] = crop
        return crops


READY, START, DONE, EXIT, DATA, LABELS, GETCACHE, PUTCACHE, CACHE = 0,1,2,3,4,5,6,7,8

class MPIBatchProducer:
    """
    Caching behaviour:
        The last producer will not produce, but hold the cache instead (cacher_rank).
        Other producers will ask the cacher_rank for a crop (batch_iter.get_crop method gets replaced)
        and only read from disk, if the cacher_rank doesnt have this crop. In that case, the new crop
        will also be sent to the cacher_rank for further keeping.
    """
    def __init__(self, batch_iter, receiver_rank, comm, caching=False, max_cache_size=90000, io_cachpath=None):
        self.log = get_logger(self.__class__.__name__, rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing for sending to process {}'.format(receiver_rank))
        self.comm = comm
        self.rank = self.comm.Get_rank()
        assert self.rank != receiver_rank, "Cannot send and receive on the same process {} {} {}".format(batch_iter.split, self.rank, receiver_rank)

        self.batch_iter = batch_iter
        self.receiver_rank = receiver_rank
        self.caching = caching
        self.max_cache_size = max_cache_size


        self.io_cachpath = io_cachpath
        if self.caching:
            # only works for padded input sizes that are the same for all inputs
            assert batch_iter.padded_input_size[0] == batch_iter.padded_input_size[1]
            # only works for TRAIN mode of batch iter
            assert self.batch_iter.sampling_mode == TRAIN
            # producer ranks start after trainer ranks. Last rank is receiver rank
            # cacher is last producer = second to last rank

            assert self.io_cachpath == None
            self.cacher_rank = self.comm.Get_size() - 2
            self.is_cacher = self.rank == self.cacher_rank
            self.num_producers = self.comm.Get_size() - 3  # NOTE assumes one Trainer! Not stable!
            if self.is_cacher:
                self.log.info('This rank is the cacher')
                self.cache = {}
            else:
                self.log.info('This rank is a producer (of {} total producers)'.format(self.num_producers))
                outer_self = self
                old_get_crop = self.batch_iter.get_crop

                # replace get crop method of batch iter with method requesting crop from cacher
                def get_crop(self, box, channel, spacing, size, fname=None, dset=None, interpolation=None):
                    self.log.debug('Requesting sample {} from cacher {}'.format(box['sample_num'], outer_self.cacher_rank))
                    outer_self.comm.send((box['sample_num'], channel, spacing, size, fname, dset, interpolation), dest=outer_self.cacher_rank, tag=GETCACHE)
                    has_crop = outer_self.comm.recv(None, source=outer_self.cacher_rank, tag=GETCACHE)
                    if has_crop == 1:
                        # receive crop
                        self.log.debug('Getting sample {} from cacher {}'.format(box['sample_num'], outer_self.cacher_rank))
                        crop = np.empty((self.padded_input_size[0], self.padded_input_size[0]), dtype='uint8')
                        outer_self.comm.Recv([crop, MPI.UINT8_T], source=outer_self.cacher_rank, tag=GETCACHE)
                    else:
                        # sample crop and send to cacher_rank before returning
                        crop = old_get_crop(box, channel, spacing, size, fname, dset, interpolation)
                        if has_crop == 2:
                            self.log.debug('Not sending sample {} to cacher {}, because cache is full'.format(box['sample_num'], outer_self.cacher_rank))
                        else:
                            self.log.debug('Sending sample {} to cacher {}'.format(box['sample_num'], outer_self.cacher_rank))
                            # sending to cacher rank with info about key
                            outer_self.comm.send((box['sample_num'], channel, spacing, size, fname, dset, interpolation), dest=outer_self.cacher_rank, tag=PUTCACHE)
                            outer_self.comm.Send([crop, MPI.UINT8_T], dest=outer_self.cacher_rank, tag=PUTCACHE)
                    return crop
                self.batch_iter.get_crop = types.MethodType(get_crop, self.batch_iter)

        elif self.io_cachpath is not None:
            assert self.batch_iter.sampling_mode == TRAIN
            # producer ranks start after trainer ranks. Last rank is receiver rank
            # cacher is last producer = second to last rank
            assert self.caching == False
            self.cache = file_cache(name="image_cache",  path=self.io_cachpath)
            outer_self = self
            old_get_crop = self.batch_iter.get_crop
            def get_crop(self, box, channel, spacing, size, fname=None, dset=None, interpolation=None):
                if channel not in outer_self.cache.keys(): 
                    outer_self.cache[channel] = {}
                if str(dset) not in outer_self.cache[channel].keys():
                    outer_self.cache[channel][str(dset)] = {}
                if  not outer_self.cache[channel][str(dset)].has( box['sample_num']) :
                    self.log.debug('Caching sample {} for channel {}, dset {}'.format(box['sample_num'], channel, dset))
                    crop =old_get_crop(box, channel, spacing, size, fname, dset, interpolation) 
                    outer_self.cache[channel][str(dset)][box['sample_num']] = crop
                    return crop
                else:
                    crop = outer_self.cache[channel][str(dset)][box['sample_num']]
                    while crop is None:
                        crop = outer_self.cache[channel][str(dset)][box['sample_num']]
                    return crop

            self.batch_iter.get_crop = types.MethodType(get_crop, self.batch_iter)


    def produce(self):
        status = MPI.Status()
        if self.caching and self.is_cacher:
            # receive requests for crops from other producers and answer them
            self.log.info('Started caching')
            exited_producers = 0
            while True:
                if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=EXIT):
                    self.comm.recv(None, source=MPI.ANY_SOURCE, tag=EXIT, status=status)
                    exited_producers += 1
                    self.log.debug('Got EXIT from producer {} (have {} of {})'.format(status.Get_source(), exited_producers, self.num_producers))
                    if exited_producers == self.num_producers:
                        self.cleanup()
                        break
                if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=GETCACHE):
                    self.log.debug('Requests from producers is available')
                    key = self.comm.recv(source=MPI.ANY_SOURCE, tag=GETCACHE, status=status)
                    source = status.Get_source()
                    self.log.debug('Received request for key {} from {}'.format(key, source))
                    if key in self.cache:
                        self.log.debug('Found key')
                        self.comm.send(1, dest=source, tag=GETCACHE)
                        self.log.debug('Sending crop to {}'.format(source))
                        self.comm.Send([self.cache[key], MPI.UINT8_T], dest=source, tag=GETCACHE)
                    else:
                        if len(self.cache.keys()) >= self.max_cache_size:
                            self.log.debug('Did not find key, but cache is full')
                            self.comm.send(2, dest=source, tag=GETCACHE)
                        else:
                            self.log.debug('Did not find key')
                            self.comm.send(0, dest=source, tag=GETCACHE)
                if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=PUTCACHE):
                    self.log.debug('Data from producers is available')
                    key = self.comm.recv(source=MPI.ANY_SOURCE, tag=PUTCACHE, status=status)
                    source = status.Get_source()
                    self.log.debug('Receiving crop from {}'.format(source))
                    crop = np.empty((self.batch_iter.padded_input_size[0], self.batch_iter.padded_input_size[0]), dtype='uint8')
                    self.comm.Recv([crop, MPI.UINT8_T], source=source, tag=PUTCACHE)
                    self.cache[key] = crop
                    if len(self.cache.keys()) % 100 == 0:
                        self.log.info('Cache size is {}, max_cache_size: {}'.format(len(self.cache.keys()), self.max_cache_size))

            self.log.info('Stopped caching')
        else:
            self.log.info('Started sending to receiver {}.'.format(self.receiver_rank))
            status = MPI.Status()
            while True:
                try:
                    crops, labels = self.batch_iter.next()
                except StopIteration:
                    # no more batches, send stopping message to receiver
                    self.log.debug('Sending DONE to {}'.format(self.receiver_rank))
                    self.comm.send(None, dest=self.receiver_rank, tag=DONE)
                    break
                # has batch, send mgs to receiver that is ready to start tansmission
                self.log.debug('Sending READY to receiver {}'.format(self.receiver_rank))
                self.comm.send(None, dest=self.receiver_rank, tag=READY)
                self.comm.recv(None, source=self.receiver_rank, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                self.log.debug('Got {} from receiver {}'.format('START' if tag == START else 'EXIT', self.receiver_rank))
                if tag == START:
                    # go ahead with sending batch
                    for crop in crops:
                        self.log.debug('Sending data to receiver {}'.format(self.receiver_rank))
                        self.comm.Send([crop, MPI.FLOAT], dest=self.receiver_rank, tag=DATA)
                    self.log.debug('Sending labels to receiver {}'.format(self.receiver_rank))
                    self.comm.send(labels, dest=self.receiver_rank, tag=LABELS)
                elif tag == EXIT:
                    if self.caching:
                        # need to tell cacher that this process is exiting
                        self.log.debug('Sending EXIT to cacher {}'.format(self.cacher_rank))
                        self.comm.send(None, dest=self.cacher_rank, tag=EXIT)
                    break
            self.log.info('Stopped sending to receiver {}'.format(self.receiver_rank))

    def cleanup(self):
        if self.io_cachpath is not None:
            self.cache.cleanup()

# ------------------ Batch Receiver ----------------------------
class MPIBatchReceiver:
    """Gather batches from produces processes and give them to MPIBatchIterator"""
    def __init__(self, batch_shape, producer_ranks, iterator_ranks, comm, caching=False):
        self.log = get_logger(self.__class__.__name__, rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing for shape {}'.format(batch_shape))

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.N = self.comm.Get_size()
        if caching:
            # ignore last producer rank - he is the cacher!
            self.producer_ranks = producer_ranks[:-1]
        else:
            self.producer_ranks = producer_ranks
        self.iterator_ranks = iterator_ranks
        self.batch_shape = batch_shape
        self.maxlen = 10

        self.alive_producers = self.producer_ranks
        self.alive_iterators = self.iterator_ranks
        self.wants_data = []
        self.has_data = deque()
        self.stop_producers = False
        self.stop_iterators = False
        self.queue = {i:deque() for i in self.alive_iterators}

    def receive(self):
        self.log.info('Started. Receiving from {}, Sending to {}'.format(self.producer_ranks, self.iterator_ranks))
        status = MPI.Status()
        while True:

            # get new message
            self.comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            source = status.Get_source()
            if source in self.producer_ranks:
                self.log.debug('Got {} from producer {}'.format('DONE' if tag == DONE else 'READY', source))
                if tag == READY:
                    self.has_data.append(source)
                elif tag == DONE:
                    self.alive_producers.remove(source)

            if source in self.iterator_ranks:
                self.log.debug('Got {} from iterator {}'.format('READY' if tag == READY else 'EXIT', source))
                if tag == READY:
                    self.wants_data.append(source)
                if tag == EXIT:
                    self.alive_iterators.remove(source)
                    # update queue accordingly
                    del self.queue[source]

            if not self.check_continue():
                break

            # stop pending iterators and producers
            if self.stop_producers:
                while len(self.has_data) > 0:
                    source = self.has_data.popleft()
                    self.send_exit(source)
                    self.alive_producers.remove(source)
            if self.stop_iterators:
                for source in self.wants_data[:]:
                    self.send_done(source)
                    self.alive_iterators.remove(source)
                    # update queue accordingly
                    del self.queue[source]
                    self.wants_data.remove(source)

            if not self.check_continue():
                break

            # receive from producers if possible
            while self.can_receive() and len(self.has_data) > 0:
                source = self.has_data.popleft()
                batch = self.receive_batch(source)
                self.put_queue(batch)

            # send to iterators if possible
            for source in self.wants_data[:]:
                if self.can_send(source):
                    batch = self.pop_queue(source)
                    self.send_batch(source, batch)
                    self.wants_data.remove(source)

        self.log.info('Stopped')

    def send_exit(self, source):
        self.log.debug('Sending EXIT to producer {}'.format(source))
        self.comm.send(None, dest=source, tag=EXIT)

    def send_done(self, source):
        self.log.debug('Sending DONE to iterator {}'.format(source))
        self.comm.send(None, dest=source, tag=DONE)

    def receive_batch(self, source):
        # receive batch from producer
        self.log.debug('Sending START to producer {}'.format(source))
        self.comm.send(None, dest=source, tag=START)
        crops = []
        for batch_shape in self.batch_shape:
            crop = np.empty(batch_shape, dtype='float32')
            self.comm.Recv([crop, MPI.FLOAT], source=source, tag=DATA)
            self.log.debug('Got data from producer {}'.format(source))
            crops.append(crop)
        labels = self.comm.recv(source=source, tag=LABELS)
        self.log.debug('Got labels from producer {}'.format(source))
        return (crops, labels)

    def send_batch(self, source, batch):
        crops, labels = batch
        # send data to iterator
        self.log.debug('Sending START to iterator {}'.format(source))
        self.comm.send(None, dest=source, tag=START)
        for crop in crops:
            self.log.debug('Sending data to iterator {}'.format(source))
            self.comm.Send([crop, MPI.FLOAT], dest=source, tag=DATA)
        self.log.debug('Sending labels to iterator {}'.format(source))
        self.comm.send(labels, dest=source, tag=LABELS)

    def put_queue(self, batch):
        for i in self.queue:
            self.queue[i].append(batch)
            self.log.debug('Putting batch in queue {} (len {})'.format(i, len(self.queue[i])))

    def pop_queue(self, iterator):
        return self.queue[iterator].popleft()

    def can_receive(self):
        # check if there is enough space in queue
        for i in self.queue:
            if len(self.queue[i]) >= self.maxlen:
                self.log.debug('Queue ist full. Receiving not possible')
                return False
        return True

    def can_send(self, iterator):
        # check if there is an element in the queue to send
        if len(self.queue[iterator]) > 0:
            return True
        else:
            self.log.debug('Queue for {} is empty. Sending is not possible'.format(iterator))
            return False

    def check_continue(self):
        if len(self.alive_producers) == 0 and sum([len(self.queue[i]) for i in self.queue]) == 0:
            #only set stop_iterators to true when queue is empty (before stopping need to send everything)
            self.stop_iterators = True
        if len(self.alive_iterators) == 0:
            self.stop_producers = True

        # update queue according to alive iterators
        #for i in self.queue.keys()[:]:
        #    if i not in self.alive_iterators:
        #        del self.queue[i]

        if len(self.alive_iterators) == 0 and len(self.alive_producers) == 0:
            return False
        else:
            return True


# ------------------ Batch Iterator ----------------------------
class MPIBatchIterator:
    """Receive batches from producer process"""

    def __init__(self, batch_shape, receiver_rank, comm):
        self.log = get_logger(self.__class__.__name__, rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing for shape {}'.format(batch_shape))

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.N = self.comm.Get_size()
        self.receiver_rank = receiver_rank

        self.batch_shape = batch_shape
        self.batch_size = batch_shape[0][0]
        self.stopped = False

    def __iter__(self):
        return self

    def next(self):
        status = MPI.Status()
        if not self.stopped:
            self.log.debug('Sending READY to receiver {}'.format(self.receiver_rank))
            self.comm.send(None, dest=self.receiver_rank, tag=READY)
            self.comm.recv(None, source=self.receiver_rank, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            self.log.debug('Got {} from receiver {}'.format('DONE' if tag == DONE else 'START', self.receiver_rank))
            if tag == DONE:
                raise StopIteration
            elif tag == START:
                crops = []
                for batch_shape in self.batch_shape:
                    crop = np.empty(batch_shape, dtype='float32')
                    self.comm.Recv([crop, MPI.FLOAT], source=self.receiver_rank, tag=DATA)
                    self.log.debug('Got data from receiver {}'.format(self.receiver_rank))
                    crops.append(crop)
                labels = self.comm.recv(source=self.receiver_rank, tag=LABELS)
                self.log.debug('Got labels from receiver {}'.format(self.receiver_rank))
                return (crops, labels)
        else:
            raise StopIteration

    def __enter__(self):
        return self

    def stop(self):
        if self.stopped is False:
            self.stopped = True
            self.log.debug('Sending EXIT to receiver {}'.format(self.receiver_rank))
            self.comm.send(None, dest=self.receiver_rank, tag=EXIT)
            self.log.info('Stopped')


    def __exit__(self, type, value, tb):
        self.stop()


# remap badly named constant
TRAIN=0
TEST=1
PREDICT=TEST
# params specific for siamesebatchiterator
siamese_batch_iter_params = ["section_file", "include_labels", "probability_volume", "deterministic",
                             "point_sampler_sample_size", "approximate_distance", "approximate_distance_level",
                             "use_threads_for_distance", "num_threads_for_distance", "coord_type"]


class SiameseBatchIterator(BasicBatchIterator):
    """Provide random pairs of boxes and their geodesic distance.
    Operates on bigbrain"""

    # -------------------------------- Init --------------------------------------------
    def __init__(self, params, sampling_mode=TRAIN):
        """ Init SiameseBatchIterator.
        Args:
            sampling_mode (int): determines the type of batches that are produced.
                If TRAIN, the returned batches will consist of (batch, label), with label a list of distances
                between the two crops. Distances are always real (not NAN)
                If TEST, the batches will consist of (batch, box), with box a list of dictionaries describing
                the bounding box used for creating the batch. Distances could be NAN
            deterministic (bool): ensure that the sequence of crops is well-defined.
        """
        # update params with default parameters
        for key, default_value in batch_iterator_params.iteritems():
            params[key] = params.get(key, default_value)
        assert len(params['input_size']) == 2, "SiameseBatchIterator needs two inputs"

        # init super class
        super(SiameseBatchIterator, self).__init__(params)

        # SiameseBatchIterator-specific initialization
        self.log = get_logger('SiameseBatchIterator', rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing.')
        for key in siamese_batch_iter_params:
            setattr(self, key, params.get(key, batch_iterator_params[key]))
            self.log.info('Setting {} to {}'.format(key, getattr(self, key)))
        self.log.info('Setting sampling_mode to {}'.format(sampling_mode))
        self.sampling_mode = sampling_mode
        self.get_box = None  # set in specific init_for_X initializers
        self.point_sampler = None  # only set for random sampling
        self.ps_labels = [l for l in self.labels if l != 'coord']

    @classmethod
    def init_for_random_sampling(cls, params, sampling_mode=TRAIN):
        self = cls(params, sampling_mode)

        point_sampler_params = {
            'sample_size': self.point_sampler_sample_size,
            'section_file': self.section_file,
            'include_labels': self.include_labels,
            'split': self.split,
            'deterministic': self.deterministic,
            'allow_nan_distance': sampling_mode != TRAIN,
            'approximate_gdist': self.approximate_distance,
            'subdivision_level': self.approximate_distance_level,
            'threaded': self.use_threads_for_distance,
            'num_threads': self.num_threads_for_distance,
            'labels': self.ps_labels,
        }
        self.point_sampler = PointSampler(self.probability_volume, point_sampler_params)

        def get_box():
            point1, point2, labels = self.point_sampler.next()
            box = []
            for pt in (point1, point2):
                bbox = BoundingBox(pt['transformed_point'], 10, 10, spacing=pt['spacing']*1000)
                box.append({'bbox':bbox, 'side':pt['side'], 'slice_no': pt['slice_no'], '3d_coords':pt['point']})

            return box, labels
        self.get_box = get_box
        return self

    @classmethod
    def init_for_boxes_file(cls, params, sampling_mode=TRAIN, num_processes=1, rank=0, is_eval_boxes_file=False, seed_num=range(0,10), side=0):
        self = cls(params, sampling_mode)
        self.side = side
        self.log.info('Rank {} of {}: Using boxes_file {}'.format(rank, num_processes, self.boxes_file))
        if not is_eval_boxes_file:
            points1, points2, labels = BoxesDB(self.boxes_file).get_all_pairs(self.ps_labels)
        else:
            seeds, points, labels = SeedBoxesDB(self.boxes_file).get_all_pairs(self.ps_labels, side=side)
            seeds = [seed for seed in seeds if seed['seed_num'] in seed_num]
            points1 = [seed for seed in seeds for _ in range(len(points))]
            labels = labels.transpose(1,0,2).reshape(-1,labels.shape[2])
            points2 = points * len(seeds)
        if 'coord' in self.labels:
            self.log.info('Getting inflated coords for points with %s'%self.coord_type)
            coords1, coords2 = ut.get_inflated_coords(points1, points2, self.coord_type)
        self.log.info('Iterates over {} box pairs'.format(len(points1)))

        boxes = []
        for i in range(len(points1)):
            if i % num_processes != rank:
                continue
            point1 = points1[i]
            point2 = points2[i]
            label = labels[i]
          #  print("my box is {} {} {}".format(point1, point2, label))
            box = []
            for pt in (point1, point2):
                bbox = BoundingBox(pt['transformed_point'], 10, 10, spacing=pt['spacing']*1000)
                box.append({'bbox':bbox, 'side':pt['side'], 'slice_no':pt['slice_no'], '3d_coords':pt['point'], 'mesh_point':pt['mesh_point']})
            if is_eval_boxes_file:
                box[0]['seed_num'] = point1['seed_num']
                box[0]['area'] = point1['area']
                box[1]['sample_num'] = point2['sample_num']
            else:
                box[0]['sample_num'] = point1['sample_num']
                box[1]['sample_num'] = point2['sample_num']
            if 'coord' in self.labels:
                final_label = list(label) + [[list(coords1[i]), list(coords2[i])]]
            else:
                final_label = label  # TODO test is inflated coords are calculated and displayed correctly in boxes!!!
            boxes.append((box, final_label))
        if not self.deterministic:
            self.log.info('Shuffling the boxes')
            np.random.shuffle(boxes)
        boxes = deque(boxes)

        def get_box():
            if len(boxes) == 0:
                raise StopIteration
            box = boxes.popleft()
            if sampling_mode == TRAIN:
                boxes.append(box)
            return box
        self.get_box = get_box
        return self

    # -------------------------------- Helper --------------------------------------------

    # -------- Overrides of BasicBatchIterator because of other file structure ------------
    def get_laplace_rotation_angle(self, box):
        # size of laplace box is 25x25 on spacing 64, because want to rotate crop according to direction of cortex in center of image
        # NOTE input of np.gradient NEEDS to be signed! Otherwise gradients are wrong! BEWARE
        dy, dx = np.gradient(self.get_crop(box, 'laplace', spacing=64, size=25).astype(np.int8))
        rotation = np.arctan2(dy.mean(), dx.mean())
        # want reverse angle to "derotate" the crop
        return -1*rotation

    def apply_mask_background(self, crops, box):
        # apply after unpadding!
        if not self.mask_background:
            return crops
        if isinstance(box, dict):
            box = [box for _ in range(self.num_inputs)]

        # get background for the different scales and sizes
        for num in range(self.num_inputs):
            sp = self.input_spacing[num]
            si = self.padded_input_size[num]
            bg = self.get_crop(box[num], 'mask', spacing=sp, size=si)
            for i, ch in enumerate(self.input_channels[num]):
                if self.input_channel_params[ch]['mask_background']:
                    val = 0
                    if self.input_channel_params[self.input_channels[num][i]]['mean'] and self.mean is not None:
                        val = self.mean[num][i]
                    crop = crops[num][i]
                    crop[bg != 127] = val
                    crops[num][i] = crop
        return crops

    # ------------------------------ SAMPLE CROPS ------------------------------------------
    def get_datum(self, **args):
        box, label = self.get_box()
        if self.sampling_mode == TRAIN:
            while np.isnan(label[0]):
                self.log.debug('Distance is NAN, sampling new point')
                box, label = self.get_box()

        #pyextrae.eventandcounters(6666, 15)
        crops = self.get_crops_for_box(box)
        #pyextrae.eventandcounters(6666, 0)
        data_augmentation_params = []
        for i in range(len(crops)):
            params = deepcopy(da.default_params)
            params.update(self.get_data_augmentation_params([crops[i],]))
            if self.laplace_rotation:
                params['rotation'] = self.get_laplace_rotation_angle(box[i])
            data_augmentation_params.append(params)

        crops = self.normalize_crops(crops)  # normalizes only the wanted channels (set with input_channel_params)
        crops = self.apply_mask_background(crops, box)  # mask out background if desired
        crops = [self.apply_data_augmentation_to_crops(data_augmentation_params[num], [crops[num]])[0] for num in range(len(crops))]
        crops = self.mean_substraction(crops)
        crops = self.unpad_crops(crops)
        # TODO take care of deterministic flag - need deterministic data augmentation params

        # get the label
        if self.sampling_mode == TRAIN:
            # label exists, and contains desired values
            pass
        elif self.sampling_mode == PREDICT:
            for b, params in zip(box, data_augmentation_params):
                b.update(params)
                for i, name in enumerate(self.labels):
                    b[name] = label[i]
            label = box
        return {'inputs': crops, 'label': label}

    def get_crop(self, box, channel, spacing, size, fname=None, dset=None, interpolation=None):
        # replace placeholders in input channel with correct value
        if fname is None:
            fname = self.input_channel_params[channel]['fname'].format(box['slice_no'])
        if interpolation is None:
            interpolation = self.input_channel_params[channel]['interpolation']
        return super(SiameseBatchIterator, self).get_crop(box, channel, spacing, size, fname=fname, dset=dset, interpolation=interpolation)


class SingleBatchIterator(SiameseBatchIterator):
    def __init__(self, params, num_processes=1, rank=0, is_eval_boxes_file=True):
        """ Init SingleBatchIterator.
        Samples single crops from SeedBoxesDB boxes_file.
        Always in sampling_mode PREDICT.
        """
        # update params with default parameters
        for key, default_value in batch_iterator_params.iteritems():
            params[key] = params.get(key, default_value)
        # init super class
        super(SiameseBatchIterator, self).__init__(params)

        # SiameseBatchIterator-specific initialization
        self.log = get_logger('SingleBatchIterator', rank=MPI.COMM_WORLD.Get_rank())
        self.log.info('Initializing.')
        for key in siamese_batch_iter_params:
            setattr(self, key, params.get(key, batch_iterator_params[key]))
            self.log.info('Setting {} to {}'.format(key, getattr(self, key)))
        self.log.info('Setting samping_mode to {}'.format(PREDICT))
        self.sampling_mode = PREDICT
        self.log.info('Rank {} of {}: Using boxes_file {}'.format(rank, num_processes, self.boxes_file))
        self.ps_labels = [l for l in self.labels if l != 'coord']
        all_points = []
        if is_eval_boxes_file:
            BoxesClass = SeedBoxesDB
        else:
            BoxesClass = BoxesDB
        for side in (0,1):
            seeds, points, _ = BoxesClass(self.boxes_file).get_all_pairs(self.ps_labels, side=side)
            all_points.extend(seeds)
            all_points.extend(points)
        self.log.info('Iterates over {} points'.format(len(all_points)))
        boxes = []
        for i,pt in enumerate(all_points):
            if i % num_processes != rank:
                continue
            bbox = BoundingBox(pt['transformed_point'], 10, 10, spacing=pt['spacing']*1000)
            box = {'bbox':bbox, 'side':pt['side'], 'slice_no':pt['slice_no'], '3d_coords':pt['point'], 'mesh_point':pt['mesh_point']}
            if 'seed_num' in pt:
                box['seed_num'] = pt['seed_num']
                box['area'] = pt['area']
            else:
                box['sample_num'] = pt['sample_num']
            boxes.append(box)
        boxes = deque(boxes)

        def get_box():
            if len(boxes) == 0:
                raise StopIteration
            box = boxes.popleft()
            return box
        self.get_box = get_box

    def get_datum(self, **args):
        box = self.get_box()

        #pyextrae.eventandcounters(6666, 20)
        crops = self.get_crops_for_box(box)
       # pyextrae.eventandcounters(6666, 0)
        data_augmentation_params = deepcopy(da.default_params )
        data_augmentation_params.update(self.get_data_augmentation_params(crops))
        if self.laplace_rotation:
            data_augmentation_params['rotation'] = self.get_laplace_rotation_angle(box)

        crops = self.normalize_crops(crops)  # normalizes only the wanted channels (set with input_channel_params)
        crops = self.apply_mask_background(crops, box)  # mask out background if desired
        crops = self.apply_data_augmentation_to_crops(data_augmentation_params, crops)
        crops = self.mean_substraction(crops)
        crops = self.unpad_crops(crops)
        # TODO take care of deterministic flag - need deterministic data augmentation params
        return {'inputs': crops, 'label': box}

def get_train_batch_iter(num_processes=1, rank=0):
    train_batch_iter = SiameseBatchIterator.init_for_boxes_file(test.data_params, sampling_mode=TRAIN, num_processes=num_processes, rank=rank)
    return train_batch_iter


def get_test_batch_iter(num_processes=1, rank=0):
    params = test.data_params.copy()
    params['split'] = 'test'  # not really needed since we are sampling from boxes file
    params['boxes_file'] = test.boxes_file_path.format('test_samples_500.sqlite')
    params['deterministic'] = True
    test_batch_iter = SiameseBatchIterator.init_for_boxes_file(params, sampling_mode=TRAIN, num_processes=num_processes, rank=rank)
    return test_batch_iter


def get_predict_batch_iter(boxes_file, num_processes=1, rank=0):
    params = test.data_params.copy()
    params['split'] = 'test'  # not really needed since we are sampling from boxes file
    params['boxes_file'] = test.boxes_file_path.format(boxes_file)
    params['deterministic'] = True
    params['batch_size'] = test.test_params['batch_size']
    params['input_size'] = params['input_size'][:1]
    params['input_spacing'] = params['input_spacing'][:1]
    params['input_channels'] = params['input_channels'][:1]
    predict_batch_iter = SingleBatchIterator(params, num_processes=num_processes, rank=rank, is_eval_boxes_file=True)
    return predict_batch_iter



def get_grid_batch_iter(split, num_processes=1, rank=0):
    from collections import namedtuple
    params = deepcopy(test.data_params)
    if params['mode'] == 'segmentation':
        network = imp.load_source('net_def', os.path.join(os.path.dirname(__file__),net_definition[0]))
        Config = namedtuple('config', 'train_params data_params test_params')
        config = Config(test.train_params, test.data_params, test.test_params)
        net = network.build_net(input_shape=utils.get_batch_shape(config, 'test')) 
        params['label_size'] = net.outputs[0].shape.as_list()[1]
    params['split'] = split
    params['batch_size'] = test.test_params['batch_size']
    params['input_size'] = test.test_params['input_size']
    
    batch_iter = BraincollectionBatchIterator.init_for_grid_sampling(params, TEST, offset=test.test_params['offset'], label_in_box=False, deterministic=True, num_processes=num_processes, rank=rank)
    return batch_iter

def get_predict_slice_batch_iter(brain_no, slice_no, num_processes=1, rank=0):
    from collections import namedtuple
    
    params = test.data_params.copy()
    if params['mode'] == 'segmentation':
        network = imp.load_source('net_def', os.path.join(os.path.dirname(__file__),net_definition[0]))
        Config = namedtuple('config', 'train_params data_params test_params')
        config = Config(test.train_params, test.data_params, test.test_params)
        net = network.build_net(input_shape=utils.get_batch_shape(config, 'test')) 
        params['label_size'] = net.outputs[0].shape.as_list()[1]
    params['split'] = None
    params['batch_size'] = test.test_params['batch_size']
    params['input_size'] = test.test_params['input_size']
    
    batch_iter = init_for_grid_sampling(params, TEST, offset=test.test_params['offset'], section=(brain_no, slice_no), label_in_box=False, num_processes=num_processes, rank=rank)
    return batch_iter
