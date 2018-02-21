import os
import imp
import numpy as np
from os.path import basename, dirname, realpath, join

current_dirname = dirname(realpath(__file__))
parent_dirname = dirname(current_dirname)
grandparent_dirname = dirname(parent_dirname)


def set_path(paths):
    for path in paths:
        if os.path.exists(path):
            break
    if not os.path.exists(path):
        return ''
    else:
        return path


BIGBRAIN_PATH = set_path(['/gpfs/data/inm1/spitzer/bigbrain', '/judac/data/inm1/spitzer/bigbrain'])
print('Setting BIGBRAIN_PATH to {}'.format(BIGBRAIN_PATH))
EXPERIMENT_PATH = set_path(['/homeb/jinm14/jinm1403/experiments/2017_Semisupervised',
'/localdata/experiments/2017_Semisupervised', '/gpfs/data/inm1/spitzer/experiments/2017_Semisupervised'])
print('Setting EXPERIMENT_PATH to {}'.format(EXPERIMENT_PATH))

volume_path = os.path.join(BIGBRAIN_PATH, 'bigbrain_release2015/volumes')
registration_path = os.path.join(BIGBRAIN_PATH, 'bigbrain_release2015/registration')

section_file = os.path.join(BIGBRAIN_PATH, 'sections.json')
prob_file = os.path.join(BIGBRAIN_PATH, 'prob_{}.nii')
transformed_coords_file = os.path.join(BIGBRAIN_PATH, 'coords_transformed.hdf5')
gray_left_file = os.path.join(volume_path, 'mesh/gray_left_327680.giir')
gray_right_file = os.path.join(volume_path, 'mesh/gray_right_327680.giir')

mesh_civet_left_file = os.path.join(volume_path, 'mesh/gray_left_rsl_327680.obj')
mesh_civet_left_inflated_file = os.path.join(volume_path, 'mesh/white_left_inflated.obj')
mesh_civet_right_file = os.path.join(volume_path, 'mesh/gray_right_rsl_327680.obj')
mesh_civet_right_inflated_file = os.path.join(volume_path, 'mesh/white_right_inflated.obj')
geodesic_coord_system_file = os.path.join(volume_path, 'mesh/geodesic_coord_system_rsl.txt')

image_path = os.path.join(BIGBRAIN_PATH, 'images', 'B20_{}.tif')
meta_path = os.path.join(BIGBRAIN_PATH, 'meta', 'B20_{}_meta.hdf5')

boxes_file_path = os.path.join(EXPERIMENT_PATH, 'samples/{0}')

# transformations
trans_to_volume_file = os.path.join(registration_path, 'to_volume/pm{}.xfm')
trans_to_mri_file = os.path.join(registration_path, 'to_mri/pm{}.xfm')
trans_to_repaired_file = os.path.join(registration_path, 'to_repaired/results/B20_{}_nlin.xfm')

# nifti volumes
areas = ['hOc1','hOc2','hOc3d','hOc3v','hOc4d','hOc4la','hOc4lp','hOc4v','hOc5','FG1','FG2','FG3','FG4']
areas_to_mpm = [249, 252, 199, 108, 143, 162, 233, 103, 105, 148, 171, 117, 123]
vol_gray_file = os.path.join(volume_path, 'full8_200um_optbal.nii.gz')
vol_mask_file = os.path.join(volume_path,'classif_200um.nii.gz')
vol_laplace_file = os.path.join(volume_path,'laplace_200um.nii.gz')
vol_bok_file = os.path.join(volume_path,'bok_200um_v2.nii.gz')
vol_atlas_file = os.path.join(volume_path,'pmaps/Anatomy_v22c_TO_BigBrain.nii')
vol_pmaps_files = [os.path.join(volume_path,'pmaps/Visual_{}_TO_BigBrain.nii').format(area) for area in areas]

areas_radius = [26.070, 20.835, 19.148, 19.220, 16.319, 19.713, 19.068, 18.014, 7.425, 12.659, 17.530, 15.358, 18.501]

# used for distance normalization. Any values below min/above max will be cut off
min_distance = 0
max_distance = 220

sides = ['left', 'right']
labels = ['geo_dist', '2d_dist', '3d_dist', 'direction']

# mean calculated over test set (training set)
mean = 0.527
std = 0.183
dirname = os.path.dirname(os.path.realpath(__file__))
labels = ['bg', 'V1', 'V2', 'V3a', 'V3d', 'V4la', 'V4lp', 'V4v', 'V5', 'Vp', 'fg3', 'fg4', 'wm', 'fg1', 'fg2', 'cor']

train_params = {
    # --- General Training ---
#    'iterations': 15000,  # 35 epochs
    'iterations': 1500,  # 35 epochs
    'stop_criterium': 'max_global',
    'mode': 'regression',
    # --- Optimizer ---
    'solver': 'adam',
    'learning_rate':0.001,
    'lr_policy': 'constant',
    'momentum': 0.9,
    'momentum2': 0.999,
    # --- Loss and Metrics ---
    'weight_decay':0,
    'loss_name': 'linear',
    'huber_delta': 30,
    'switch_c1': 5.,
    'switch_c2': 3.,
    'switch_c3': 220.,
    'switch_value': 30,
    'metrics_names': ['inv_linear'],
    # --- Testing ---
    'test_iter':16,
    'test_interval':200,
    # --- Display and Save ---
    'display':5,
    'log_interval':1,
    'snapshot':1000,
    'show_activation': False,
    'log_dir': os.path.join(current_dirname, '../../logs', basename(parent_dirname), basename(current_dirname))
}

data_params = {
    'batch_size':32,
    'mode': 'regression',
    'split': 'train',
    'labels': ['geo_dist'],
    'boxes_file': boxes_file_path.format('train_samples_power_40000.sqlite'),
    'deterministic': True,
    'mask_background': False,
    'input_size': [1019, 1019],
    'input_spacing': [2, 2],
    'input_channels': [('gray',), ('gray',)],  # options gray, cellseg

    'mean': [(mean,),(mean,)],  # [(0.5,),(0.5,)],
    'std': [(std,),(std,)],  # [(0.5,),(0.5,)],

    'laplace_rotation': True
}

test_params = {
    'batch_size': 32,
}

mpi_params = {
    'train_producer_ranks': range(0,13),
    'test_producer_ranks': range(13,17),
    'train_receiver_rank': 17,
    'test_receiver_rank': 18,
    'trainer_ranks': [19],

    'predict_producer_ranks': range(0,18),
    'predict_receiver_rank': 18,
    'predicter_ranks': [19]
}

net_definition = ['net_definition.py']

label_colors_dict = {}
label_colors_dict['V1']   = 'FFFF00' #Yellow
label_colors_dict['V2']   = '0000FF' #Blue
label_colors_dict['V3a']  = '800000' #Maroon 
label_colors_dict['V3d']  = 'FF0000' #Red
label_colors_dict['V4la'] = '008000' #Green
label_colors_dict['V4lp'] = '00FF00' #Lime
label_colors_dict['V4v']  = '808000' #Olive
label_colors_dict['V5']   = '800080' #Purple
label_colors_dict['Vp']   = 'FF00FF' #Fuchsia
label_colors_dict['fg3']  = '00FFFF' #Aqua
label_colors_dict['fg4']  = '000080' #Navy
label_colors_dict['bg']   = '000000' #Black
label_colors_dict['wm']   = 'FFFFFF' #White
label_colors_dict['fg1'] = 'B0C4DE' # light steel blue
label_colors_dict['fg2'] = '7FFFD4' # aquamarine
label_colors_dict['cor'] = 'D3D3D3' #light gray -- unknown cortex
label_colors_dict['CORTEX'] = 'D3D3D3' #light gray

# convert HEX colors to float values in [0,1]
for key, value in label_colors_dict.iteritems():
    label_colors_dict[key] = tuple(float(int(value[i:i + 2], 16))/255 for i in
                                   range(0, 6, 2))

# Correspondence between label and label index
label_name_to_number = {'bg':   0, 
                        'V1':   1,
                        'V2':   2,
                        'V3a':  3, 
                        'V3d':  4,
                        'V4la': 5,
                        'V4lp': 6,
                        'V4v':  7,
                        'V5':   8,
                        'Vp':   9,
                        'fg3':  10,
                        'fg4':  11,
                        'wm':   12,
                        'fg1':  13,
                        'fg2':  14,
                        'cor':  15}
label_number_to_name = {value: key for key, value in
                        label_name_to_number.iteritems()}

def get_label_number(label_name):
    if label_name in label_name_to_number:
        return label_name_to_number[label_name]
    return None

def get_label_name(label_number):
    if label_number in label_number_to_name:
        return label_number_to_name[label_number]
    return None

def get_labels_array():
    return sorted(label_name_to_number.keys(), key=label_name_to_number.get)

def get_colors_array():
    return [label_colors_dict[name] for name in get_labels_array()]

# Display labels names
display_label_name = ['bg', 'hOc1', 'hOc2', 'hOc4d', 'hOc3d', 'hOc4la',
                      'hOc4lp', 'hOc4v', 'hOc5', 'hOc3v', 'FG4', 'FG3',
                      'wm', 'FG1', 'FG2', 'cor']

def get_display_label_name(label_name):
    """ label name: either string of number"""
    if isinstance(label_name, str):
        label_name = get_label_number(label_name)
    return display_label_name[label_name]



