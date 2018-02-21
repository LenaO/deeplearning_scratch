import math
from image import NEAREST, CUBIC
import os 

EXPERIMENT_PATH='/work/jinm14/jinm1403/experiments/2016_SegmentationPostISBI'

BRAINCOLLECTION_PATH = '/gpfs/homeb/pcp0/pcp0063/brain-atlas/braincollection'


# --- Paths ---
image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/{0}_{1}_Pyramid.hdf5')
cellsegmented_image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/cellsegmentation/segmented_{0}_{1}_Pyramid.hdf5')
pmap_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/pmaps/pmaps_{0}_{1}.hdf5')
bias_corrected_image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/bias_correction/{0}_{1}_Pyramid.hdf5')
gli_image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/cellsegmentation/gli/{0}_{1}_Pyramid.hdf5')
laplace_image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/laplace/laplace_{0}_{1}_Pyramid.hdf5')
bg_segmented_image_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/masks/bgsegmentation/bg_cor_wm/BgWmSegmentation_{0}_{1}_Pyramid.hdf5')
euclidean_distance_path = os.path.join(BRAINCOLLECTION_PATH, '{0}/masks/squared_euclidean_distance/distance_{0}_{1}.hdf5')

boxes_file_path = os.path.join(BRAINCOLLECTION_PATH, 'crops/{0}')
braincollection_db_path = os.path.join(BRAINCOLLECTION_PATH, 'braincollection.sqlite')

# --- Visual System Labels ---
# TODO this needs to be redone: should also look at format_label_array function!
# Colors for the labels 
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



