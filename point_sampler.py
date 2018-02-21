from __future__ import print_function, division
import nibabel as nib
import trimesh
from my_utils import get_logger
import numpy as np
import gdist
import my_utils as ut
from collections import deque
import h5py
import networkx as nx
import test_params as test
import nibabel as nb
import numpy as np


# TODO incorporate in other functions as well
def write_ply_with_color(fname, coords, colors):
    from plyfile import PlyData, PlyElement
    vertex = np.array([tuple(list(vert) + list(col[:3])) for vert, col in zip(coords, colors)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(fname)


# function to load mesh geometry
def load_mesh_geometry(surf_mesh):
    # if input is a filename, try to load it with nibabel
    if isinstance(surf_mesh, basestring):
        if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                surf_mesh.endswith('inflated')):
            coords, faces = nb.freesurfer.io.read_geometry(surf_mesh)
        elif surf_mesh.endswith('gii'):
            coords, faces = nb.gifti.read(surf_mesh).getArraysFromIntent(nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
                            nb.gifti.read(surf_mesh).getArraysFromIntent(nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
        elif surf_mesh.endswith('vtk'):
            coords, faces, _ = read_vtk(surf_mesh)
        elif surf_mesh.endswith('ply'):
            coords, faces = read_ply(surf_mesh)
        elif surf_mesh.endswith('obj'):
            coords, faces = read_obj(surf_mesh)
        elif isinstance(surf_mesh, dict):
            if ('faces' in surf_mesh and 'coords' in surf_mesh):
                coords, faces = surf_mesh['coords'], surf_mesh['faces']
            else:
                raise ValueError('If surf_mesh is given as a dictionary it must '
                                 'contain items with keys "coords" and "faces"')
        else:
            raise ValueError('surf_mesh must be a either filename or a dictionary '
                             'containing items with keys "coords" and "faces"')
    return {'coords':coords,'faces':faces}


# function to load mesh data
def load_mesh_data(surf_data, gii_darray=0):
    # if the input is a filename, load it
    if isinstance(surf_data, basestring):
        if (surf_data.endswith('nii') or surf_data.endswith('nii.gz') or
                surf_data.endswith('mgz')):
            data = np.squeeze(nb.load(surf_data).get_data())
        elif (surf_data.endswith('curv') or surf_data.endswith('sulc') or
                surf_data.endswith('thickness')):
            data = nb.freesurfer.io.read_morph_data(surf_data)
        elif surf_data.endswith('annot'):
            data = nb.freesurfer.io.read_annot(surf_data)[0]
        elif surf_data.endswith('label'):
            data = nb.freesurfer.io.read_label(surf_data)
        # check if this works with multiple indices (if dim(data)>1)
        elif surf_data.endswith('gii'):
            fulldata = nb.gifti.giftiio.read(surf_data)
            n_vectors = len(fulldata.darrays)
            if n_vectors == 1:
                data = fulldata.darrays[gii_darray].data
            else:
                print("Multiple data files found, output will be matrix")
                data = np.zeros([len(fulldata.darrays[gii_darray].data), n_vectors])
                for gii_darray in range(n_vectors):
                    data[:,gii_darray] = fulldata.darrays[gii_darray].data
        elif surf_data.endswith('vtk'):
            _, _, data = read_vtk(surf_data)
        elif surf_data.endswith('txt'):
            data=np.loadtxt(surf_data)
        else:
            raise ValueError('Format of data file not recognized.')
    elif isinstance(surf_data, np.ndarray):
        data = np.squeeze(surf_data)
    return data


## function to write mesh data
def save_mesh_data(fname, surf_data):
    if isinstance(fname, basestring) and isinstance(surf_data,np.ndarray):
        if (fname.endswith('curv') or fname.endswith('thickness') or
                fname.endswith('sulc')):
            nb.freesurfer.io.write_morph_data(fname,surf_data)
        elif fname.endswith('txt'):
            np.savetxt(fname,surf_data)
        elif fname.endswith('vtk'):
            if 'data' in surf_dict.keys():
                write_vtk(fname,surf_dict['coords'],surf_dict['faces'],surf_dict['data'])
            else:
                write_vtk(fname,surf_dict['coords'],surf_dict['faces'])
        elif fname.endswith('gii'):
            print('please write lovely write gifti command')
        elif fname.endswith('mgh'):
            print('please write lovely write mgh command, or retry saving as .curv file')
    else:
        raise ValueError('fname must be a filename and surf_data must be a numpy array')


# function to read vtk files
# ideally use pyvtk, but it didn't work for our data, look into why
def read_vtk(file):
    '''
    Reads ASCII coded vtk files using pandas,
    returning vertices, faces and data as three numpy arrays.
    '''
    import pandas as pd
    import csv
    # read full file while dropping empty lines
    try:
        vtk_df=pd.read_csv(file, header=None, engine='python')
    except csv.Error:
        raise ValueError('This vtk file appears to be binary coded currently only ASCII coded vtk files can be read')
    vtk_df=vtk_df.dropna()
    # extract number of vertices and faces
    number_vertices=int(vtk_df[vtk_df[0].str.contains('POINTS')][0].iloc[0].split()[1])
    number_faces=int(vtk_df[vtk_df[0].str.contains('POLYGONS')][0].iloc[0].split()[1])
    # read vertices into df and array
    start_vertices= (vtk_df[vtk_df[0].str.contains('POINTS')].index.tolist()[0])+1
    vertex_df=pd.read_csv(file, skiprows=range(start_vertices), nrows=number_vertices, sep='\s*', header=None, engine='python')
    if np.array(vertex_df).shape[1]==3:
        vertex_array=np.array(vertex_df)
    # sometimes the vtk format is weird with 9 indices per line, then it has to be reshaped
    elif np.array(vertex_df).shape[1]==9:
        vertex_df=pd.read_csv(file, skiprows=range(start_vertices), nrows=int(number_vertices/3)+1, sep='\s*', header=None, engine='python')
        vertex_array=np.array(vertex_df.iloc[0:1,0:3])
        vertex_array=np.append(vertex_array, vertex_df.iloc[0:1,3:6], axis=0)
        vertex_array=np.append(vertex_array, vertex_df.iloc[0:1,6:9], axis=0)
        for row in range(1,(int(number_vertices/3)+1)):
            for col in [0,3,6]:
                vertex_array=np.append(vertex_array, np.array(vertex_df.iloc[row:(row+1),col:(col+3)]),axis=0)
        # strip rows containing nans
        vertex_array=vertex_array[ ~np.isnan(vertex_array) ].reshape(number_vertices,3)
    else:
        print ("vertex indices out of shape")
    # read faces into df and array
    start_faces= (vtk_df[vtk_df[0].str.contains('POLYGONS')].index.tolist()[0])+1
    face_df=pd.read_csv(file, skiprows=range(start_faces), nrows=number_faces, sep='\s*', header=None, engine='python')
    face_array=np.array(face_df.iloc[:,1:4])
    # read data into df and array if exists
    if vtk_df[vtk_df[0].str.contains('POINT_DATA')].index.tolist()!=[]:
        start_data=(vtk_df[vtk_df[0].str.contains('POINT_DATA')].index.tolist()[0])+3
        number_data = number_vertices
        data_df=pd.read_csv(file, skiprows=range(start_data), nrows=number_data, sep='\s*', header=None, engine='python')
        data_array=np.array(data_df)
    else:
        data_array = np.empty(0)

    return vertex_array, face_array, data_array


# function to read ASCII coded ply file
def read_ply(file):
    import pandas as pd
    import csv
    # read full file and drop empty lines
    try:
        ply_df = pd.read_csv(file, header=None, engine='python')
    except csv.Error:
        return read_ply_binary(file)
        #raise ValueError('This ply file appears to be binary coded currently only ASCII coded ply files can be read')
    ply_df = ply_df.dropna()
    # extract number of vertices and faces, and row that marks the end of header
    number_vertices = int(ply_df[ply_df[0].str.contains('element vertex')][0].iloc[0].split()[2])
    number_faces = int(ply_df[ply_df[0].str.contains('element face')][0].iloc[0].split()[2])
    end_header = ply_df[ply_df[0].str.contains('end_header')].index.tolist()[0]
    # read vertex coordinates into dict
    vertex_df = pd.read_csv(file, skiprows=range(end_header + 1),
                            nrows=number_vertices, sep='\s*', header=None,
                            engine='python')
    vertex_array = np.array(vertex_df)
    # read face indices into dict
    face_df = pd.read_csv(file, skiprows=range(end_header + number_vertices + 1),
                          nrows=number_faces, sep='\s*', header=None,
                          engine='python')
    face_array = np.array(face_df.iloc[:, 1:4])

    return vertex_array, face_array


# function to read binary coded ply file
def read_ply_binary(file):
    import re
    f = open(file, 'rb')
    line = ''
    while 'end_header' not in line:
        line = f.readline()
        if 'element vertex' in line:
            num_vertices = int(re.findall('\d+', line)[0])
        if 'element face' in line:
            num_faces = int(re.findall('\d+', line)[0])
    vertices = np.fromfile(f, count=num_vertices*3, dtype=np.float32).reshape(-1, 3)
    dt = np.dtype([('f1', np.uint8), ('f2', np.uint32), ('f3', np.uint32), ('f4', np.uint32)])
    faces = np.fromfile(f, count=num_faces*4, dtype=dt)
    faces = np.array([list(el)[1:] for el in faces], dtype=int)
    return vertices, faces


#function to read MNI obj mesh format
def read_obj(file):
    def chunks(l,n):
      """Yield n-sized chunks from l"""
      for i in xrange(0, len(l), n):
          yield l[i:i+n]
    def indices(lst,element):
        result=[]
        offset = -1
        while True:
            try:
                offset=lst.index(element,offset+1)
            except ValueError:
                return result
            result.append(offset)
    fp=open(file,'r')
    n_vert=[]
    n_poly=[]
    k=0
    Polys=[]
	# Find number of vertices and number of polygons, stored in .obj file.
	#Then extract list of all vertices in polygons
    for i, line in enumerate(fp):
         if i==0:
    	#Number of vertices
             n_vert=int(line.split()[6])
             XYZ=np.zeros([n_vert,3])
         elif i<=n_vert:
             XYZ[i-1]=map(float,line.split())
         elif i>2*n_vert+5:
             if not line.strip():
                 k=1
             elif k==1:
                 Polys.extend(line.split())
    Polys=map(int,Polys)
    npPolys=np.array(Polys)
    triangles=np.array(list(chunks(Polys,3)))
    return XYZ, triangles;



# function to save mesh geometry
def save_mesh_geometry(fname,surf_dict):
    # if input is a filename, try to load it with nibabel
    if isinstance(fname, basestring) and isinstance(surf_dict,dict):
        if (fname.endswith('orig') or fname.endswith('pial') or
                fname.endswith('white') or fname.endswith('sphere') or
                fname.endswith('inflated')):
            nb.freesurfer.io.write_geometry(fname,surf_dict['coords'],surf_dict['faces'])
#            save_freesurfer(fname,surf_dict['coords'],surf_dict['faces'])
        elif fname.endswith('gii'):
            write_gifti(fname,surf_dict['coords'],surf_dict['faces'])
        elif fname.endswith('vtk'):
            if 'data' in surf_dict.keys():
                write_vtk(fname,surf_dict['coords'],surf_dict['faces'],surf_dict['data'])
            else:
                write_vtk(fname,surf_dict['coords'],surf_dict['faces'])
        elif fname.endswith('ply'):
            write_ply(fname,surf_dict['coords'],surf_dict['faces'])
        elif fname.endswith('obj'):
            save_obj(fname,surf_dict['coords'],surf_dict['faces'])
            print('to view mesh in brainview, run the command:\n')
            print('average_objects ' + fname + ' ' + fname)
    else:
        raise ValueError('fname must be a filename and surf_dict must be a dictionary')

def write_gifti(surf_mesh, coords, faces):
    coord_array = nb.gifti.GiftiDataArray(data=coords,
                                       intent=nb.nifti1.intent_codes[
                                           'NIFTI_INTENT_POINTSET'])
    face_array = nb.gifti.GiftiDataArray(data=faces,
                                      intent=nb.nifti1.intent_codes[
                                           'NIFTI_INTENT_TRIANGLE'])
    gii = nb.gifti.GiftiImage(darrays=[coord_array, face_array])
    nb.gifti.write(gii, surf_mesh)


def save_obj(surf_mesh,coords,faces):
#write out MNI - obj format
    n_vert=len(coords)
    XYZ=coords.tolist()
    Tri=faces.tolist()
    with open(surf_mesh,'w') as s:
        line1="P 0.3 0.3 0.4 10 1 " + str(n_vert) + "\n"
        s.write(line1)
        k=-1
        for a in XYZ:
            k+=1
            cor=' ' + ' '.join(map(str, XYZ[k]))
            s.write('%s\n' % cor)
        s.write('\n')
        for a in XYZ:
            s.write(' 0 0 0\n')
        s.write('\n')
        l=' ' + str(len(Tri))+'\n'
        s.write(l)
        s.write(' 0 1 1 1 1\n')
        s.write('\n')
        nt=len(Tri)*3
        Triangles=np.arange(3,nt+1,3)
        Rounded8=np.shape(Triangles)[0]/8
        N8=8*Rounded8
        Triangles8=Triangles[0:N8]
        RowsOf8=np.split(Triangles8,N8/8)
        for r in RowsOf8:
            L=r.tolist()
            Lint=map(int,L)
            Line=' ' + ' '.join(map(str, Lint))
            s.write('%s\n' % Line)
        L=Triangles[N8:].tolist()
        Lint=map(int,L)
        Line=' ' + ' '.join(map(str, Lint))
        s.write('%s\n' % Line)
        s.write('\n')
        ListOfTriangles=np.array(Tri).flatten()
        Rounded8=np.shape(ListOfTriangles)[0]/8
        N8=8*Rounded8
        Triangles8=ListOfTriangles[0:N8]
        ListTri8=ListOfTriangles[0:N8]
        RowsOf8=np.split(Triangles8,N8/8)
        for r in RowsOf8:
            L=r.tolist()
            Lint=map(int,L)
            Line=' ' + ' '.join(map(str, Lint))
            s.write('%s\n' % Line)
        L=ListOfTriangles[N8:].tolist()
        Lint=map(int,L)
        Line=' ' + ' '.join(map(str, Lint))
        s.write('%s\n' % Line)


def write_vtk(filename, vertices, faces, data=None, comment=None):

    '''
    Creates ASCII coded vtk file from numpy arrays using pandas.
    Inputs:
    -------
    (mandatory)
    * filename: str, path to location where vtk file should be stored
    * vertices: numpy array with vertex coordinates,  shape (n_vertices, 3)
    * faces: numpy array with face specifications, shape (n_faces, 3)
    (optional)
    * data: numpy array with data points, shape (n_vertices, n_datapoints)
        NOTE: n_datapoints can be =1 but cannot be skipped (n_vertices,)
    * comment: str, is written into the comment section of the vtk file
    Usage:
    ---------------------
    write_vtk('/path/to/vtk/file.vtk', v_array, f_array)
    '''

    import pandas as pd
    # infer number of vertices and faces
    number_vertices=vertices.shape[0]
    number_faces=faces.shape[0]
    if data is not None:
        number_data=data.shape[0]
    # make header and subheader dataframe
    header=['# vtk DataFile Version 3.0',
            '%s'%comment,
            'ASCII',
            'DATASET POLYDATA',
            'POINTS %i float'%number_vertices
             ]
    header_df=pd.DataFrame(header)
    sub_header=['POLYGONS %i %i'%(number_faces, 4*number_faces)]
    sub_header_df=pd.DataFrame(sub_header)
    # make dataframe from vertices
    vertex_df=pd.DataFrame(vertices)
    # make dataframe from faces, appending first row of 3's (indicating the polygons are triangles)
    triangles=np.reshape(3*(np.ones(number_faces)), (number_faces,1))
    triangles=triangles.astype(int)
    faces=faces.astype(int)
    faces_df=pd.DataFrame(np.concatenate((triangles,faces),axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False, float_format='%.0f', sep=' ')
    # if there is data append second subheader and data
    if data != None:
        datapoints=data.shape[1]
        sub_header2=['POINT_DATA %i'%(number_data),
                     'SCALARS EmbedVertex float %i'%(datapoints),
                     'LOOKUP_TABLE default']
        sub_header_df2=pd.DataFrame(sub_header2)
        data_df=pd.DataFrame(data)
        with open(filename, 'a') as f:
            sub_header_df2.to_csv(f, header=False, index=False)
        with open(filename, 'a') as f:
            data_df.to_csv(f, header=False, index=False, float_format='%.16f', sep=' ')


def write_ply(filename, vertices, faces, comment=None):
    import pandas as pd
    print ("writing ply format")
    # infer number of vertices and faces
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    # make header dataframe
    header = ['ply',
            'format ascii 1.0',
            'comment %s' % comment,
            'element vertex %i' % number_vertices,
            'property float x',
            'property float y',
            'property float z',
            'element face %i' % number_faces,
            'property list uchar int vertex_indices',
            'end_header'
             ]
    header_df = pd.DataFrame(header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices)
    # make dataframe from faces, adding first row of 3s (indicating triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False,
                         float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False,
                        float_format='%.0f', sep=' ')

default_params = {
    'sample_size': 1000,
    'section_file': test.section_file,
    'labels': ['geo_dist'],
    'include_labels': ['good', 'excellent'],
    'split': None,
    'transformed_coords_file': test.transformed_coords_file,

    'mesh_left_file': test.mesh_civet_left_file,  # const.gray_left_file,
    'mesh_right_file': test.mesh_civet_right_file,  # const.gray_right_file,
    'geodesic_coord_system': test.geodesic_coord_system_file,

    'deterministic': True,
    'allow_nan_distance': True,

    'approximate_gdist': False,  # allow approximation
    'subdivision_level': 0,  # level of approximation (0 very coarse, inf very precise)

    'threaded': True,  # threaded calculation of geodesic distance
    'num_threads': 1  # number of openmp thread to use
}


class PointSampler(object):
    """ Generates pairs of points for siamese network training.
    Points are samples according to a probability volume.
    Returns the points on the 1um coordinate system, their sections numbers and their distances,
    POINTS HAVE VALUE X, Y!!!
    Uses the transformed coordinates in self.transformed_coords_file
    """

    # Variables initialized by __init__
    log = None
    probs = None
    probs_shape = None
    probs_spacing = None
    probs_offset = None
    mesh_left = None
    mesh_right = None
    transformed_coords = None

    def __init__(self, probability_volume, params={}):
        self.log = get_logger(self.__class__.__name__)
        self.log.info('Initialize PointSampler with volume {}'.format(probability_volume))

        # load probability_volume
        prob_file = nib.load(probability_volume)
        self.probs = np.array(prob_file.get_data()).flatten()
        self.probs = self.probs / self.probs.sum()
        self.probs_shape = prob_file.shape
        self.probs_spacing = prob_file.affine[0,0]
        self.probs_offset = prob_file.affine[:3,3]
        self.log.info('Loaded probability volume with spacing {} and offset {}'.format(self.probs_spacing, self.probs_offset))

        # init other parameters
        for key, default_value in default_params.iteritems():
            setattr(self, key, params.get(key, default_value))
            self.log.info('Setting {} to {}'.format(key, getattr(self, key)))

        # load meshes
        self.mesh_left = Mesh(self.mesh_left_file, coord_system=self.geodesic_coord_system, subdivision_level=self.subdivision_level, approximate=self.approximate_gdist, threaded=self.threaded, num_threads=self.num_threads)
        self.mesh_right = Mesh(self.mesh_right_file, coord_system=self.geodesic_coord_system, subdivision_level=self.subdivision_level, approximate=self.approximate_gdist, threaded=self.threaded, num_threads=self.num_threads)

        # load transformed_coords_file
        self.transformed_coords = h5py.File(self.transformed_coords_file, 'r')['coords']
        self.transformed_coords_sections = list(self.transformed_coords.attrs['sections'])
        if self.transformed_coords.attrs['spacing'] != self.probs_spacing:
            self.log.error('Probs spacing %d and transformed coords spacing %d are different! Potential problems!!', self.probs_spacing, self.transformed_coords.attrs['spacing'])
        self.log.info('Loaded transformed coords file')

        if self.deterministic:
            # seed random generator to get same points each time
            np.random.seed(0)

        self.next = self.point_pair_iterator().next

    # --- Helper functions ---
    def get_point_from_coord(self, x,y, section):
        """Returns points dict from given coord and section."""
        section_index = self.transformed_coords_sections.index(section)
        trans_x, trans_y, side = self.transformed_coords[section_index,y,x]
        mesh_point = ut.coord_to_mesh([x,y], section, spacing=self.probs_spacing, offset=self.probs_offset[:2][::-1], _type='obj')
        point = {'point': [x,y], 'mesh_point': mesh_point, 'transformed_point': [trans_x, trans_y],
                 'side': int(side), 'slice_no':section, 'spacing': self.probs_spacing}
        return point

    # --- Point / Pair generation
    def generate_points(self, num_points):
        """Generate num_points new random points.
        Calculate the transformed coords as well.
        Args:
            num_points (int): number for points that are generated
        Returns:
            point (dict): keys 'point', 'mesh_point', 'transformed_point', 'side', 'slice_no', 'spacing'
        """
        sections_for_value = ut.get_sections_for_coords(self.section_file, self.include_labels, self.split, self.probs_spacing, self.probs_offset[2])
        self.log.info('Generating {} new points'.format(num_points))
        points = np.random.choice(np.arange(0,self.probs.size), size=num_points, p=self.probs)
        coords = np.unravel_index(points, self.probs_shape)
        # choose section numbers from z coords
        sections = [np.random.choice(sections_for_value[coord]) for coord in coords[2]]

        self.log.info('Getting data for {} new points'.format(num_points))
        # build points with transformed coordinates and side
        points = []
        for i, section in enumerate(sections):
            y = coords[0][i]
            x = coords[1][i]
            points.append(self.get_point_from_coord(x,y,section))
        return points

    def generate_pairs(self, num_pairs):
        """Calculate num_pairs new random point pairs.
        Uses self.generate_points to get the individual points and then calculates their distances
        Args:
            num_pairs (int): number of pairs that are generated
        Returns:
            pairs (list of [point, point])
        """
        self.log.info('Generating %d new pairs'%num_pairs)
        points = deque(self.generate_points(num_pairs*2+10))
        pairs = []
        for _ in range(num_pairs):
            if len(points) < 2:
                points.extend(self.generate_points(10))
            point1 = points.popleft()
            point2 = points.popleft()
            if not self.allow_nan_distance:
                tmp = deque()
                while point1['side'] != point2['side']:
                    # put point2 in tmp queue and get new one
                    tmp.appendleft(point2)
                    if len(points) < 1:
                        points.extend(self.generate_points(10))
                    point2 = points.popleft()
                # clear tmp queue
                points.extendleft(tmp)
            pairs.append([point1, point2])
        self.log.info('Done generating %d new pairs'%num_pairs)
        return pairs

    def point_pair_iterator(self):
        """Iterate over random point pairs with their geodesic distance.
        Generate self.sample_size points on first call, and then on demand self.sample_size more.
        Yields:
            point1, point2, distance: points are dict returned by generate_points
        """
        queue = deque()
        while True:
            if len(queue) < 1:
                pairs = self.generate_pairs(self.sample_size)
                labels = None
                for label_name in self.labels:
                    if label_name in ('geo_dist', '2d_dist', '3d_dist'):
                        distances = self.get_distance(pairs, distance_type=label_name)
                        if labels is None:
                            labels = np.expand_dims(distances, -1)
                        else:
                            labels = np.concatenate((labels, np.expand_dims(distances, -1)), axis=-1)
                    elif label_name == 'direction':
                        directions = self.get_direction(pairs)
                        if labels is None:
                            labels = np.expand_dims(directions, -1)
                        else:
                            labels = np.conatenate((labels, np.expand_dims(directions, -1)), axis=-1)
                queue.extend([[p[0], p[1], lbl] for p, lbl in zip(pairs, labels)])
            point1, point2, lbl = queue.popleft()
            yield point1, point2, lbl

    def get_distance(self, pairs, distance_type='geo_dist'):
        """Geodesic distance between a list of point pairs.
        Args:
            points [(point1, point2)]: list of (dict, dict) with keys 'mesh_point' and 'side'
        Returns:
            list of geodesic distances between each point1 and point2.
            Return nan if the points are from different hemispheres.
        """
        self.log.info('Calculating distance for %d pairs using distance %s'%(len(pairs), distance_type))
        pairs = np.array(pairs)
        distance = np.zeros(len(pairs), dtype=np.float32)*np.nan
        for side, mesh in enumerate([self.mesh_left, self.mesh_right]):
            selected = np.array([True if p1['side'] == p2['side'] and p1['side'] == side else False for p1,p2 in pairs])
            points1 = [p1['mesh_point'] for p1, _ in pairs[selected]]
            points2 = [p2['mesh_point'] for _, p2 in pairs[selected]]
            dist = mesh.geodesic_distance(points1, points2, distance_type=distance_type)
            distance[selected] = dist
        return distance

    def get_direction(self, pairs):
        self.log.info('Calculating direction for %d pairs'%len(pairs))
        points1 = [p['mesh_point'] for p in np.asarray(pairs)[:,0]]
        points2 = [p['mesh_point'] for p in np.asarray(pairs)[:,1]]
        direction = ut.get_direction(points1, points2)
        return direction


class PointSamplerWithSeed(PointSampler):
    def __init__(self, seeds, probability_volume, params={}):
        """Sample
        Args:
            seeds: list of tuples (coord, section)
        """
        super(PointSamplerWithSeed, self).__init__(probability_volume, params)
        seed_points = []
        for seed in seeds:
            (x,y), section = seed
            seed_points.append(self.get_point_from_coord(x,y,section))
        assert (ut.get_sides(seed_points) == seed_points[0]['side']).all(), "all seeds need to have the same side!"
        self.side = seed_points[0]['side']
        self.log.info('Sampling for side %d'%self.side)
        self.seed_points = seed_points

    def generate_pairs(self, num_pairs):
        """Calculate num_pairs new point "pairs" with the first point being all seeds and the second point a random point.
        Uses self.generate_points to get the individual points and then calculates distances for all seeds to the second point
        Args:
            num_pairs (int): number of pairs that are generated
        Returns:
            pairs (list of [points, point])
        """
        self.log.info('Generating %d new pairs'%num_pairs)
        points = deque(self.generate_points(num_pairs*2+10))
        pairs = []
        point1 = self.seed_points
        for _ in range(num_pairs):
            if len(points) < 2:
                points.extend(self.generate_points(10))
            point2 = points.popleft()
            if not self.allow_nan_distance:
                # get new point2 because sides do not match
                while point2['side'] != self.side:
                    if len(points) < 1:
                        points.extend(self.generate_points(10))
                    point2 = points.popleft()
            pairs.append([point1, point2])
        self.log.info('Done generating %d new pairs'%num_pairs)
        return pairs

    def get_distance(self, pairs, distance_type='geo_dist'):
        """Geodesic distance between a list of point pairs.
        Assumes that all points have same side (this is the case with PointSamplerWithSeed)
        Assumes that first point in all pairs are the seed points.
        Args:
            points [(points1, point2)]: list of (list(dict), dict) with keys 'mesh_point' and 'side'
        Returns:
            list of geodesic distances between each point in points1 and corresponding point2.
            Return nan if the points are from different hemispheres.
        """
        self.log.info('Calculating distance for %d pairs using distance %s'%(len(pairs), distance_type))
        distances = np.zeros((len(pairs),len(pairs[0][0])), dtype=np.float32)
        mesh = [self.mesh_left, self.mesh_right][self.side]
        for i,pair in enumerate(pairs):
            self.log.debug('Calculating distance for pair %d'%i)
            source = pair[1]['mesh_point']
            targets = [pt['mesh_point'] for pt in pair[0]]
            dist = mesh.geodesic_distance_from_source(source, targets, distance_type=distance_type)
            distances[i,:] = dist
        return distances

    def get_direction(self, pairs):
        self.log.info('Calculating direction for %d pairs'%len(pairs))
        direction = []
        for points1,p2 in pairs:
            direction.append(ut.get_direction([p1['mesh_point'] for p1 in points1], [p2['mesh_point']]))
        return np.asarray(direction)


class PointSamplerPowerLaw(PointSampler):
    def __init__(self, probability_volume, num_pairs_random, num_pairs_power, num_points_power, params={}):
        '''Create PointSampler for num_pairs_random+num_pairs_power pairs **for each side**'''
        #self.log = logging.getLogger('PS')
        super(PointSamplerPowerLaw, self).__init__(probability_volume, params)

        num_points = num_pairs_random*2 + num_points_power
        G = self._calculate_graph(num_pairs_random, num_pairs_power, num_points_power)
        assert G.number_of_nodes() == num_points
        assert G.number_of_edges() == num_pairs_random + num_pairs_power
        if min(G.degree().values()) == 0:
            self.log.warning("Samples without pair exist in the current graph")

        # both sides have different random permutations of edges and samples
        self.pairs = [self._calculate_pairs(G), self._calculate_pairs(G, num_points)]
        self.graph = G
        self.num_points = num_points
        self.points = None
        self.data = None

    def _calculate_pairs(self, G, start=0):
        # calculate random permutation of edges as pairs
        samples = np.random.permutation(np.arange(start, start+G.number_of_nodes()))
        pairs = []
        for e in G.edges_iter():
            pairs.append([samples[e[0]], samples[e[1]]])
        pairs = np.asarray(pairs)
        # get random permutation of pairs
        return pairs[np.random.permutation(np.arange(len(pairs)))]

    def _calculate_graph(self, num_pairs_random, num_pairs_power, num_points_power):
        # calculate connection graph
        self.log.debug("Calculating %d pairs from %d samples with power-law distribution"%(num_pairs_power, num_points_power))
        G = nx.barabasi_albert_graph(int(num_points_power), int(num_pairs_power/num_points_power))
        # add random edges untill desired number of pairs is reached
        while G.number_of_edges() < num_pairs_power:
            missing_edges = num_pairs_power - G.number_of_edges()
            G.add_edges_from(zip(np.random.randint(0,num_points_power, size=missing_edges),
                             np.random.randint(0,num_points_power, size=missing_edges)))
        # add remaining pairs
        self.log.debug("Calculating %d pairs with random distibution"%num_pairs_random)
        edges = zip(np.arange(num_points_power, num_points_power+num_pairs_random*2, 2),
                    np.arange(num_points_power+1, num_points_power+num_pairs_random*2, 2))
        G.add_edges_from(edges)
        self.log.info("Calculated graph with %d edges and %d nodes"%(G.number_of_edges(), G.number_of_nodes()))
        return G

    def generate_points(self):
        '''Generate random points using superclass method. '''
        points_per_side = [[], []]
        points = super(PointSamplerPowerLaw, self).generate_points(self.num_points*2+100)
        #points = [{'x': np.random.randint(10000), 'side': np.random.randint(2)} for _ in range(self.num_points*2+100)]
        while len(points_per_side[0]) < self.num_points or len(points_per_side[1]) < self.num_points:
            for pt in points:
                s = pt['side']
                l = len(points_per_side[s])
                if l < self.num_points:
                    pt['sample_num'] = l+self.num_points*s
                    points_per_side[s].append(pt)
            #points = [{'x': np.random.randint(10000), 'side': np.random.randint(2)} for _ in range(self.num_points*2+100)]
            points = super(PointSamplerPowerLaw, self).generate_points(100)
        self.points = points_per_side

    def generate_pairs(self):
        if self.points is None:
            self.generate_points()
        points = self.points[0] + self.points[1]

        pairs = np.concatenate((self.pairs[0], self.pairs[1]))
        np.random.shuffle(pairs)
        pairs = [[points[pair[0]], points[pair[1]]] for pair in pairs]

        labels = None
        for label_name in self.labels:
            if label_name in ('geo_dist', '2d_dist', '3d_dist'):
                distances = self.get_distance(pairs, distance_type=label_name)
                if labels is None:
                    labels = np.expand_dims(distances, -1)
                else:
                    labels = np.concatenate((labels, np.expand_dims(distances, -1)), axis=-1)
            elif label_name == 'direction':
                directions = self.get_direction(pairs)
                if labels is None:
                    labels = np.expand_dims(directions, -1)
                else:
                    labels = np.conatenate((labels, np.expand_dims(directions, -1)), axis=-1)
        self.data = pairs, labels

    def point_pair_iterator(self):
        if self.data is None:
            self.generate_pairs()

        for pair, label in zip(*self.data):
            yield pair[0], pair[1], label


class Mesh(object):
    def __init__(self, mesh_file, coord_system=None, approximate=False, subdivision_level=0, threaded=True, num_threads=1, inflated_file=None):
        self.log = get_logger(self.__class__.__name__)
        self.log.info('Initializing mesh for {}'.format(mesh_file))
        # load data using Konrads mesh_io
        data = mesh_io.load_mesh_geometry(mesh_file)
        self.coords = data['coords'].astype(np.float64)
        self.triangs = data['faces'].astype(np.int32)
        # previously loaded data with
        #data = nib.load(mesh_file)
        #self.coords = data.darrays[0].data.astype(np.float64)
        #self.triangs = data.darrays[1].data.astype(np.int32)
        self.mesh = trimesh.Trimesh(self.coords, self.triangs)
        self.gdist_algo = gdist.GeodesicAlgorithm(self.coords, self.triangs,
                                                  approximate=approximate, subdivision_level=subdivision_level)
        self.threaded = threaded
        self.num_threads = num_threads
        self.coords_2d = None
        if coord_system is not None:
            self.log.info('Loading geodesic coordinate system from {}'.format(coord_system))
            self.coords_2d = self.load_geodesic_coord_system(coord_system)
        if inflated_file is not None:
            self.log.info('Loading inflated mesh from {}'.format(inflated_file))
            inflated_data = mesh_io.load_mesh_geometry(inflated_file)
            self.inflated_coords = inflated_data['coords']
        # create mapping of trimesh coords to mesh coords
        # sort both arrays (save indices)
        coords_ind = np.argsort(self.coords.view('f8,f8,f8'), order=['f0', 'f1', 'f2'], axis=0).flatten()
        trimesh_ind = np.argsort(self.mesh.vertices.view('f8,f8,f8'), order=['f0', 'f1', 'f2'], axis=0).flatten()
        # sort the indices of trimesh (trimesh_ind[trimesh_ind_argsort] == [0,1,2,...])
        trimesh_ind_argsort = np.argsort(trimesh_ind, axis=0)
        # mapping of trimesh index to mesh index mesh[trimesh_to_mesh[0]] = trimesh[0]
        self.trimesh_to_coord = coords_ind[trimesh_ind_argsort]

    def load_geodesic_coord_system(self, coord_file):
        coords = np.zeros((len(self.coords), 2))
        for i,line in enumerate(open(coord_file, 'r')):
            coords[i] = map(float, line.split(','))
        return coords

    def closest_points_on_mesh(self, points):
        """
        For each point calculate the closest vertex of the mesh.
        Args:
            points (nx3 ndarray): list of input points
            coords (mx3 ndarray): list of coords
            mesh (Trimesh): mesh consisting of coords
        Returns:
            (n,): list of indices (of coords) of the closest vertex to each point
        """
        dist, ind_mesh = trimesh.proximity.ProximityQuery(self.mesh).vertex(points)
        self.log.debug('Distance to mesh {}'.format(dist))
        ind_coord = self.trimesh_to_coord[ind_mesh]
        assert (self.coords[ind_coord] == self.mesh.vertices[ind_mesh]).all()
        return ind_coord

    def distance_to_mesh(self, points):
        dist, _ = trimesh.proximity.ProximityQuery(self.mesh).vertex(points)
        return dist

    def geodesic_distance_from_source(self, source_point, target_points, distance_type='geo_dist'):
        """ Calculates geodesic distance between one source point and a list of target points """
        if len(target_points) == 0:
            return []
        assert len(source_point) == 3 and len(target_points[0]) == 3
        ind_source = self.closest_points_on_mesh([source_point])
        ind_target = self.closest_points_on_mesh(target_points)
        if distance_type == '2d_dist':
            if self.coords_2d is None:
                self.log.error('Cannot calculate 2d distance without geodesic coordinate system. Using geo_dist as fallback!')
                distance_type = 'geo_dist'
            geo_dist = np.linalg.norm(self.coords_2d[ind_source]-self.coords_2d[ind_target], axis=1)
        if distance_type == 'geo_dist':
            geo_dist = self.gdist_algo.compute_gdist(np.array(ind_source, dtype=np.int32), np.array(ind_target, dtype=np.int32))
        if distance_type == '3d_dist':
            geo_dist = np.linalg.norm(self.coords[ind_source]-self.coords[ind_target], axis=1)
        return geo_dist

    def geodesic_distance(self,points_a, points_b, distance_type='geo_dist'):
        """ Caluclate geodesic distance between list of points a [(3,)] and list of points b [(3,)] """
        if len(points_a) == 0:
            return []
        assert len(points_a) == len(points_b)
        assert len(points_a[0]) == 3 and len(points_b[0]) == 3
        ind_a = self.closest_points_on_mesh(points_a)
        ind_b = self.closest_points_on_mesh(points_b)
        geo_dist = []
        if distance_type == '2d_dist':
            if self.coords_2d is None:
                self.log.error('Cannot calculate 2d distance without geodesic coordinate system. Using geo_dist as fallback!')
                distance_type = 'geo_dist'
            else:
                geo_dist = np.linalg.norm(self.coords_2d[ind_a] - self.coords_2d[ind_b], axis=1)
        if distance_type == 'geo_dist':
            if self.threaded:
                geo_dist = self.gdist_algo.compute_gdist_pairs(np.array(ind_a, dtype=np.int32), np.array(ind_b, dtype=np.int32), num_threads=self.num_threads)
            else:
                for a, b in zip(ind_a, ind_b):
                    geo_dist.extend(self.gdist_algo.compute_gdist(np.array([a], dtype=np.int32), np.array([b], dtype=np.int32)))
        if distance_type == '3d_dist':
            # calculate euclidean distances, from original points in mesh coord space
            geo_dist = np.linalg.norm(np.asarray(points_a) - np.asarray(points_b), axis=1)
        self.log.debug('Geodesic distance {}'.format(geo_dist))
        return geo_dist

    # ported from create_transformed_coords in bigbrain data folder
    @staticmethod
    def get_side(mesh_points, mesh_left, mesh_right):
        """ 0 for left, 1 for right """
        dist_left = mesh_left.distance_to_mesh(mesh_points)
        dist_right = mesh_right.distance_to_mesh(mesh_points)
        return np.argmin([dist_left, dist_right], axis=0)

