import os
import glob
import time
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from pdb import set_trace as st

padding = 1.2

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class FullPSRField(Field):
    def __init__(self, transform=None, multi_files=None):
        self.transform = transform
        # self.unpackbits = unpackbits
        self.multi_files = multi_files
    
    def load(self, model_path, idx, category):

        # try:
        # t0 = time.time()
        if self.multi_files is not None:
            psr_path = os.path.join(model_path, 'psr', 'psr_{:02d}.npz'.format(idx))
        else:
            psr_path = os.path.join(model_path, 'psr.npz')

        psr_dict = np.load(psr_path)
        # t1 = time.time()
        psr = psr_dict['psr']
        psr = psr.astype(np.float32)
        # t2 = time.time()
        # print('load PSR: {:.4f}, change type: {:.4f}, total: {:.4f}'.format(t1 - t0, t2 - t1, t2-t0))
        data = {None: psr}
        
        if self.transform is not None:
            data = self.transform(data)

        return data

class FullPSRFieldAbc(Field):
    def __init__(self, grid_res: int, transform=None):
        self.transform = transform
        self.grid_res = grid_res
    
    def load(self, model_path, idx, category):

        # try:
        # t0 = time.time()
        model_name = os.path.basename(model_path)
        dataset_name = os.path.basename(os.path.split(model_path)[0])
        psr_path = os.path.join('data', 'p2s', dataset_name, 'psr', '{}.npz'.format(model_name))

        def _make_dpsr():  # called in workers -> get item -> can be concurrent
            from src.dpsr import DPSR
            from torch import from_numpy
            
            dataset_path = os.path.split(model_path)[0]
            gt_mesh_path = os.path.join(dataset_path, '03_meshes', '{}.ply'.format(model_name))
            mesh = trimesh.load(gt_mesh_path)

            points, face_idx = mesh.sample(100000, return_index=True)
            normals = mesh.face_normals[face_idx]

            # to [0..1] like in https://github.com/autonomousvision/shape_as_points/blob/main/scripts/process_shapenet.py
            points = points / 2.0 / padding + 0.5

            dpsr = DPSR(res=(self.grid_res, self.grid_res, self.grid_res), sig=0)
            psr_gt = dpsr(from_numpy(points.astype(np.float32))[None], 
                          from_numpy(normals.astype(np.float32))[None]).squeeze().cpu().numpy().astype(np.float16)
            
            os.makedirs(os.path.dirname(psr_path), exist_ok=True)
            np.savez(psr_path, psr=psr_gt)

        if not os.path.exists(psr_path):  # create DPSR on the fly for next time...
            with open('myfile.txt', 'w') as fp:  # create empty file to avoid concurrency problems
                pass
            
            print('Warning: creating DPSR for {}'.format(psr_path))
            _make_dpsr()

        load_successful = False
        attempts = 0
        while not load_successful:
            try:
                attempts += 1
                psr_dict = np.load(psr_path)
                load_successful = True
            except Exception as ex:
                import time
                time.sleep(3.0)

                if attempts >= 3:
                    print('Warning: encountered broken NPZ, creating DPSR for {}'.format(psr_path))
                    _make_dpsr()
                    psr_dict = np.load(psr_path)

        # t1 = time.time()
        psr = psr_dict['psr']
        psr = psr.astype(np.float32)
        # t2 = time.time()
        # print('load PSR: {:.4f}, change type: {:.4f}, total: {:.4f}'.format(t1 - t0, t2 - t1, t2-t0))
        data = {None: psr}
        
        if self.transform is not None:
            data = self.transform(data)

        return data


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, data_type=None, transform=None, multi_files=None, padding=0.1, scale=1.2):
        self.file_name = file_name
        self.data_type = data_type # to make sure the range of input is correct
        self.transform = transform
        self.multi_files = multi_files
        self.padding = padding
        self.scale = scale

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            # num = np.random.randint(self.multi_files)
            # file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
            file_path = os.path.join(model_path, self.file_name, 'pointcloud_%02d.npz' % (idx))
        
        if os.path.isfile(file_path):
            pointcloud_dict = np.load(file_path)

            points = pointcloud_dict['points'].astype(np.float32)
            normals = pointcloud_dict['normals'].astype(np.float32)
        else:  # assume P2S datasets
            model_name = os.path.basename(model_path)
            dataset_dir = os.path.split(model_path)[0]
            file_path = os.path.join(dataset_dir, '04_pts', '{}.xyz.npy'.format(model_name))
            pointcloud_npy = np.load(file_path)
            points_scan = pointcloud_npy.astype(np.float32)

            # normals_scan = np.zeros_like(points_scan)
            # normals_scan[:, 0] = 1.0

            gt_mesh_path = os.path.join(dataset_dir, '03_meshes', '{}.ply'.format(model_name))
            mesh = trimesh.load(gt_mesh_path)
            points, face_idx = mesh.sample(points_scan.shape[0], return_index=True)
            normals = mesh.face_normals[face_idx]

            # to [0..1] like in https://github.com/autonomousvision/shape_as_points/blob/main/scripts/process_shapenet.py
            points = points / 2.0 / padding + 0.5
            points_scan = points_scan / 2.0 / padding + 0.5
            
        data = {
            None: points,
            'normals': normals,
            'points_scan': points_scan,
        }
        if self.transform is not None:
            data = self.transform(data)
        
        if self.data_type == 'psr_full' or self.data_type == 'psr_full_abc':
            # scale the point cloud to the range of (0, 1)
            data[None] = data[None] / self.scale + 0.5
            
            if 'points_scan' in data.keys():
                data['points_scan'] = data['points_scan'] / self.scale + 0.5

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
