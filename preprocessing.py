# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utilities to create, read, write tf.Examples.'''

import h5py as h5
import os
from parse import *
from torch.utils import data
import numpy as np



def save_h5_examples(fname, features, pi, value):
    '''
    Args:
        features: [N, N, FEATURE_DIM] nparray of uint8
        pi: [N * N + 1] nparray of float32
        value: float
    '''
    file=h5.File(fname, "w")
    dfeatures = file.create_dataset('features', data = features)
    dpi = file.create_dataset('pi', data = pi)
    dvalue = file.create_dataset('value', data = value)
    file.close()

class Dataset(data.Dataset):
    def __init__(self, data_path, model_name):
        self.list_IDs=os.listdir(os.path.join(data_path, model_name))
        self.lengths=[]
        self.data_path=data_path
        self.model_name=model_name
        for name in self.list_IDs :
            _, length = parse("{}-{}.hdf5", name)
            self.lengths.append(int(length))
        self.cumsum = np.cumsum(self.lengths)



    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        nfile=np.where(self.cumsum>idx)[0][0]
        f=h5.File(os.path.join(self.data_path, self.model_name, self.list_IDs[nfile]))
        features=f['features'][idx-self.cumsum[nfile-1 if nfile >= 1 else 0]]
        pi=f['pi'][idx-self.cumsum[nfile-1 if nfile >=1 else 0]]
        value=f['value'][idx-self.cumsum[nfile-1 if nfile >=1 else 0]]
        sample = {'features': features, 'pi':pi, 'value' : value}
        return sample





def make_train_dataset():
    pass





