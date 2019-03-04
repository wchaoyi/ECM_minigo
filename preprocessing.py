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
import numpy as np

from torch.utils import data



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




