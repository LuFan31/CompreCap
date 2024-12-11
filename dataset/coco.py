#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import OrderedDict
import argparse
import json
import math
import os
import base64
from os.path import exists, join, split
from io import BytesIO
from tqdm import tqdm

import time

import numpy as np
from PIL import Image


_COCO_PANOPTIC_INFORMATION = {
    'num_classes': 133,
    'ignore_label': 255,
}
    
class PanopticCOCO():

    def __init__(self,data_root, anno_file, use_extend_caption):

        self.image_root = os.path.join(data_root, 'images')

        with open(anno_file, 'r') as f:
            info = json.load(f)
        f.close
        self.annos = info

        self.num_classes = _COCO_PANOPTIC_INFORMATION['num_classes']
        self.ignore_label = _COCO_PANOPTIC_INFORMATION['ignore_label']
        self.use_extend_caption = use_extend_caption
        
    def __len__(self):
        return len(self.annos)

    def get_data(self, idex):
        image = self.annos[idex]['image']

        objects = []
        descriptions = []
        masks = []
        relations = []
        for info in self.annos[idex]['info']:
            cur_object = info['category']
            objects.append(cur_object)
            if self.use_extend_caption:
                descriptions.append(info['extend_description'])
            else:
                descriptions.append(info['short_description'])
            masks.append(info['mask_out'])
            relations.append(info['relation'])

        return image, objects, descriptions, masks, relations