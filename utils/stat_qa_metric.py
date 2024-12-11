from dataset.coco import PanopticCOCO
from tqdm import tqdm
from PIL import Image
import os
import base64
import zlib
import numpy as np
import struct
import pdb
import random
import json
from nltk.corpus import wordnet
import random
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# with open ('../data/finegrade_object_QA/Fine-grained_QA_gpt4v_novisible_inpromptV3.jsonl', 'r') as infile:
#     num_correct, num_total, num_blindness = 0, 0, 0
#     for line in infile:
#         qa_line = json.loads(line)
#         image_name = qa_line['image_name']
#         object_category = qa_line['category']
#         prediction = qa_line['prediction']
#         res = qa_line['correct']
#         if res == 'Yes!':
#             num_correct += 1
#         num_total += 1
#         if 'can not be seen in the image' in prediction:
#             num_blindness += 1
#     print(f"The accuracy is {num_correct/num_total}")
#     print(f"The blindness rate is {num_blindness/num_total}")
#     print(f"The the error rate is {(num_total-num_correct-num_blindness)/num_total}")
#     print(f"The blindness rate in the all inaccuracy is {(num_blindness)/(num_total-num_correct)}")
#     print(f"The hulla rate is {(num_total-num_correct-num_blindness)/(num_total-num_blindness)}")

def get_binary_mask(image, arr, dtype='uint8'):
    w, h = image.size
    if not arr or len(arr) <= 1 or not isinstance(arr, str):
        return None
    str_data = base64.b64decode(arr) # dtype: bytes
    if dtype == 'uint32':
        data = np.array(struct.unpack('I' * (len(str_data) // 4), str_data), dtype=np.uint32)
    else:
        data = np.frombuffer(str_data, dtype=np.uint8) # dtype: ndarray
    decompressed_data = zlib.decompress(data)  # dtype: bytes
    array = np.frombuffer(decompressed_data, dtype=np.uint8)
    array = np.where(array > 0, 1, array)
    array = array.reshape((1, h, w))[0]
    return array

def compute_pixel_coverage(image_name, object_mask_out):
    image = Image.open(os.path.join('../data/coco_ospery/images',image_name))
    object_mask = get_binary_mask(image, object_mask_out)
    object_pixel_coverage = np.sum(object_mask)/(object_mask.shape[0]*object_mask.shape[1])
    return object_pixel_coverage    

with open('../data/coco_ospery/short_anno.json', 'r') as f:
    descriptions = json.load(f)
    description_dict = {}
    for item in descriptions:
        image_name = item['image']
        description_dict[image_name] = item['info']

sns.set(style='whitegrid')

# with open (f'../data/QA_json/finegri_desc_qa.jsonl', 'r') as infile:
#     object_pixel_coverage_list = []
#     num_samll_object = 0
#     for line in infile:
#         num_samll_object += 1
#         qa_line = json.loads(line)
#         image_name = qa_line['image_name']
#         object_category = qa_line['category']
#         cur_desc = description_dict[image_name]
#         for desc in cur_desc:
#             if desc['category'] == object_category:
#                 object_mask_out = desc['mask_out']
#                 object_pixel_coverage = compute_pixel_coverage(image_name, object_mask_out)
#                 object_pixel_coverage_list.append(object_pixel_coverage)

#     data = np.array(object_pixel_coverage_list)
#     print(f'The number of small object is {num_samll_object}')
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data, bins=num_samll_object, kde=True, color='skyblue', stat='count')

#     plt.title('The pixel area occupies of objects in JigsawQA')
#     plt.xlabel('Pixel Area Occupy')
#     plt.ylabel('Frequency(%)')
#     plt.xlim(0, 0.05)
#     plt.savefig(f'small_object_coverage_distribution.pdf')
#     plt.clf()


for file in os.listdir('../data/FineGri_QA'):
    if 'finegri_desc_qa' in file:
         desc_file = file
         method_name = desc_file.split('-finegri_desc_qa.jsonl')[0]
    elif 'GPT4v-fine' in file:
        desc_file = file
        method_name = 'GPT4v'
    else:
        desc_file = None
    if desc_file is not None:
        num_error = 0
        object_pixel_coverage_list = []
        # pdb.set_trace()
        with open (f'../data/FineGri_QA/{desc_file}', 'r') as infile:
            for line in infile:
                qa_line = json.loads(line)
                image_name = qa_line['image_name']
                object_category = qa_line['category']
                prediction = qa_line['prediction']
                res = qa_line['correct']
                if res == 'No!':
                    num_error += 1
                    cur_desc = description_dict[image_name]
                    # pdb.set_trace()
                    for desc in cur_desc:
                        if desc['category'] == object_category:
                            object_mask_out = desc['mask_out']
                            object_pixel_coverage = compute_pixel_coverage(image_name, object_mask_out)
                            object_pixel_coverage_list.append(object_pixel_coverage)


            data = np.array(object_pixel_coverage_list)

            # print(f'The current file is {file}')
            print(f'The number of error is {num_error}')
            plt.figure()
            sns.histplot(data, bins=num_error, kde=True, color='skyblue')

            plt.title('Data Distribution from 0 to 0.05')
            plt.xlabel('Value')
            plt.ylabel('Frequency(%)')
            plt.xlim(0, 0.05)
            plt.savefig(f'../data/sata_qa/{method_name}-error-object_pixel_coverage.png')
            plt.clf()