import os
import re
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm

class BaseEvaluator(object):
    def __init__(self, args):
        self.device = args.device
        self.long_caption = self.load_longcaption(args.longcap_file)
        self.count = 0
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_info = {}
        self.reset_record()
        self.save_index = args.k

    def reset_record(self):
        self.results = {
            'object coverage': 0.,
            'pixel coverage': 0.,
            'object score': 0.,
            'relation score': 0.,
        }
        self.count = 0

    def load_longcaption(self, path):
        with open(path, 'r') as f:
            long_caption = json.load(f)
        f.close()
        return long_caption
    
    def split_caption(self, text):
        texts = re.split(r'\n|\\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap

    def update_object_coverage(self, coverage):
        self.results['object coverage'] += coverage*100.
        self.count += 1

    def update_pixel_coverage(self, coverage):
        self.results['pixel coverage'] += coverage*100.

    def update_score(self, score):
        self.results['object score'] += score

    def update_relation_score(self, score):
        self.results['relation score'] += score

    def start(self):
        pass

    def log_results(self, image_name, results):
        self.log_info[image_name] = results
                
    def get_results(self):
        result_logs = {}
        for k, v in self.results.items():
            result_logs[k] = f'{v/self.count:.2f}'
            string = f'{k}: {v/self.count:.2f}'
            if 'coverage' in k: string = string+'%'
            print(string)
        with open(os.path.join(self.output_dir, str(self.save_index)+'_result.json'), 'w') as f:
            log_info = [result_logs]+[self.log_info]
            json.dump(log_info, f, indent=2)

