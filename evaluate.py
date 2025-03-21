import os
import re
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from PIL import Image
import base64
import zlib

import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
import spacy
import pdb

from dataset.coco import PanopticCOCO

from utils.evaluator import BaseEvaluator
from utils.metric.llama3 import LLama


def build_scorer(model):
    if 'Llama' in model:
        scorer = LLama(model)
    else:
        raise ValueError(f"{model} is not avalible")
    return scorer

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

class Evaluator(BaseEvaluator):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

        self.scorer = build_scorer(args.llm)

        self.nlp = spacy.load('en_core_web_lg')
        self.bert = SentenceTransformer(args.bert)

        # get evaluate data from DCI
        self.data = PanopticCOCO(args.data_root, args.anno, args.use_extend_caption)
        

        self.soft_coverage = args.soft_coverage
        self.threshold = args.threshold
        self.num_data = min(args.data_num, len(self.data)) if args.data_num>0 else len(self.data)
    
    def get_key_words(self, sub_captions):
        # find all noun words from sub_captions, 
        # and the relation between caption and word: 
        # word2caption{word_id1: sub_caption_id1, word_id2: sub_caption_id2}
        words = []
        word2caption = {}
        for i, text in enumerate(sub_captions):
            doc = self.nlp(text)
            candidates = []
            for token in doc:
                if token.tag_[0] == 'N':
                    candidates.append(str(token.lemma_.lower()))

            if len(candidates) == 0:
                continue

            cur_words = candidates
                
            start_id = len(words)
            words = words + cur_words
            for w in range(start_id, len(words)):
                word2caption[w] = i

        return words, word2caption


    def start(self):

        with torch.no_grad():
            for index in tqdm(range(self.num_data)):
                image, gt_objects, gt_description, mask, gt_relations = self.data.get_data(index)
                if len(self.long_caption[image]) == 0:
                    print('There is not long captions for image: ', image)
                    continue

                sub_captions = self.split_caption(self.long_caption[image])
                phrases, word2caption = self.get_key_words(sub_captions)
                if len(gt_objects)==0:
                    continue

                no_overlap_gt_objects = list(set(gt_objects))
                match_object, object_score = self.get_coverage(no_overlap_gt_objects, phrases) # matrix of Bool
                des_score, relation_score = self.get_accuracy(gt_objects, no_overlap_gt_objects, gt_description, gt_relations, \
                    sub_captions, word2caption, match_object)
                self.get_pixel_coverage(des_score, image, mask)

                result = {'object':{}, 'attribute':{}, 'relation':{}}

                for k, gt_object in enumerate(no_overlap_gt_objects):
                    result['object'][gt_object] = float(object_score[k])
                for k, des in enumerate(gt_description):
                    result['attribute'][des] = float(des_score[k])
                relations = []
                for r in gt_relations:
                    if isinstance(r, list):
                        relations += r
                    elif isinstance(r, str):
                        relations.append(r)
                assert len(relations)==len(relation_score)
                for k, relation in enumerate(relations):
                    result['relation'][relation] = float(relation_score[k])

                self.log_results(image, result)



    def get_coverage(self, no_overlap_gt_objects, phrases):
        # calculate object-level coverage
        # how many kinds of objects (no overlap) are mentioned in the caption produced by mllm

        # the cos sim between words
        # cosine_scores = torch.zeros(len(no_overlap_gt_objects), len(phrases))
        # gts_embed = [self.nlp(gt) for gt in no_overlap_gt_objects]
        # preds_embed = [self.nlp(pred) for pred in phrases]
        # for i, gt in enumerate(gts_embed):
        #     for j, pred in enumerate(preds_embed):
        #         cosine_scores[i,j] = gt.similarity(pred)
        with torch.no_grad():
            gts_embed = self.bert.encode(no_overlap_gt_objects)
            preds_embed = self.bert.encode(phrases)
            cosine_scores = self.bert.similarity(gts_embed, preds_embed)
            
        max_sim = cosine_scores.max(dim=0)[0]
        gt2preds_max_sim = cosine_scores.max(dim=1)[0]
        
        # The two words displaying the highest similarity along the x and y axes are considered to be a matched objects.
        match_object = (cosine_scores==max_sim.unsqueeze(0))*(cosine_scores==gt2preds_max_sim.unsqueeze(-1))
        contained_object_idx = torch.where(match_object.int().sum(dim=-1)>0)[0]
        object_score = match_object.int().sum(dim=-1)>0

        if self.soft_coverage:
            # If need to report soft value,
            # the reliability of the gt object mentioned in the caption is the max similarity between  it and all compared phrases.
            contained_object_score = [gt2preds_max_sim[i].item() for i in contained_object_idx]
            object_score = object_score*gt2preds_max_sim
            cur_coverage = sum(contained_object_score)/len(no_overlap_gt_objects)
        else:
            contained_object = [no_overlap_gt_objects[i] for i in contained_object_idx]
            cur_coverage = len(contained_object)/len(no_overlap_gt_objects)
        self.update_object_coverage(cur_coverage)
        return match_object, object_score

    def get_accuracy(self, gt_objects, no_overlap_gt_objects, gt_description, gt_relations, sub_captions, word2caption, match_object):
        _match_object = []
        for gt in gt_objects:
            object_id = no_overlap_gt_objects.index(gt)
            _match_object.append(match_object[object_id])
        match_object = torch.stack(_match_object, dim=0)
        
        long_text = '. '.join(sub_captions)

        preds, gts, rels, rels_preds = [], [], [], []
        for k in range(len(gt_description)):
            cur_match = list(match_object[k,:].int().nonzero().view(-1))
            data_index = len(preds)

            gts.append(gt_description[k])
            object_captions = sorted([word2caption[int(cur_)] for cur_ in cur_match])
            pred_caption = [sub_captions[subcaption_id] for subcaption_id in object_captions]
            # cat all subcaptions corresponding to the same gt. 
            preds.append('. '.join(pred_caption))

            # the preds sentence contain all mentioned the object in the relation
            if len(gt_relations[k])>0:
                for cur_obj_gt_relation in gt_relations[k]:
                    for k_obj, gt_object in enumerate(gt_objects):
                        if k_obj==k:
                            continue
                        if gt_object in cur_obj_gt_relation:
                            cur_match_2 = list(match_object[k_obj,:].int().nonzero().view(-1))
                            object_captions = object_captions+sorted([word2caption[int(cur_)] for cur_ in cur_match_2])
                    relation_pred_caption = [sub_captions[subcaption_id] for subcaption_id in sorted(list(set(object_captions)))]
                    rels_preds.append('. '.join(relation_pred_caption))
                    rels.append(cur_obj_gt_relation)
            else:
                rels_preds.append('')
        
        object_score, relation_score = self.scorer.evaluate(gts, preds, rels, rels_preds) # the scores of each gt objects
        self.update_score(sum(object_score)/len(gt_description))

        assert len(relation_score)>0
        self.update_relation_score(sum(relation_score)/len(relation_score))
        
        return object_score, relation_score

    def get_pixel_coverage(self, scores, image, mask):
        # calculate pixel-level coverage
        # Each object has a corresponding mask
        image = Image.open(os.path.join(self.data.image_root,image))
        seen_area = np.zeros((image.size[1], image.size[0]))
        
        for score, object_mask in zip(scores, mask):
            object_mask = get_binary_mask(image, object_mask)
            seen_area = seen_area + object_mask*score/5
            seen_area = np.clip(seen_area, 0, 1, out=None)

        total_area = np.zeros((image.size[1], image.size[0]))
        for object_mask in mask:
            object_mask = get_binary_mask(image, object_mask)
            total_area = 1*((total_area + object_mask)>0)

        # cur_coverage = np.sum(seen_area)/(seen_area.shape[0]*seen_area.shape[1])
        cur_coverage = np.sum(seen_area)/np.sum(total_area)
        self.update_pixel_coverage(cur_coverage)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("LongCaption-Evaluator", add_help=True)
    parser.add_argument("--data_root", type=str, required=True, help="path to image file")
    parser.add_argument("--anno", type=str, required=True, help="path to annotation file")
    parser.add_argument("--use_extend_caption", action="store_true")
    parser.add_argument("--longcap_file", type=str, required=True, help="path to longcaption")

    parser.add_argument("--llm", type=str,  help="path to llm")
    parser.add_argument("--bert", type=str,  help="path to bert weight")
    parser.add_argument("--soft_coverage", action="store_true", help="use word similarity as coverage")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--threshold", type=int, default=3, help="the threshold for calculating pixel-level coverage")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--k", type=int, default=0, help="eval time")
    parser.add_argument("--data-num", type=int, default=-1, help="the num of data to be used for eval")
    args = parser.parse_args()

    evaluator =  Evaluator(args)
    print("Start evaluating long caption from ", args.longcap_file)
    evaluator.start()
    if evaluator.count<evaluator.num_data:
        print('only ', str(evaluator.count), ' data are used for evaluation (<', str(evaluator.num_data),').')
    evaluator.get_results()