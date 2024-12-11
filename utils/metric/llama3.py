import transformers
import torch
from transformers import AutoTokenizer
from utils.metric.utils import *

class LLama:
    def __init__(self, model):

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
        )

    def evaluate(self, gts, preds, relations, relation_preds, object_align=None):
        scores = []
        relation_scores = []
        for phrase, sentence in zip(gts, preds):
            prompt = prompt_template.format(sentence=sentence, phrase=phrase)
            if len(sentence)==0:
                scores.append(0)
                continue   
            score = self.generate(prompt)
            scores.append(score)

        for relation, relation_sentence in zip(relations, relation_preds):
            if len(relation_sentence)==0:
                relation_scores.append(0)
                continue   
            prompt = relation_prompt_template.format(sentence=relation_sentence, phrase=relation)
            score = self.generate(prompt)
            relation_scores.append(score)

        return scores, relation_scores

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        score = get_first_digit(outputs[0]["generated_text"][len(prompt):])
        return score