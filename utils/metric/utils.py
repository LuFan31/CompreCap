import re

# prompt_template="Sentence: {sentence}. Phrase: {phrase}. Please provide an integer score as a single number from 0 to 5."
# relation_prompt_template="Sentence: {sentence}. Phrase: {phrase}. Please check if the sentence accurately describes the relationship that is referenced in the provided phrase and provide an integer score as a single number from 0 to 5."
sys_prompt= "You are a Natural Language Processing (NLP) expert. A curious human will give you a sentence and a phrase. The human want you to help with analyzing whether the sentence includes similar concept with the given phrase and rate the similarity on a scale from 0 to 5, with 0 being 'completely lacks similar concepts' and 5 being 'extremely has similar concepts'. You need to give helpful and reliable answers to help the human."


prompt_template="Sentence: {sentence}. Phrase: {phrase}. Please provide an integer score as a single number from 0 to 5 without explanation."
relation_prompt_template="Sentence: {sentence}. Phrase: {phrase}. Please check if the sentence accurately describes the relationship that is referenced in the provided phrase and provide an integer score as a single number from 0 to 5 without explanation."

def get_first_digit(text):
    pattern = r"\d+"
    matcher = re.search(pattern, text)
    if matcher:
        return int(matcher.group())
    else: return -1