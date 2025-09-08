# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from math import sqrt, pow, exp
import numpy as np

cos_model_name = "distiluse-base-multilingual-cased"
def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
def squared_sum(x):
    """ return 3 rounded square rooted value """
    
    return round(sqrt(sum([a*a for a in x])),3)
 
def euclidean_distance(x,y):
    """ return euclidean distance between two lists """
    
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def distance_to_similarity(distance):
    return 1/exp(distance)

def cos_similarity(x,y):
    """ return cosine similarity between two lists """    
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    return round(numerator/float(denominator),3)

def create_embeddings (text, SentenceTransformer_model): 
    embeddings = SentenceTransformer_model.encode(list(text))
    if len(embeddings) !=0:
        return list(embeddings[0])
    else:
        return [0]
def calculate_bleu_scores(references, hypotheses):
    """
    Calculates BLEU 1-4 scores based on NLTK functionality

    Args:
        references: List of reference sentences
        hypotheses: List of generated sentences

    Returns:
        bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores

    """
    #return len(references), len(hypotheses)
    bleu_1 = np.round(corpus_bleu(references, hypotheses, weights=(1.0, 0., 0., 0.)), decimals=2)
    return bleu_1


#arguments to be parsed from command line
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
# ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
# ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
# ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
# ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
# ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
# ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
# args = ap.parse_args()

# #the output file
# output = None
# if args.output:
#     output = args.output
# else:
#     from os.path import dirname, basename, join
#     output = join(dirname(args.input), 'eda_' + basename(args.input))

# #number of augmented sentences to generate per original sentence
# num_aug = 9 #default
# if args.num_aug:
#     num_aug = args.num_aug

# #how much to replace each word by synonyms
# alpha_sr = 0.1#default
# if args.alpha_sr is not None:
#     alpha_sr = args.alpha_sr

# #how much to insert new words that are synonyms
# alpha_ri = 0.1#default
# if args.alpha_ri is not None:
#     alpha_ri = args.alpha_ri

# #how much to swap words
# alpha_rs = 0.1#default
# if args.alpha_rs is not None:
#     alpha_rs = args.alpha_rs

# #how much to delete words
# alpha_rd = 0.1#default
# if args.alpha_rd is not None:
#     alpha_rd = args.alpha_rd

# if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
#      ap.error('At least one alpha should be greater than zero')

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    # print("terprint")
    newDF        = pd.DataFrame()
    # print(train_orig)
    lines = open(train_orig, 'r', encoding='utf-16').readlines()
    # lines = open(train_orig, 'r', encoding='utf-8').readlines()
    SentenceTransformer_model = SentenceTransformer(cos_model_name)

    for i, row in enumerate(lines):
#         if i < 1309:
#             continue
        if len(row) < 2:  # Validate that the row has at least two columns
            continue
        print(i)
        parts = row[:-1].split('\t')
        label = parts[0]
        sentence = parts[1]
        # print (sentence)
        if sentence == '':
            continue
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        # print ("num_aug :", num_aug)
        # print ("aug_sentences :", aug_sentences)
        for a in range(len(aug_sentences)):
            # print ("a :", a)
            text = sentence
            all_text = aug_sentences[a]

            # Embedding
            embd1 = create_embeddings(text=text, SentenceTransformer_model=SentenceTransformer_model)
            embd2 = create_embeddings(text=all_text, SentenceTransformer_model=SentenceTransformer_model)

            new_embd1 = ','.join(str(x) for x in embd1)
            new_embd2 = ','.join(str(x) for x in embd2)

            # Similarities
            esim = euclidean_distance(embd1, embd2)
            csim = cos_similarity(embd1, embd2)
            jsim = jaccard_similarity(text, all_text)

            # BLEU
            txt_split = text.split()
            all_text_split = all_text.split()
            min_len = min(len(txt_split), len(all_text_split))
            bleu = calculate_bleu_scores(txt_split[:min_len], all_text_split[:min_len])

            # Buat row dataframe
            tmp = {
                'text': [sentence],
                'label': [label],
                'all_text': [all_text],
                'original_embedding': [new_embd1],
                'new_embedding': [new_embd2],
                'ecu_similarity': [esim],
                'cos_similarity': [csim],
                'jacc_similarity': [jsim],
                'bleu_similarity': [bleu]
            }

            tmpDF = pd.DataFrame(tmp)
            newDF = pd.concat([newDF, tmpDF], ignore_index=True)
    # print("terprint")
    # print(newDF)
    newDF.to_csv(output_file, sep="\t", header=0, index=False) #encoding='utf-16