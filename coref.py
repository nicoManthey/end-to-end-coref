# %%
import re
import os
from subprocess import Popen, PIPE
from pathlib import Path
from random import shuffle
import random
from collections import defaultdict
from itertools import groupby, combinations
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from tqdm import tqdm
from typed_ast.ast27 import Pass
import yaml
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#from highlight_text import ax_text, fig_text


from torch.autograd import Variable, Function
from torch.nn.modules.loss import _Loss, _WeightedLoss

import time



################################################
#             read-in CoNLL-files
#################################################


def read_conll_file(input_file_path):
    """ Read a conll-file and return a list of document-objects. """
    documents = []
    document_genre = Path(input_file_path).name.split("_")[0]
    col_to_id = {
        "doc_id": 0,
        "part_id": 1,
        "word_id": 2,
        "word": 3,
        "pos": 4,
        "parse_tree": 5,
        "predicate_lemma": 6,
        "predicate_lemma_frameset": 7,
        "word_sense": 8,
        "speaker": 9,
        "named_entities": 10,
        "corefs": 11,
    }

    with open(input_file_path, "r+") as file:
        
        while True:
            line = file.readline()
            if not line:
                break
            

            if line.split()[:2] == ["#begin", "document"]:
                data = {key: [] for key in col_to_id}
                utterance = [[] for x in range(12)]
                corefs = []
                token_idx_list = []     # nested list, each element holds the line_idx of this utterance's tokens
                utt_line_idx = []
                raw_text = []
                line_id = 0
            

            elif len(line) < 3:
                """ End of an utterance """
                if utterance != [[] for x in range(12)]:
                    data["word"].append(utterance[col_to_id["word"]])
                    data["speaker"].append(utterance[col_to_id["speaker"]])
                    data["corefs"].extend(corefs)
                    utterance = [[] for x in range(12)]
                    corefs = []
                    token_idx_list.append(utt_line_idx)
                    utt_line_idx= []

            elif line.split() == ["#end", "document"]:
                doc = Document(
                    token_idx=token_idx_list,
                    utterances=data["word"],
                    utt_corefs=data["corefs"],
                    speaker=data["speaker"],
                    genre=document_genre,
                    raw_text=raw_text,
                    file_path=input_file_path
                )
                documents.append(doc)
                
            else:
                """ Beginning of or within an utterance """
                utt_line_idx.append(line_id)
                splitted = re.split(r"[\s]+", line)
                splitted = splitted[:11] + [splitted[-2]]
                assert len(splitted) == 12

                utterance[col_to_id["word"]].append(
                    re.sub("\/", "", splitted[col_to_id["word"]].lower())
                )

                utterance[col_to_id["speaker"]
                    ] = splitted[col_to_id["speaker"]].lower()

                corefs.append(splitted[col_to_id["corefs"]])
            
            raw_text.append(line)
            line_id += 1

    return documents


def extract_beginnings(search_string):
    """ Extracts all beginnings from a coref row in a conll file. 
        Return empty list if nothing found. """
    result = [elem.strip('(') for elem in re.findall(r"\(\d+", search_string)]
    result = [int(elem) for elem in result]
    return result


def extract_endings(search_string):
    """ Extracts all endings from a coref row in a conll file. 
        Returns empty list if nothing found. 
    """
        
    result = [elem.strip(')') for elem in re.findall(r"\d+\)", search_string)]
    result = [int(elem) for elem in result]
    return result


def get_processed_docs_from_dir(directory):
    """Process all conll_files in a dir (and sub-dirs).
    Return list of document-objects."""
    directory = Path(directory)
    file_list = directory.glob("*_conll")
    file_list = [x for x in file_list if x.is_file() and ".swp" not in str(x)]

    documents = []
    print(f'Now reading in files from:\n{directory} ')
    for conll_file in file_list:
        documents.extend(read_conll_file(conll_file))
    return documents


################################################
#               Storing Data
#################################################

class Document:
    def __init__(self, token_idx, utterances, utt_corefs, speaker, genre, raw_text, file_path):
        self.text = [t for utt in utterances for t in utt]      # the whole doc as a list of tokens
        self.token_idx = token_idx      # nested list holding the line_idx of each token in the conll file
        self.utterances = utterances    # a list of all utterances
        self.utt_corefs = utt_corefs
        self.speaker = speaker          # a list with the speaker for each utterance
        self.genre = genre              # string
        self.raw_text = raw_text        # a list of all the lines of the conll file, needed for eval script
        self.tags = None                # predicted coreference tags in evaluation
        self.file_path = file_path
        self.spans = self._get_spans(utterances, speaker, raw_text)
        self.gold_spans = self._get_gold_spans(utt_corefs, raw_text)
        self.p_spans = None             # to hold predicted spans in evaluation
        
        # Upon creating the document: check if everything aligns
        if True:
            self.test_span_idx_and_gold_span_idx_align()
         
        if False:
            print(f'len(flatten(token_idx)) = {len(flatten(token_idx))}')
            print(f'len(self.text) = {len(self.text)}')
            print(f'len(utt_corefs) = {len(utt_corefs)}')    
            print(f'len(flatten(utterances)) = {len(flatten(utterances))}')     
            print(f'len(raw_text): {len(raw_text)}\n')



    def _get_spans(self, utterances, speaker, raw_text):
        """ Returns a list of all possible spans in that document. 
            No cross-sentence spans. 
            Max span-length L = 10 tokens. """
        span_list = []
        span_id = 0
        iter_id = 0
        for utt_id, utt in enumerate(utterances):
            L = 10
            for word_id, word in enumerate(utt):
                
                if len(utt) - word_id < 11:
                    L = len(utt) - word_id
                for l in range(1, L):
                    start = flatten(self.token_idx)[iter_id]
                    end = start + l
                    tokens = [line.split()[3].lower() for line in raw_text[start : end]]
                    span_list.append(
                        Span(id = span_id,
                             start=start,
                             end=end,
                             tokens=tokens,
                             speaker=speaker[utt_id]))
                    span_id += 1
                iter_id += 1

        return span_list


    def _get_gold_spans(self, utt_corefs, raw_text):
        """ Returns a list of all gold_spans in the document,
            sorted by start, end. 
            Give gold_spans the same ids as spans. 
        """
        gold_spans = []
        open_spans = []
        for i, elem in enumerate(utt_corefs):           # 1d list
            label_starts = extract_beginnings(elem)
            label_ends = extract_endings(elem)
            
            for label in label_starts:
                open_spans.append(Gold_Span(start=flatten(self.token_idx)[i], 
                                            label=label))

            for label in label_ends:
                for span in open_spans:
                    if span.label == label:
                        
                        # close span
                        span.end = flatten(self.token_idx)[i]+1
                        span.tokens = [line.split()[3].lower() for line in raw_text[span.start : span.end]]

                        gold_spans.append(span)
                        open_spans.remove(span)
        
        gold_spans = sorted(gold_spans, key=lambda s: (s.start, s.end))
        
        for g_span in gold_spans:
            for span in self.spans:
                if (g_span.start, g_span.end) == (span.start, span.end):
                    g_span.id = span.id
        
        for id, g_span in enumerate(gold_spans):
            g_span.yi = gold_spans[:id]

        return gold_spans


    def test_span_idx_and_gold_span_idx_align(self):
        """ When reading-in the files:
            Check if spans and gold_spans align.
            Also check if spans and gold_spans align with raw_text line numbers.
            If not, print but don't raise error.
        """
        for gold_span in self.gold_spans:
            if gold_span.id is not None:
                same_id_span = self.spans[gold_span.id]
                try:
                    assert gold_span.start == same_id_span.start
                    assert gold_span.end == same_id_span.end
                    assert gold_span.tokens == same_id_span.tokens
                    assert gold_span.tokens == [line.split()[3].lower() for line in 
                                                self.raw_text[gold_span.start : gold_span.end]]
                    
                    assert gold_span.label in extract_beginnings(self.raw_text[gold_span.start].split()[-1])
                    assert gold_span.label in extract_endings(self.raw_text[gold_span.end -1 ].split()[-1])
                    
                except AssertionError:
                    print(f'spans / gold_spans / raw_text do not align for this doc:')
                    print(f'{self.file_path}')
                    print('gold_span:')
                    print(f'{gold_span}')
                    print('span:')
                    print(f'{same_id_span}')
                    print('raw_text lines:')
                    print(f'{[line for line in self.raw_text[gold_span.start : gold_span.end]]}')
                    print(f'gold_span.label: {gold_span.label}')
                    print(f'extract_beginnings(self.raw_text[gold_span.start].split()[-1]): {extract_beginnings(self.raw_text[gold_span.start].split()[-1])}')
                    print(f'extract_endings(self.raw_text[gold_span.end].split()[-1]): {extract_endings(self.raw_text[gold_span.end].split()[-1])}')
                    print()
                    

    def __len__(self):
        return len(self.utterances)


    def __iter__(self):
        for i in range(len(self.utterances)):
            yield (self.utterances[i], self.speaker[i])


class Span:
    """ Holding span start_id, end_id, length, tokens. """
    def __init__(self, start, end=None, id=None, tokens=None, speaker=None, s_m=None):
        self.start = start          # id of start_token in the document text
        self.end = end              # id of end_token in the document_text
        self.id = id                # to index g_i after pruning the spans
        self.tokens = [t.lower() for t in tokens] if tokens else None
        self.length = len(tokens) if tokens else None
        self.speaker = speaker
        self.yi = None              # will hold a list of span objects preceding this span
        self.yi_idx = None          # will store the idx of each span in self.yi
        self.s_m = s_m              # mention score s_m
        self.att_weights = None
        

    def __repr__(self):
        return f'''start: {self.start}, end: {self.end}, id: {self.id}, speaker: {self.speaker}  {" ".join(self.tokens)}'''


class Gold_Span(Span):
    """ Span with additional label attribute. """
    def __init__(self, start, end=None, id=None, tokens=None, label=None):
        super().__init__(start, end, id, tokens)
        self.label = label

    def __repr__(self):
        return f'''start: {self.start}, end: {self.end},, id: {self.id}, label: {self.label}   {" ".join(self.tokens)}'''


class Dataset:
    def __init__(self, documents):
        self.data_dir = None            # train / dev / or test dir
        self.documents = documents      # a list of document-objects
        self.predicted_docs = None      # predicted documents with conll eval script


    def get_docs_in_random_order(self):
        random_doc_idx = list(range(len(self.documents)))
        shuffle(random_doc_idx)
        for id in random_doc_idx:
            yield self.documents[id]
        

    def get_genre_set(self):
        """ Returns a set of all genres in the dataset.
            Used to instanciate Genre_Embeddings. """
        genre_set = [doc.genre for doc in self.documents]
        return set(genre_set)


    def __len__(self):
        return len(self.documents)


class Vocab:
    def __init__(self, datasets, token_set=None):
        self.token_set = self._get_token_set(datasets) if token_set == None else token_set
        self._itos = {i: token for i, token in enumerate(self.token_set)}           # i+1: token
        self._stoi = {token: i for i, token in enumerate(self.token_set)}           # token: i+1
        self.oov_id = 0
        self.oov_token = 'unk'
    
    def stoi(self, word):
        try:
            out = self._stoi[word]
        except KeyError:
            out = self.oov_id
        return out

    def itos(self, id):
        try:
            out = self._itos[id]
        except KeyError:
            out = self.oov_token
        return out

    def __iter__(self):
        yield from self._stoi.keys()

    def __len__(self):
        return len(self._itos.keys())

    def _get_token_set(self, datasets):
        """ Get a set of all tokens in the dataset, 
            so we load only the word_embeddings that 
            are in the dataset. """
        tokens = []
        for ds in (datasets):
            for doc in ds.documents:
                for sentence in doc.utterances:
                    for token in sentence:
                        tokens.append(token)
        return set(tokens)


################################################
#                 Features
#################################################

class Pretrained_Embeddings:
    """ Class to hold the pretrained embeddings for Glove and Turian. """

    def __init__(self, pretrained_embeds_file, vocab):
        self.pretrained_embeds_file = pretrained_embeds_file
        self.vocab = vocab
        self.word2vec = self.build_word2vec()

    def build_word2vec(self):
        """Load a dict from words to pretrained embeddings.
        Normalize embeddings to unit vectors."""
        word2vec = {}
        with open(self.pretrained_embeds_file) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                
                try:
                    word = line.split()[0]
                    vec = [float(digit) for digit in line.split()[1:]]
                except:
                    pass
                if word in self.vocab.token_set:
                    word2vec[word] = F.normalize(torch.Tensor(vec), p=2, dim=0)
        self.embed_dim = len(vec)
        return word2vec

    def to_weights(self):
        """Returns a weight matrix of the word2vec dict.
        Out-of-vocab words are zero-vectors.
        shape = [embed_dim, len(train_vocab)]"""
        return torch.stack(
            tensors=[
                self.word2vec[self.vocab._itos[id]]
                if self.vocab._itos[id] in self.word2vec.keys()
                else torch.ones(self.embed_dim)
                for id in vocab._itos.keys()],
            dim=1,
            ).T


class Genre_Embedding(nn.Module):
    def __init__(self, genre_set, embed_dim=20):
        super().__init__()
        self.genre_to_idx = {genre: torch.tensor(
            i) for i, genre in enumerate(genre_set)}
        self.oov_token = torch.tensor(len(genre_set))
        self.embeds = nn.Embedding(len(genre_set) + 1, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def stoi(self, genre):
        if genre in self.genre_to_idx.keys():
            out = self.genre_to_idx[genre]
        else:
            out = self.oov_token
        return out

    def forward(self, genres):
        genre_tensor = to_cuda(torch.stack([self.stoi(g) for g in genres]))
        out = self.embeds(genre_tensor)
        out = self.dropout(out)
        return to_cuda(out)


class Speaker_Embedding(nn.Module):
    """ To encode the three possibilities:
        span_i, span_j have the same speaker,
        don't have the same speaker,
        speaker unknown.
    """
    def __init__(self, embed_dim=20):
        super().__init__()
        self.embeds = nn.Embedding(3, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, speakers):
        """ Input is a list of strings (speakers)
            Output is [bs, embed_dim] """
        speakers_tensor = to_cuda(torch.stack(speakers))
        out = self.embeds(speakers_tensor)
        out = self.dropout(out)
        return to_cuda(out)


class Span_Length_Embedding(nn.Module):
    """ Embedding for span_length between 1 and 10 
    """

    def __init__(self):
        super().__init__()
        self.embeds = nn.Embedding(10, 20)
        self.dropout = nn.Dropout(0.2)

    def forward(self, length):
        """ Input is either a scalar tensor or a tensor of batch_size. """
        out = self.embeds(length)
        out = self.dropout(out)
        return to_cuda(out)


class Span_Distance_Embedding(nn.Module):
    """ Embedding for distance between span_1 and span_2 
    """

    def __init__(self):
        super().__init__()
        self.bins = [1, 2,3,4,5,8,16,32,64]
        self.embeds = nn.Embedding(len(self.bins), 20)
        self.dropout = nn.Dropout(0.2)

    def dist_to_bin_id(self, dist):
        for id, bin in enumerate(self.bins):
            if id < len(self.bins)-1:
                if dist in range(self.bins[id], self.bins[id+1]):
                    res = id
                    break
            else:
                res = id
        return torch.tensor(res)

    def forward(self, distances):
        """ Input is a list of distances """
        distances = to_cuda(torch.stack([self.dist_to_bin_id(d) for d in distances]))
        out = self.embeds(distances)
        oot = self.dropout(out)
        return to_cuda(out)
        #bin_id = self.dist_to_bin_id(distance)
        #return self.embeds(torch.tensor(bin_id))


################################################
#                 Helper Functoins
#################################################


def assert_span_tokens_equal_raw_text_tokens(span, doc):
    raw_text_token = [elem.split()[3].lower() for elem in doc.raw_text[span.start: span.end]]
    try:
        assert span.tokens == raw_text_token
    except:
        print(span)
        print(doc.file_path)
        for line in doc.raw_text[span.start : span.end + 1]:
            print(line)

def show_cluster_paper(spans, probs, doc, sav_file='results_my.txt'):
    """ Outputs the most likely cluster inference-cluster-configuration. 
        An antecedent yi for span_i is chosen iff p(yi) > p(epsilon). """

    graph = nx.Graph()
    
    corefs_chosen = 0
    
    # choose for each span its antecedent if its prob > p(epsilon)
    for i, span in enumerate(spans):
        if i > 0:
            mention_prob = probs[i][len(span.yi_idx)]
            
            # if some antecedent-prob is higher than epsilon-prob, draw connection:
            if torch.max(probs[i][:len(span.yi_idx)]) > mention_prob:
                antecedent_id = torch.argmax(probs[i][:len(span.yi_idx)]).item()
                link = spans[antecedent_id]
                graph.add_edge((span.start, span.end - 1), (link.start, link.end - 1))
                corefs_chosen += 1
                
                
                print(f'connected span {span.tokens} with link {link.tokens}')
                assert_span_tokens_equal_raw_text_tokens(span, doc)
                assert_span_tokens_equal_raw_text_tokens(link, doc)

    clusters = list(nx.connected_components(graph))

    with open(sav_file, 'a') as f:
        f.write('Clustering paper:\n')
        for elem in clusters:
            f.write(f'{elem}\n')
    
    return clusters, corefs_chosen


def show_cluster_repo(spans, probs, doc, att_weights=None, sav_file='results_my.txt'):
    """ Outputs a coref cluster (e.g. a list of sets) 
        given spans and probs. """
    
    print(doc.file_path)
    
    graph = nx.Graph()
    
    # Cluster found coreference links
    for i, span in enumerate(spans):

        # Loss implicitly pushes coref links above 0, rest below 0
        # found_corefs = elem if antecedent_prob > mention_prob
        found_corefs = [idx
                        for idx, _ in enumerate(span.yi_idx)
                        if probs[i, idx] > probs[i, len(span.yi_idx)]]
    
        # If we have any
        if any(found_corefs):
            
            # Add edges between all spans in the cluster
            for coref_idx in found_corefs:
                link = spans[coref_idx]
                graph.add_edge((span.start, span.end - 1), (link.start, link.end - 1))
                
                print(f"connected span {span.tokens} with link {link.tokens}, att_weights: {span.att_weights}, {link.att_weights}")
                assert_span_tokens_equal_raw_text_tokens(span, doc)
                assert_span_tokens_equal_raw_text_tokens(link, doc)             

    clusters = list(nx.connected_components(graph))
                
    with open(sav_file, 'a') as f:
        f.write('Clustering repo:\n')
        for elem in clusters:
            f.write(f'{elem}\n')
    
    return clusters, len(found_corefs)


def create_file_if_not_exists(file):
    if not os.path.exists(file):
        f = open(file, "w+")
        f.close()


def clean_file_content(file):
    if os.path.exists(file):
        f = open(file, "r+")
        f.seek(0)
        f.truncate()
        f.close()    


def pack(tensors):
    """ Pack list of tensors, provide reorder indexes """

    # Get sizes
    sizes = [t.shape[0] for t in tensors]

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1]

    # Resort the tensor accordingly
    sorted_tensors = [tensors[i] for i in size_sort]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_sequence(sorted_tensors)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)
    #size_sort = torch.argsort(sizes, dim=-1, descending=False) → LongTensor

    return packed, reorder


def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes


def pad_and_stack_for_att(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = [F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)]
    
    #for elem in padded[:100]:
    #    print(elem.shape)

    padded = torch.stack(padded, dim = 0)

    return padded, sizes


def sort_out_crossing_spans(sorted_spans):
    """ Remove overlapping spans:
        Accept spans in decreasing order of mention scores.
        Sort out span_i if it fulfills one of the conditions compared
        to a previously accepted span_j:
        si.i1 < sj.i1 <= si.i2 < sj.i2   OR
        sj.i1 < si.i1 <= sj.i2 < si.i2 """ 
        
    non_overlapping = []
    for i, span_i in enumerate(sorted_spans):
        taken = False
        for j, span_j in enumerate(non_overlapping):
            if span_i.start < span_j.start <= span_i.end < span_j.end:
                taken = True
                break
            if span_j.start < span_i.start <= span_j.end < span_i.end:
                taken = True
                break
        if not taken:
            non_overlapping.append(span_i)
    return non_overlapping


def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack.
    Output is a list of tensors. """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped


def prune(spans, T, lbda=0.4):
    """ Input: Spans with mention scores.
        1) Sort out crossing spans.
        2) Keep only 40 % spans with the highest mention scores.
        Output: Spans, sorted by start_id, end_id
    """
    sorted_spans = sorted(spans, key=lambda s: s.s_m, reverse=True)
    non_crossing_spans = sort_out_crossing_spans(sorted_spans)
    non_crossing_spans = non_crossing_spans[:int(T*lbda)]
    sorted_spans = sorted(spans, key=lambda s: s.s_m, reverse=True)
    pruned_spans = sorted(non_crossing_spans, key=lambda s: (s.start, s.end))
    return pruned_spans


def speaker_label(s1, s2):
    """ Compute if two spans have the same speaker or not """
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return to_cuda(idx)


def to_cuda(x):
    """ Put tensor to cuda """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def truncate_document(doc, num_keep_sents=50):
    """ If the document has more than 50 sentences:
        randomly choose a window of 50 sents and 
        cut out everything before and after. 
    """
    if len(doc) > num_keep_sents:
        
        # start and end indices for 2-dimensional lists
        end = random.randint(num_keep_sents, len(doc))
        start = end - num_keep_sents
        
        # get start and end indices for 1-dimensional lists
        start_at = len(flatten(doc.token_idx[:start]))
        keep_until = start_at + len(flatten(doc.token_idx[start : end]))
    
        start_raw = len(doc.utt_corefs[:start_at]) + start + 1
        end_raw = len(doc.utt_corefs[:keep_until]) + end + 1
        
        # Create new raw text
        new_raw_text = []
        new_raw_text.extend(doc.raw_text[start_raw : end_raw])
        new_raw_text.extend('\n')
        #for line in new_raw_text:
        #    print(line)

        # Create new mapping words / sentences to raw_text line_idx
        token_idx_list = []
        id = 0
        for _, sent in enumerate(doc.token_idx[start : end]):
            this_sent = []
            for _ in sent:
                this_sent.append(id)
                id += 1
            token_idx_list.append(this_sent)
            id += 1

        if False:
            # token_idx_list should have the same shape, only starting from 0
            for i, sublist in enumerate(doc.token_idx[start : end]):
                print(len(doc.token_idx[start : end][i]), doc.token_idx[start : end][i])
                print(len(token_idx_list[i]), token_idx_list[i])

        if False:
            for i in range(len(flatten(doc.utterances[start : end]))):
                print(flatten(doc.utterances[start : end])[i])
                print(flatten(token_idx_list)[i])
                print(new_raw_text[i])
                print()
                
        
        new_doc = Document(token_idx = token_idx_list,
                        utterances = doc.utterances[start : end],
                        utt_corefs = doc.utt_corefs[start_at : keep_until],
                        speaker = doc.speaker,
                        genre = doc.genre,
                        raw_text = new_raw_text,
                        file_path = doc.file_path)
        
        return new_doc
    return doc


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def generic_loss_test(test_model, some_output, print_=False, verbose=False):
    """ Input: model, model.forward()
        Output: A plot showing gradient flow. 
        This functions purpose is to test if pytorch's 
        autograd graph works correctly. 
    """
    test_model.train()
    target = torch.randn(size=(some_output.shape))
    target.requires_grad = True
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=[p for p in test_model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    loss = loss_fn(some_output, target)
    print('Before loss.backward():')
    print(list(test_model.parameters())[0].grad )
    loss.backward()
    if print_:
        plot_grad_flow(test_model.named_parameters())
    print('After loss.backward():')
    print(list(test_model.parameters())[0].grad )
    if verbose:
        for name, param in test_model.named_parameters():
            print(name, param.grad)

    optimizer.step()


def analyze_span_lengths(spans):
    """ Get the distribution of lengths of predicted spans. 
        Returns a dict where keys are span_lengths and values 
        are how often spans with this length were predicted.
    """
    lengths = defaultdict(list)
    
    for span in spans:
        lengths[len(span.tokens)].append(1)
    
    eval_dict = {k : len(v) for k,v in lengths.items()}
    
    return eval_dict

################################################
#                 Neural Nets
#################################################


class FFNN(nn.Module):
    """ 
    FFNN_alpha  : used to compute attention-weights
    FFNN_m      : used to compute mention scores
    FFNN_a      : used to compute attention scores AND (?) antecedent scores s_a(i,j)
    """

    def __init__(self, in_size, out_size, n_neurons=150):
        super().__init__()
        self.fc1 = nn.Linear(in_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, out_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #x = F.elu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #x = F.elu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Character_CNN(nn.Module):
    """ Build a character-level representation of a sentence. 
        If dimension doesn't fit for concatenation with word_embeds,
        maybe just insert a FFNN after char_cnn to get the dimensionality right."""
    def __init__(self, embed_dim, n_cnn_filters, pad_len=30):
        super().__init__()
        self.alphabet = self._get_alphabet()
        self.char2id = {char : id for id, char in enumerate(self.alphabet)}
        self.pad_len = pad_len

        self.embeds = nn.Embedding(num_embeddings=len(self.alphabet),
                                   embedding_dim=embed_dim,
                                   padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=pad_len, 
                               out_channels=n_cnn_filters, 
                               kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=pad_len, 
                               out_channels=n_cnn_filters, 
                               kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=pad_len, 
                               out_channels=n_cnn_filters, 
                               kernel_size=5)

    def _get_alphabet(self):
        """ Include all the english ASCI symbols except upper case letters. """
        #alphabet = [chr(i) for i in range(32, 127)]
        alphabet = [chr(i) for i in range(32, 65)]
        alphabet.extend([chr(i) for i in range(91, 127)])
        return alphabet
        
    def forward(self, input):
        sent = self.sent_to_tensor_odf_idx(input)
        embedded = self.embeds(sent)
        c1 = self.conv1(embedded)                   # [seq_len, n_cnn_filters, 6]
        c2 = self.conv2(embedded)                   # [seq_len, n_cnn_filters, 5]
        c3 = self.conv3(embedded)                   # [seq_len, n_cnn_filters, 4]
        out = torch.cat((c1,c2,c3), dim=2)          # [seq_len, n_cnn_filters, 15]
        out = F.max_pool1d(out, out.shape[2]).view(out.shape[0], out.shape[1]) # [seq_len, n_cnn_filters]
        return out

    def sent_to_tensor_odf_idx(self, sent):
        """ Returns a tensor of shape [num_words, pad_len] of char_idx. """
        return torch.stack([self.word2id(token) for token in sent])

    def word2id(self, word):
        """ Returns a 1d tensor of char_idx of a token, padded to pad_len. """
        try:
            res = [self.char2id[char] for char in word]
        except:
            res = [0 for char in word]      # zero-tensor in unlikely case of oov
        res = res + (self.pad_len - len(res)) * [0]
        return to_cuda(torch.tensor(res[:self.pad_len]))


class Mention_Scorer(nn.Module):
    """ Takes as input the span representation g_i and returns 
        the mention score s_m: 
        s_m(i) = W_m * FFNN_m(g_i)
        Also prunes prunes the spans and stores the mention scores therein.
    """
    def __init__(self, in_size, out_size=1):
        super().__init__()
        self.scorer = FFNN(in_size, out_size)

    def forward(self, g_i, doc):
        spans = doc.spans
        s_m = self.scorer(g_i)

        for id, score in enumerate(s_m.detach()):
            spans[id].s_m = score
        
        # sort out crossing spans, 
        # prune to 40 % highest mention score,
        # sort in order of appearance in text
        pruned_spans = prune(spans, T=len(doc.text))

        # Store each spans possible antecedents in the span object
        for i, span_i in enumerate(pruned_spans):
            # update antecedent canditates
            span_i.yi = pruned_spans[max(0, i-250) : i]        

        return pruned_spans, s_m.squeeze() # [num_possible_spans]


class Antecedent_Scorer(nn.Module):
    def __init__(self, in_size, genre_embeds, out_size=1):
        super().__init__()
        self.ffnn = FFNN(in_size, out_size)
        self.span_dist_embeds = Span_Distance_Embedding()
        self.genre_embeds = genre_embeds
        self.speaker_embeds = Speaker_Embedding()
    

    def forward(self, p_spans, g_i, s_m, doc):
        
        # results in n * (n+1) / 2 comparisons between span_i and span_j
        mention_ids, antecedent_ids, \
            distances, genres, speakers = zip(*[(span_i.id, span_j.id,
                                                span_i.end - span_j.start, doc.genre,
                                                speaker_label(span_i, span_j))
                                                for i, span_i in enumerate(p_spans)
                                                for j , span_j in enumerate(span_i.yi)])

        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        phi_ij = torch.cat((self.span_dist_embeds(distances),
                            self.genre_embeds(genres),
                            self.speaker_embeds(speakers)), dim=1)

        vec_i = torch.index_select(g_i, 0, mention_ids)
        vec_j = torch.index_select(g_i, 0, antecedent_ids)
        
        sm_i = torch.index_select(s_m, 0, mention_ids)
        sm_j = torch.index_select(s_m, 0, antecedent_ids)
        
        s_ij = self.ffnn(torch.cat([vec_i, vec_j, vec_j * vec_j, phi_ij], dim=1)).squeeze()
        
        coref_scores = torch.sum(torch.stack((sm_i, sm_j, s_ij)), dim=0)

        antecedent_idx = [len(s.yi) for s in p_spans if len(s.yi)]
        
        for i, span in enumerate(p_spans):
            span.yi_idx = [s.id for s in span.yi]

        split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_cuda(torch.tensor([0.])).requires_grad_()
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensor, dim=0) for tensor in with_epsilon]
        
        # pad probs to a single tensor
        # [num_p_spans, num_p_spans]
        probs = nn.utils.rnn.pad_sequence(probs,batch_first=True, padding_value=1000)    # 1000
        return p_spans, probs


class Word_Vec_Cat(nn.Module):
    def __init__(self,
                 vocab,
                 glove_weights,
                 turian_weights,
                 n_cnn_filters=50):
        super().__init__()
        self.vocab = vocab

        self.glove_embedding = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove_embedding.weight.data.copy_(to_cuda(glove_weights))
        self.glove_embedding.weight.requires_grad = False


        self.turian_embedding = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        self.turian_embedding.weight.data.copy_(to_cuda(turian_weights))
        self.turian_embedding.weight.requires_grad = False

        self.char_cnn = Character_CNN(
            embed_dim=8, n_cnn_filters=n_cnn_filters, pad_len=30)

        self.dropout = nn.Dropout(0.5)

    def embed_tokens(self, tokens):
        """ Returns the concatenated glove, turian, char_cnn embeds for 
            a sentence [seq_len, 400]
        """
        sent_tensor = to_cuda(torch.stack(
            [torch.tensor(self.vocab.stoi(t)) for t in tokens]))  # [seq_len]
        glove_out = self.glove_embedding(sent_tensor)  # [seq_len, 300]
        turian_out = self.turian_embedding(sent_tensor)  # [seq_len, 50]
        char_cnn_out = self.char_cnn.forward(tokens)  # [seq_len, 50]
        concat = torch.cat((glove_out, turian_out, char_cnn_out), dim=1) # [seq_len, 400]
        return concat

  
    def embed_batch(self, batch):
        """ Input is a list of token-lists.
            Output is a batched embed_tensor [bs, pad_len, 400] """
        embeds = [self.dropout(to_cuda(self.embed_tokens(sent))) for sent in batch]
        #padded, sizes = pad_and_stack(embeds)  # [bs, pad_len, 400]   tensor, padded with zeros
        #return padded, sizes
        return embeds


    def forward(self, doc):
        """ Input:  a Document object
            Output: a tensor of word_embeddings     [bs, pad_len, 400]
        """
        span_tokens = [span.tokens for span in doc.spans]       
        #padded, sizes = self.embed_batch(span_tokens)       # [bs, pad_len, 400]
        #return padded, sizes
        embeds = self.embed_batch(span_tokens)
        return embeds


def get_last_seq_items(packed, lengths): # mine
    """ To get the last word of each sequence out of a packed-sequence object. 
        Batched. 
    """
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    ))
    sorted_lengths = lengths[packed.sorted_indices]
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


class Test_Encoder(nn.Module):
    """ Input:  tensor of word_embeddings    [bs, pad_len, 400]
        Applies lstm, applies attention-averaging over lstm_output,
        build span_length_embeddings, concat all that and output.
    """

    def __init__(self, config, h_dim=200):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=400,      #lstm_input_size,
            hidden_size=200,     #h_dim,
            num_layers=1,
            bidirectional=True,
        )
        
        self.ffnn_a = FFNN(400,1)
        
        self.config = config
        
        self.dropout = nn.Dropout(0.2)
        
        self.span_length_embeds = Span_Length_Embedding()        

    def forward(self, embeds, doc):
        """ Input:  tensor or word_embeddings
            Output: 
                g_i: cat(lstm_out[first_word],      [bs, 1220]
                         lstm_out[last_word], 
                         weighted_avg(lstm_out), 
                         phi_ij)
                att_weights:                        [bs, seq_len]
        """

        x_lens = torch.stack([torch.tensor(s.shape[0]) for s in embeds]).squeeze()
        padded_embeds = to_cuda(nn.utils.rnn.pad_sequence(embeds, batch_first=True))
        
        x_packed = to_cuda(pack_padded_sequence(padded_embeds, x_lens, batch_first=True, enforce_sorted=False))
        output_packed, _ = self.lstm(x_packed)
        
        lstm_last_items = to_cuda(get_last_seq_items(output_packed, x_lens))     # [bs, 400]
        
        lstm_out_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        lstm_out_padded = self.dropout(lstm_out_padded)
        
        
        
        ############################
        weighed_embeds = []
        for i, size in enumerate(output_lengths):
            a_t = to_cuda(self.ffnn_a(lstm_out_padded[i][:size]))
            
            softmaxed_att = F.softmax(a_t.squeeze(), dim=0).unsqueeze(-1)
            weighed_embed_i = torch.sum(torch.mul(softmaxed_att, embeds[i][:size]), dim=0)     # [400]
            weighed_embeds.append(weighed_embed_i)
            doc.spans[i].att_weights = softmaxed_att.squeeze().detach().cpu().numpy()            
            
        ############################
        """
        a_t = to_cuda(self.ffnn_a(lstm_out_padded))              # [bs, L, 1]
        
        weighed_embeds = []
        for i, size in enumerate(output_lengths):       # iterate over every span
            softmaxed_att = F.softmax(a_t[i][:size].squeeze(), dim=0).unsqueeze(-1)
            weighed_embed_i = torch.sum(torch.mul(softmaxed_att, embeds[i][:size]), dim=0)     # [400]
            weighed_embeds.append(weighed_embed_i)
            doc.spans[i].att_weights = softmaxed_att.squeeze().detach().cpu().numpy()
        """
        ############################

        weighed_embeds = torch.stack(weighed_embeds)                # [bs, 400]
        
        phi_i = torch.stack([self.span_length_embeds(to_cuda(s.clone().detach())) for s in output_lengths]) # [bs, 20]

        g_i = torch.cat((lstm_out_padded[:,0,:],    # [bs, 400]     first word of sentenc
                         lstm_last_items,           # [bs, 400]     last word of sentence
                         weighed_embeds,              # [bs, 400]     attention-weighed sum of word-embeddings
                         phi_i),                    # [bs, 20]
                        dim=1)                      # = [bs, 1220]
        
        return g_i, a_t


class New_Coref(nn.Module):
    def __init__(self,
                 config,
                 vocab, 
                 glove_weights, 
                 turian_weights,
                 genre_embeds,
                 s_m_in_size=1220,
                 s_a_in_size=3720
                 ):
        super().__init__()
        self.config = config
        self.word_vec_cat = Word_Vec_Cat(vocab, glove_weights, turian_weights)
        self.span_encoder = Test_Encoder(config)
        self.mention_scorer = Mention_Scorer(s_m_in_size)
        self.antecedent_scorer = Antecedent_Scorer(s_a_in_size, genre_embeds)
        

    def forward(self, doc):
        embeds = self.word_vec_cat(doc)
        g_i, att_weights = self.span_encoder(embeds, doc)
        p_spans, s_m = self.mention_scorer(g_i, doc)
        p_spans, probs = self.antecedent_scorer(p_spans, g_i, s_m, doc)
        return p_spans, probs, att_weights


def my_custom_loss(spans, input, target):
    eps = 1e-8
    sum_1 = []
    sum_2 = []
    for i, span in enumerate(spans):
        if i > 1:   # got infinity at i == 0... should think of a better solution than this at some time.
            consider_input = input[i][:len(span.yi_idx)]
            consider_target = target[i][:len(span.yi_idx)]
            gold_scores = consider_input + torch.log(consider_target.double())  # [num_of_this_spans_antecedents]
            exp = torch.exp(gold_scores)                            # [num_of_this_spans_antecedents]
            sum_1.append(torch.sum(exp).clamp_(eps, 1 - eps))
            
            exp_2 = torch.exp(consider_input)
            sum_2.append(torch.sum(exp_2))
    sum_1 = torch.stack(sum_1)                                  # [num_spans]
    sum_2 = torch.stack(sum_2)
    log = torch.log(sum_1)                                      # [num_spans]

    log_norm = torch.log(sum_2)

    loss = log_norm - log                                       # [num_spans]
    summed_loss = torch.sum(loss)                               # scalar
    print(f'my loss: {summed_loss}')
    return summed_loss


class My_Loss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, config, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.config = config

    def forward(self, spans, input, target):
        """ Input: Probabilities                        [num_of_mentions, num_of_mentions]
            target: gold_indices of antecedents.        [num_of_mentions, num_of_mentions]
                    1 where true antecedent was found,
                    0 everywhere else.
        """

        print(f'probs where gold_indices is non-zero')
        print(input[torch.nonzero(target, as_tuple=True)])

        my_loss = my_custom_loss(spans, input, target)
        
        eps = 1e-8

        
        mul = torch.mul(input, target) # all values zero except where mentions or coref-links are found
        sum_1 = torch.sum(mul, dim=1).clamp_(eps, 1 - eps)
        log = torch.log(sum_1)
        repo_loss = torch.sum(log) * -1

        print(f'repo loss: {repo_loss}')
        if config['use_my_loss']:
            return my_loss
        if not config['use_my_loss']:
            return repo_loss


def get_gold_mentions_gold_corefs(gold_spans):
    """ Returns gold_corefs, len(gold_corefs), gold_mentions, len(gold_mentions)
    """

    gold_links = defaultdict(list)
    for gs in gold_spans:
        gold_links[gs.label].append(gs.id)

    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                            for gold in gold_links.values()])

    return gold_corefs, len(gold_corefs)


def get_gold_indices(probs, spans, gold_spans):
    """ Build target-tensor for loss-computation.
        Compare which of the predicted spans are actually in the gold_spans.
        Set gold_indices to 1 at indexes where mention or antecedent is found.
    """
    print(f'len(spans): {len(spans)}   probs.shape: {probs.shape}')
    
    mentions_found, corefs_found  = 0, 0
    gold_corefs, total_corefs = get_gold_mentions_gold_corefs(gold_spans)
    gold_indices = to_cuda(torch.zeros_like(probs))

    # keep track of span.id that are correctly identified mentions
    correct_mention_idx = []
    for id, span in enumerate(spans):
        # if one of the found mentions is in the gold_mentions            
        for gold_span in gold_spans:
            if span.id == gold_span.id:
                #print(gold_span)
                mentions_found += 1
                correct_mention_idx.append(span.id)

    # for the found mentions: are some of them coreference-linked in gold data?
    my_golds = list(combinations(correct_mention_idx, 2))
    linked_spans_found = []

    for i, link in enumerate(gold_corefs):
        if link in my_golds:

            # now I must find the correct ids of span_j and span_i in probs
            # to update probs at that position
            for id_i, span_i in enumerate(spans):           # can range up to 500-600
                for id_j, span_j in enumerate(span.yi):     # span.yi holds max 250 span objects
                    if (span_j.id, span_i.id) == link:
                        gold_indices[id_i, id_j] = 1
                        print(f'Coreferent: {" ".join(span_j.tokens)} /// {" ".join(span_i.tokens)}')
                        if span_i not in linked_spans_found:
                            linked_spans_found.append(span_i)
                        if span_j not in linked_spans_found:
                            linked_spans_found.append(span_j)                            
                        corefs_found += 1

    # mention correctly found but no antecedents found.
    # set gold_indices at this position to 1
    for id, span in enumerate(spans):
        if span.id in correct_mention_idx and span not in linked_spans_found:
            gold_indices[id, len(span.yi_idx)] = 1
            print(f'Mention: {" ".join(span.tokens)}')
    
    return (gold_indices, mentions_found, len(gold_spans),
            corefs_found, total_corefs, gold_corefs)


class Trainer:
    def __init__(self, model, datasets, config):
        self.model = to_cuda(model)
        self.ds_train = datasets[0]
        self.ds_dev = datasets[1]
        self.ds_test = datasets[2]
        
        self.config = config
        
        self.optimizer = torch.optim.Adam(params=[p for p in new_coref.parameters()
                                            if p.requires_grad],
                                    lr=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=100,
                                                    gamma=0.001)

        self.loss_fn = My_Loss(config)


    def train(self):
        for epoch in tqdm(range(self.config['train_for_x_epochs'])):
            print(f'\nEpoch: {epoch}\n')

            loss_mean_epoch, mentions_found_percent, corefs_found_percent, \
                corefs_chosen_percent = self.run_training_epoch()

            with open('results_my.txt', 'a') as f:
                f.write('##################################################################\n')
                f.write(f'Epoch: {epoch}')
                f.write(f'loss_mean_epoch = {str(loss_mean_epoch)}\n')
                f.write(f'mentionds_found_percent = {str(mentions_found_percent)}\n')
                f.write(f'corefs_found_percent: {str(corefs_found_percent)}\n')
                f.write(f'corefs_chosen_percent: {str(corefs_chosen_percent)}\n')
                f.write('\n\n\n')
            
            df.loc[epoch] = [loss_mean_epoch, mentions_found_percent,
                             corefs_found_percent, corefs_chosen_percent]
            df.to_csv('train_metrics_my.csv')
            
            if epoch % self.config['eval_every_x_epochs'] == 0:
                results = self.evaluate(dataset=self.ds_dev,
                                        sample_subset=True)          

            if epoch % self.config['save_model_every_x_epochs'] == 0:
                self.save_model(savepath=str(Path(__file__).parent.joinpath('model_savs').joinpath(str(datetime.now()))))


    def run_training_epoch(self):
        """ Run a single training epoch. Return performance metrics. """
        self.model.train()
        
        losses_epoch = []
        mentions_found_epoch = []
        mentions_total_epoch = []
        corefs_found_epoch = []
        corefs_total_epoch = []
        corefs_chosen_epooch = []
        corefs_gold_total_epoch = []

        for _, doc in tqdm(enumerate(random.sample(self.ds_train.documents, 
                                                   min(num_docs_for_train, 
                                                       len(self.ds_train.documents))))):

            loss, mentions_found, \
                total_mentions, corefs_found, \
                    total_corefs, corefs_chosen, \
                        total_gold_corefs = self.train_document(doc)

            losses_epoch.append(loss)
            mentions_found_epoch.append(mentions_found)
            mentions_total_epoch.append(total_mentions)
            corefs_found_epoch.append(corefs_found)
            corefs_total_epoch.append(total_corefs)
            corefs_chosen_epooch.append(corefs_chosen)
            corefs_gold_total_epoch.append(total_gold_corefs)
            
        self.scheduler.step()
        
        return np.mean(losses_epoch), \
            np.sum(mentions_found_epoch) / np.sum(mentions_total_epoch), \
            np.sum(corefs_found_epoch) / np.sum(corefs_total_epoch), \
            np.sum(corefs_chosen_epooch) / np.sum(corefs_gold_total_epoch)


    def train_document(self, doc):
        self.optimizer.zero_grad()
        
        trunc_doc = truncate_document(doc, num_keep_sents=50)
        
        spans, probs, att_weights = self.model(trunc_doc)
        
        gold_indices, mentions_found, total_mentions, \
        corefs_found, total_corefs, \
            gold_corefs = get_gold_indices(probs, spans, trunc_doc.gold_spans)

        #_, num_corefs_chosen = show_cluster_paper(spans, probs, trunc_doc)
        _, num_corefs_chosen = show_cluster_repo(spans, probs, trunc_doc)    
        
        loss = self.loss_fn(spans, probs, gold_indices)

        # backpropagate only if gradients are to be expected
        if not (mentions_found == 0 and corefs_found == 0):
            start = time.time()
            loss.backward()
            self.optimizer.step()
            end = time.time()
            print(f'%%%%%%%%%%%%%% Time taken for backprop: {end - start}')
        
        # Write the results out for later viewing
        with open('results_my.txt', 'a') as f:
            f.write(f'loss = {loss.item()}\n')
            f.write(f'mentionds_found = {mentions_found}\n')
            f.write(f'toral_mentions: = {total_mentions}\n')
            f.write(f'corefs_found: {corefs_found}\n')
            f.write(f'total_corefs: {total_corefs}\n')
            f.write(f'corefs_chosen: {num_corefs_chosen}\n\n')
            f.write('\n\n\n')            
        
        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, num_corefs_chosen, len(gold_corefs))
    

    def evaluate(self, dataset, sample_subset=True, comment_string=None):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        method = 'single'
        print('Evaluating on validation corpus...\n\n')
        if sample_subset:
            sampled_eval_docs = random.sample(dataset.documents, self.config['num_docs_for_eval'])
        if not sample_subset:
            sampled_eval_docs = dataset.documents
        eval_dataset = Dataset(sampled_eval_docs)
        predicted_docs = []
        predicted_spans = []
        for _, doc in tqdm(enumerate(sampled_eval_docs)):

            pred_doc, spans, probs, att_weights = self.predict(doc, method)
            pred_doc.p_spans = spans
            predicted_docs.append(pred_doc)
            predicted_spans.extend(spans)
            
            predicted_spans_lengths = analyze_span_lengths()

        eval_dataset.predicted_docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(eval_dataset, method)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([self.config['eval_script_path'], 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        results = str(stdout).split('TOTALS')[-1]
        
        # Write the results out for later viewing
        eval_results_file = 'preds/eval_my_' + method + '.txt'
        create_file_if_not_exists(eval_results_file)
        
        with open(eval_results_file, 'a') as f:
            if comment_string:
                f.write(comment_string + method)
            f.write(results)
            f.write(f'Distribution of lengths of predicted spans:\n')
            for k,v in predicted_spans_lengths.items():
                f.write(f"{k}: {v}\n")
            f.write('\n\n\n')

        return results


    def check_if_span_idx_and_raw_text_align(self, spans, doc, verbose=False):
        """ If span.tokens does not align with raw text: 
            Print error message, but continue code.
            This is necessary because span.tokens must align with doc.raw_text
            in order for the official evaluation script to work properly.
        """
        for i, span in enumerate(spans):
            raw_text_token = [elem.split()[3].lower() for elem in doc.raw_text[span.start: span.end]]
            
            if verbose:
                print(' '.join(span.tokens))
                for elem in raw_text_token:
                    print(elem)
                print()
            try:
                assert span.tokens == raw_text_token
            except AssertionError:
                print('Assertion Error: span.tokens != doc.raw_text.tokens[span.start : span.end]')
                print(doc.file_path)
                print(len(flatten(doc.token_idx)))
                print(f'span.start = {span.start}, span.end = {span.end}')
                print(f'len(doc.raw_text): {len(doc.raw_text)}')
                print(span.tokens)
                print(raw_text_token)
                break



    def predict(self, doc, method):
        """ Predict coreference clusters in a document. """      

        # Set to eval mode
        self.model.eval()

        # Pass the document through the model
        with torch.no_grad():
            spans, probs, att_weights = self.model(doc)

        self.check_if_span_idx_and_raw_text_align(spans, doc)
        clusters, num_corefs_chosen = show_cluster_repo(spans, probs, doc, att_weights)

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc.raw_text))]

        # idx: id of the cluster-element
        # cluster: a set of tuples
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:                      # (284, 285)
                with open('results_my.txt', 'a') as f:
                    if i1 == i2:
                        token_tags[i1].append(f'({idx})')

                    else:
                        token_tags[i1].append(f'({idx}')
                        token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc, spans, probs, att_weights


    def prepare_golds_file_and_preds_file(self, method):
        # Make predictions directory if there isn't one already
        golds_file, preds_file = f'preds/golds_my_{method}.txt', f'preds/predictions_my_{method}.txt'
        if not os.path.exists('preds/'):
            os.makedirs('preds/')
        create_file_if_not_exists(golds_file)
        clean_file_content(golds_file)
        create_file_if_not_exists(preds_file)
        clean_file_content(preds_file)
        return golds_file, preds_file


    def to_conll(self, eval_dataset, method):
        """ Write to out_file the predictions, return CoNLL metrics results """

        golds_file, preds_file = self.prepare_golds_file_and_preds_file(method)

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text 
                                      for doc in eval_dataset.documents])
        with open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with open(preds_file, 'w', encoding='utf-8', errors='strict') as f:
            
            pred_file_line_id = -1
            for doc in eval_dataset.predicted_docs[:self.config['num_docs_for_eval']]:
                
                # doc.p_spans holds the predicted spans.
                # it should be that the tokens in
                # doc.raw_text[span.start : span.end] == span.tokens

                for doc_line_id, line in enumerate(doc.raw_text):
                    pred_file_line_id += 1
                    
                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()

                        tokens[-1] = doc.tags[doc_line_id]
                        
                        if doc.tags[doc_line_id] != '-':
                            print(f'Line: {pred_file_line_id}, word: {tokens[3]}, tag: {doc.tags[doc_line_id]}\n')
                            # find span where span.start == line_id or span.end == line_id
                            self.print_spans_that_start_or_end_with_line_id(doc, doc_line_id)
                        
                        # Rewrite it back out
                        f.write('\t'.join(tokens))                        

                    f.write('\n')

        return golds_file, preds_file


    def print_spans_that_start_or_end_with_line_id(self, doc, line_id):
        s_start_line = [span for span in doc.p_spans if span.start == line_id]
        s_end_line = [span for span in doc.p_spans if span.end == line_id]
        if s_start_line:
            for s in s_start_line:
                print(s)
        if s_end_line:
            for s in s_end_line:
                print(s)
        print()


    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')


    def load_model(self, loadpath, map_location=torch.device('cpu')):
        """ Load state dictionary into model """
        state = torch.load(loadpath, map_location)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)


if __name__ == '__main__':
    ################################################
    #                   Main
    #################################################

    # extract config.yml into a dict named config
    with open('config.yml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            for k, v in config.items():
                print(f'{k:<22} {v}')

        except yaml.YAMLError as exc:
            print(exc)

    num_docs_for_train = config['num_docs_for_train']      # and eval since I use the same dataset for debugging purposes

    DATA_DIR = Path(config['DATA_DIR'])
    if config['use_full_data']:
        TRAIN_DIR = DATA_DIR.joinpath(config['TRAIN_DIR'])
        DEV_DIR = DATA_DIR.joinpath(config['DEV_DIR'])
        TEST_DIR = DATA_DIR.joinpath(config['TEST_DIR'])
    if not config['use_full_data']:
        TRAIN_DIR = DATA_DIR.joinpath(config['TRAIN_DIR_MIN'])
        DEV_DIR = DATA_DIR.joinpath(config['DEV_DIR_MIN'])
        TEST_DIR = DATA_DIR.joinpath(config['TEST_DIR_MIN'])

    if config['use_real_glove']:
        GLOVE_FILE = config['GLOVE_FILE_300_FULL']
    if not config['use_real_glove']:
        GLOVE_FILE = config['GLOVE_FILE_300_SAMPLE']
    TURIAN_FILE_50 = config['TURIAN_FILE_50']

    if config['recompute_datasets']:
        ds_train = Dataset(get_processed_docs_from_dir(TRAIN_DIR))
        ds_dev = Dataset(get_processed_docs_from_dir(DEV_DIR))
        ds_test = Dataset(get_processed_docs_from_dir(TEST_DIR))
        
        print(f'len(ds_train.documents) = {len(ds_train.documents)}')
        print(f'len(ds_dev.documents) = {len(ds_dev.documents)}')
        print(f'len(ds_test.documents) = {len(ds_test.documents)}')        

        datasets = (ds_train, ds_dev, ds_test)
        vocab = Vocab(datasets)
        genre_set = ds_train.get_genre_set()
        genre_embeds = Genre_Embedding(genre_set)
        
        if config['save_vocab_and_genre_set_to_file']:
            with open("vocab.txt", "wb") as fp:
                pickle.dump(list(vocab.token_set), fp)
            
            with open("genre_set.txt", "wb") as fp:
                pickle.dump(list(genre_set), fp)
        
    if config['load_only_evaluation_dataset']:
        ds_train = Dataset(get_processed_docs_from_dir(DEV_DIR))
        ds_dev = ds_train
        ds_test = ds_train
        datasets = (ds_train, ds_dev, ds_test)

    if config['load_vocab_and_genre_set']:
        with open("vocab.txt", "rb") as fp:
            token_set = pickle.load(fp)

        with open("genre_set.txt", "rb") as fp:   # Unpickling
            genre_set = pickle.load(fp)            
        
        vocab = Vocab(datasets, token_set=token_set)
        genre_embeds = Genre_Embedding(genre_set)
        
    # Randomly truncate some documents to check if truncation works properly
    for doc in random.sample(datasets[0].documents, 
                            min(10, len(datasets[0].documents))):
        for num_keep in range(3, 8):
            _ = truncate_document(doc, num_keep)
    
    # minimal dataset to check for errors in inference step
    ds_min_eval = Dataset(random.sample(datasets[0].documents,
                                        min(10, len(datasets[0].documents))))


    if config['recompute_embeddings']:
        print('Start Loading word vectors from file.')
        GLOVE = Pretrained_Embeddings(GLOVE_FILE, vocab)
        TURIAN = Pretrained_Embeddings(TURIAN_FILE_50, vocab)
        GLOVE_WEIGHTS = GLOVE.to_weights()
        TURIAN_WEIGHTS = TURIAN.to_weights()    
        print(f"GLOVE.word2vec has a length of {len(GLOVE.word2vec)}")
        print(f"TURIAN.word2vec has a length of {len(TURIAN.word2vec)}")

    if config['save_embeddings_to_file']:
        torch.save(GLOVE_WEIGHTS, 'glove_weights.pt')
        torch.save(TURIAN_WEIGHTS, 'turian_weights.pt')
        print('--------> Embeddings were saved to file. <-------')

    if config['load_embeddings_from_file']:
        GLOVE_WEIGHTS = torch.load(config['glove_pt'])
        TURIAN_WEIGHTS = torch.load(config['turian_pt'])

    print(f"There are {len(vocab)} words in the vocab.")
    print(f"GLOVE_WEIGHTS.shape = {GLOVE_WEIGHTS.shape}")
    print(f"TURIAN_WEIGHTS.shape = {TURIAN_WEIGHTS.shape}")
    print(f'genre_embeddings have shape: {genre_embeds.embeds}')
    results_txt = 'results_my.txt'
    results_txt = 'results_my.txt'                 # store all kinds of metrics during training
    create_file_if_not_exists(results_txt)
    clean_file_content(results_txt)
    
    columns = ["loss_mean_epoch",
            "mentionds_found_percent", 
            "corefs_found_percent", 
            "corefs_chosen_percent"]
    df = pd.DataFrame(columns = columns)        # train metrics for quick overview

    new_coref = New_Coref(config, vocab, GLOVE_WEIGHTS, TURIAN_WEIGHTS, genre_embeds)   
    trainer = Trainer(new_coref, datasets, config)
    if config['load_pretrained_model_from_file']:
        trainer.load_model(config['coref_model_path'])
    trainer.train()
    #####################
      
    #####################
    
    trainer.evaluate(dataset=ds_dev,
                     sample_subset=False,
                     comment_string='Now Evaluating Test Dataset')
    

# %% ########################################
#############################################
#############################################



# %% ########################################
#############################################
#############################################




# %% ############################################################
#################################################################
#################################################################
from collections import defaultdict
lengths = defaultdict(list)
span_length = 4
lengths[span_length].append(1)
lengths[span_length].append(1)
lengths[span_length].append(1)



print(lengths)


eval = {k : len(v) for k,v in lengths.items()}
print(eval)