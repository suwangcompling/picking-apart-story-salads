# Copyright 2018 @Jacob Su Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
sys.path.insert(0, os.getcwd())
import time
import random
import shutil
import dill
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper

from helpers import Indexer, batch, checkpoint_model
from itertools import chain, product
from collections import defaultdict

from kmedoids import kMedoids
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score

from pairwise_classifier import *

class MixtureReader:
    
    def __init__(self, data_dir, data_type, context):
        
        assert data_type in ['nyt', 'wiki']
        
        self.data_dir = data_dir
        self.data_type = data_type
        self.context = context # int: 0 or context-length.
        
    def get_mixture(self, filename):
        
        if self.data_type == 'nyt':
            return self.__get_nyt_mixture(filename)
        else: # == wiki
            return self.__get_wiki_mixture(filename)
        
    def __get_nyt_mixture(self, filename):
        
        da, db, doc_mix = dill.load(open(self.data_dir+filename, 'rb'))
        doc_lbs = []
        for sentcode in doc_mix:
            if sentcode in da:
                doc_lbs.append(0)
            else:
                doc_lbs.append(1)
        if self.context:
            CTX_LEN = self.context
            doc_mix_flat = list(chain.from_iterable(doc_mix))
            doc_mix_len = len(doc_mix_flat)
            ctx = np.array([doc_mix_flat[:CTX_LEN]]) if doc_mix_len>=CTX_LEN else np.array([doc_mix_flat+[0]*(CTX_LEN-doc_mix_len)])            
            return doc_mix, doc_lbs, ctx
        return doc_mix, doc_lbs
    
    def __get_wiki_mixture(self, filename):
        
        doc_mix, doc_lbs = dill.load(open(self.data_dir+filename, 'rb'))
        if self.context:
            CTX_LEN = self.context
            doc_mix_flat = list(chain.from_iterable(doc_mix))
            doc_mix_len = len(doc_mix_flat)
            ctx = np.array([doc_mix_flat[:CTX_LEN]]) if doc_mix_len>=CTX_LEN else np.array([doc_mix_flat+[0]*(CTX_LEN-doc_mix_len)])            
            return doc_mix, doc_lbs, ctx
        return doc_mix, doc_lbs
        


class PscKMedoids:
    
    def __init__(self, psc_clf, data_type):
        
        self.psc_clf = psc_clf
        self.mix_reader = MixtureReader(self.psc_clf.config['data_dir'],
                                        data_type='nyt' if 'nyt' in self.psc_clf.config['data_dir'] else 'wiki',
                                        context=self.psc_clf.config['context_length'] if self.psc_clf.config['context'] else 0)
        self.out_file_path = psc_clf.config['out_file_path']

    def __to_sentence(self, indices):
        words = []
        for index in indices:
            word = self.psc_clf.indexer.get_object(index)
            if word is None:
                words.append('UNK')
            else:
                words.append(word)
        return ' '.join(words)

    def __to_labels(self, C, doc_len): # C: {cls:[datum_id, ...], ...}
        lbs = [0]*doc_len
        for idx in C[1]:
            lbs[idx] = 1
        return lbs

    def __flip_clust(self, clust):
        return np.array([0 if i==1 else 1 for i in clust])

    def __clust_accuracy(self, true, pred):
        return max(accuracy_score(true, pred),
                   accuracy_score(true, self.__flip_clust(pred)))    
        
    def __dist(self, x1, x2):
        
        x1, x1_len = batch([x1])
        x2, x2_len = batch([x2])
        fd = {self.psc_clf.input_x1:x1, self.psc_clf.input_x1_length:x1_len,
              self.psc_clf.input_x2:x2, self.psc_clf.input_x2_length:x2_len,
              self.psc_clf.keep_prob:1.0}
        if self.psc_clf.config['context']:
            fd[self.psc_clf.input_ctx] = self.ctx
        conf = self.psc_clf.sess.run(self.psc_clf.scores, feed_dict=fd)
        return 1-conf[0]  
    
    def evaluate_single(self, doc_mix, doc_lbs, ctx=None, method='average', return_pred=True):
        
        if ctx is not None:
            self.ctx = ctx
        doc_mix_sq, _ = batch(doc_mix)
        doc_mix_sq = doc_mix_sq.T
        _, doc_mix_clust = kMedoids(squareform(pdist(doc_mix_sq,metric=self.__dist)), 2)
        doc_prd = self.__to_labels(doc_mix_clust, len(doc_mix))
        acc = self.__clust_accuracy(doc_lbs, doc_prd)
        if return_pred:
            return acc, doc_prd
        return acc 
    
    def evaluate_rand(self, k=100, verbose=True):
        
        accs = []
        filenames = np.random.choice(self.psc_clf.FILENAMES, size=k, replace=False)
        if self.out_file_path is not None: # clear out file for new writing.
            out_file = open(self.out_file_path, 'w')
        for filename in filenames:
            if self.mix_reader.context:
                doc_mix, doc_lbs, ctx = self.mix_reader.get_mixture(filename)
                result = self.evaluate_single(doc_mix, doc_lbs, ctx, self.out_file_path is not None)
            else:
                doc_mix, doc_lbs = self.mix_reader.get_mixture(filename, self.out_file_path is not None)
                result = self.evaluate_single(doc_mix, doc_lbs)
            if out_file_path is None:
                acc = result
            else:
                acc, prd = result
                out_file.write('FILE ID: ' + str(filename) + '\n')
                for prd_lb, true_lb, indices in zip(prd, doc_lbs, doc_mix):
                    out_file.write('TRUE = '+str(true_lb)+' | '+'PRED = '+str(prd_lb)+' | '+self.__to_sentence(indices)+'\n')
            out_file.write('\n\n')
            accs.append(acc)
            if verbose:
                print('File {}: acc = {}'.format(filename, acc))
        out_file.close()
        avg_acc = np.mean(accs)
        print('\nAverage accuracy = {}'.format(avg_acc))
        return avg_acc
    
    def evaluate_given(self, filenames, verbose=True):
        
        accs = []
        if self.out_file_path is not None: # clear out file for new writing.
            out_file = open(self.out_file_path, 'w')
        for filename in filenames:
            if self.mix_reader.context:
                doc_mix, doc_lbs, ctx = self.mix_reader.get_mixture(filename)
                result = self.evaluate_single(doc_mix, doc_lbs, ctx, self.out_file_path is not None)
            else:
                doc_mix, doc_lbs = self.mix_reader.get_mixture(filename)
                result = self.evaluate_single(doc_mix, doc_lbs)
            if self.out_file_path is None:
                acc = result
            else:
                acc, prd = result
                out_file.write('FILE ID: ' + str(filename) + '\n')
                for prd_lb, true_lb, indices in zip(prd, doc_lbs, doc_mix):
                    out_file.write('TRUE = '+str(true_lb)+' | '+'PRED = '+str(prd_lb)+' | '+self.__to_sentence(indices)+'\n')
            out_file.write('\n\n')
            accs.append(acc)
            if verbose:
                print('File {}: acc = {}'.format(filename, acc))              
        out_file.close()
        avg_acc = np.mean(accs)
        print('\nAverage accuracy = {}'.format(avg_acc))
        return avg_acc   
    
    
if __name__ == "__main__": 
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--emb_size', type=int)
    parser.add_argument('--n_layer', type=int)
    parser.add_argument('--hid_size', type=int)
    parser.add_argument('--keep_prob', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--info_path', type=str)
    parser.add_argument('--init_with_glove', type=bool)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--restore_dir', type=str)
    parser.add_argument('--restore_name', type=str)
    parser.add_argument('--load_from_saved', type=bool)
    parser.add_argument('--track_dir', type=str)
    parser.add_argument('--new_track', type=bool)
    parser.add_argument('--session_id', type=str)
    parser.add_argument('--mutual_attention', type=bool)
    parser.add_argument('--context', type=bool)
    parser.add_argument('--context_length', type=int)
    parser.add_argument('--out_file_path', type=str)
    args = parser.parse_args()

    config = {'batch_size': args.batch_size, 'vocab_size': args.vocab_size, 'emb_size': args.emb_size,
              'n_layer': args.n_layer, 'hid_size': args.hid_size,
              'keep_prob': args.keep_prob, 'learning_rate': args.learning_rate,
              'n_epoch': args.n_epoch, 'train_size': args.train_size, 'verbose': args.verbose,
              'save_freq': args.save_freq,
              'data_dir': args.data_dir, 'info_path': args.info_path,
              'init_with_glove': args.init_with_glove,
              'save_dir': args.save_dir, 'save_name': args.save_name,
              'restore_dir': args.restore_dir, 'restore_name': args.restore_name,
              'load_from_saved': args.load_from_saved,
              'track_dir': args.track_dir, 'new_track': args.new_track, 'session_id': args.session_id,
              'mutual_attention': args.mutual_attention, 
              'context': args.context, 'context_length': args.context_length,
              'out_file_path': args.out_file_path}
    
    psc_clf = PairwiseSentenceClassifier(config)
    kmed = PscKMedoids(psc_clf, data_type='nyt')
    print('\n')
    sample_files = os.listdir('nyt_sample/')
    kmed.evaluate_given(sample_files)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    