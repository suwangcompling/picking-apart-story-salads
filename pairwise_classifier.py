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

    
class PairwiseSentenceClassifier:
    
    def __init__(self, config):
        
        self.config = config
        
        self.FILENAMES = os.listdir(self.config['data_dir'])
        self.indexer, self.word2emb = dill.load(open(self.config['info_path'], 'rb'))
        
        if self.config['init_with_glove']:
            glove_embs = []
            for i in range(len(self.indexer)):
                glove_embs.append(self.word2emb[self.indexer.get_object(i)])
            self.glove_embs = np.array(glove_embs)
        else:
            del self.word2emb
        
        if self.config['load_from_saved']:
            self.__load_saved_graph()
            print('Model loaded for continued training!')
        else:
            self.__build_new_graph()  
            print('New model built for training!')
            
    # Build & train new graph
        
    def __build_new_graph(self):
        
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        self.__init_placeholders()
        self.__init_embeddings()
        
        self.cell = MultiRNNCell([DropoutWrapper(LSTMCell(self.config['hid_size']),
                                                 output_keep_prob=self.keep_prob)]*self.config['n_layer'])
        
        self.__run_nets(add_mutual_attention=self.config['mutual_attention'],
                        add_context_reader=self.config['context'])
        self.__run_score_and_predictions()
        self.__run_accuracy()
        self.__run_optimization()
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    
    def __init_placeholders(self):
        
        self.input_x1 = tf.placeholder(tf.int32, [None, None], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, None], name='input_x2')
        self.input_x1_length = tf.placeholder(tf.int32, [None], name='input_x1_length')
        self.input_x2_length = tf.placeholder(tf.int32, [None], name='input_x2_length')
        self.input_y  = tf.placeholder(tf.int32, [None], name='input_y')
        
        if self.config['context']:
            self.input_ctx = tf.placeholder(tf.int32, [1, self.config['context_length']], name='input_ctx')
            
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")     
            
    def __init_embeddings(self):
        
        with tf.variable_scope('Emebeddings'):
            self.embeddings = tf.get_variable('embeddings', [self.config['vocab_size'], self.config['emb_size']], 
                                         initializer=tf.contrib.layers.xavier_initializer())
            if self.config['init_with_glove']:
                glove_init = self.embeddings.assign(self.glove_embs)
            self.input_x1_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_x1) 
            self.input_x2_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_x2)
            if self.config['context']:
                self.input_ctx_embedded = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, self.input_ctx), -1)
    
    def __run_lstm(self, inputs, inputs_length):
        
        ((fw_outputs,bw_outputs), 
         (fw_final_state,bw_final_state)) = ( 
            tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell,
                                            cell_bw=self.cell,
                                            inputs=inputs,
                                            sequence_length=inputs_length,
                                            dtype=tf.float32, time_major=True) )
        if self.config['context']:
            return tf.concat([tf.concat([fw_state_tuple.h,bw_state_tuple.h], 1) 
                      for fw_state_tuple,bw_state_tuple in zip(fw_final_state,bw_final_state)], 1), \
                   tf.transpose(tf.concat([fw_outputs,bw_outputs], 2), [1,0,2])
        else:
            return tf.concat([tf.concat([fw_state_tuple.h,bw_state_tuple.h], 1) 
                      for fw_state_tuple,bw_state_tuple in zip(fw_final_state,bw_final_state)], 1)
        
    def __run_attention(self, outputs, state):
        
        W_d = tf.get_variable('W_d', [self.config['hid_size']*2, self.config['hid_size']*2], 
                              initializer=tf.contrib.layers.xavier_initializer())
        W_s = tf.get_variable('W_s', [self.config['hid_size']*2*self.config['n_layer'], self.config['hid_size']*2], 
                              initializer=tf.contrib.layers.xavier_initializer())
        d_W = tf.tensordot(outputs, W_d, axes=[[2],[0]])
        s_W = tf.expand_dims(tf.matmul(state, W_s), axis=1)
        a_tsr = tf.nn.tanh(tf.add(d_W, s_W))
        W_a = tf.get_variable('W_a', [self.config['hid_size']*2, 1], 
                              initializer=tf.contrib.layers.xavier_initializer())
        a_W = tf.nn.softmax(tf.tensordot(a_tsr, W_a, axes=[[2],[0]]), dim=1)
        d_a = tf.reduce_sum(tf.multiply(outputs, a_W), axis=1)
        return d_a  
    
    def __run_cnn(self, inputs):
        
        FILTER_SIZES = [3,4,5]
        NUM_FILTERS = 50
        NUM_CHANNELS = 1
        CTX_LEN = self.config['context_length']
        
        pool_outputs = []
        for i,filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope('CNN-ctx-%s' % filter_size):
                filter_shape = [filter_size, EMB_SIZE, NUM_CHANNELS, NUM_FILTERS]
                W = tf.get_variable('W', filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b', [NUM_FILTERS], initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pool = tf.nn.max_pool(h, ksize=[1,CTX_LEN-filter_size+1,1,1], strides=[1,1,1,1], 
                                      padding='VALID', name='pool')
                pool_outputs.append(pool)
        num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
        h_pool_flat = tf.nn.dropout(tf.reshape(tf.concat(pool_outputs, 3), [-1, num_filters_total]), keep_prob)
        return h_pool_flat     
    
    def __run_nets(self, add_mutual_attention=False, add_context_reader=False):
        
        if add_mutual_attention and add_context_reader:
            with tf.variable_scope('Bi-LSTM') as scope:
                final_state_x1, outputs_x1 = self.__run_lstm(self.input_x1_embedded, self.input_x1_length)
                scope.reuse_variables() 
                final_state_x2, outputs_x2 = self.__run_lstm(self.input_x2_embedded, self.input_x2_length)
            with tf.variable_scope('Mutual-Attention') as scope:
                x1_to_x2_att = self.__run_attention(outputs_x2, final_state_x1)
                scope.reuse_variables()
                x2_to_x1_att = self.__run_attention(outputs_x1, final_state_x2)
            with tf.variable_scope('Context-reader'):
                bc, _ = tf.unstack(tf.shape(final_state_x1))
                ctx = tf.tile(self.__run_cnn(self.input_ctx_embedded), [bc, 1])
            self.final_vec_x1 = tf.concat([final_state_x1, x1_to_x2_att, ctx],axis=1)
            self.final_vec_x2 = tf.concat([final_state_x2, x2_to_x1_att, ctx],axis=1)
            self.final_vec_size = self.config['hid_size']*2*self.config['n_layer'] + \
                                  self.config['hid_size']*2 + \
                                  3*50 # n_filter * len(filter-sizes)
        
        elif add_mutual_attention:
            with tf.variable_scope('Bi-LSTM') as scope:
                final_state_x1, outputs_x1 = self.__run_lstm(self.input_x1_embedded, self.input_x1_length)
                scope.reuse_variables() 
                final_state_x2, outputs_x2 = self.__run_lstm(self.input_x2_embedded, self.input_x2_length)
            with tf.variable_scope('Mutual-Attention') as scope:
                x1_to_x2_att = self.__run_attention(outputs_x2, final_state_x1)
                scope.reuse_variables()
                x2_to_x1_att = self.__run_attention(outputs_x1, final_state_x2) 
            self.final_vec_x1 = tf.concat([final_state_x1, x1_to_x2_att],axis=1)
            self.final_vec_x2 = tf.concat([final_state_x2, x2_to_x1_att],axis=1)
            self.final_vec_size = self.config['hid_size']*2*self.config['n_layer'] + \
                                  self.config['hid_size']*2
                
        elif add_context_reader:
            with tf.variable_scope('Bi-LSTM') as scope:
                final_state_x1, outputs_x1 = self.__run_lstm(self.input_x1_embedded, self.input_x1_length)
                scope.reuse_variables() 
                final_state_x2, outputs_x2 = self.__run_lstm(self.input_x2_embedded, self.input_x2_length) 
            with tf.variable_scope('Context-reader'):
                bc, _ = tf.unstack(tf.shape(final_state_x1))
                ctx = tf.tile(self.__run_cnn(self.input_ctx_embedded), [bc, 1])
            self.final_vec_x1 = tf.concat([final_state_x1, ctx],axis=1)
            self.final_vec_x2 = tf.concat([final_state_x2, ctx],axis=1) 
            self.final_vec_size = self.config['hid_size']*2*self.config['n_layer'] + \
                                  3*50
        
        else:
            with tf.variable_scope('Bi-LSTM') as scope:
                final_state_x1 = self.__run_lstm(self.input_x1_embedded, self.input_x1_length)
                scope.reuse_variables()
                final_state_x2 = self.__run_lstm(self.input_x2_embedded, self.input_x2_length)
            self.final_vec_x1 = final_state_x1
            self.final_vec_x2 = final_state_x2
            self.final_vec_size = self.config['hid_size']*2*self.config['n_layer']
        
    def __run_score_and_predictions(self):
        
        W_bi = tf.get_variable('W_bi', [self.final_vec_size, self.final_vec_size], 
                               initializer=tf.contrib.layers.xavier_initializer())
        self.scores = tf.nn.sigmoid(tf.diag_part(tf.matmul(tf.matmul(self.final_vec_x1,W_bi),
                                                           tf.transpose(self.final_vec_x2))), name='scores')
        self.predictions = tf.cast(tf.round(self.scores), tf.int32, name='predictions')
    
    def __run_accuracy(self):
        
        with tf.name_scope('Accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
    
    def __run_optimization(self):
        
        with tf.name_scope('Loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32), 
                                                             logits=self.scores)
            self.loss = tf.reduce_mean(losses, name='loss')  
    
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name='train_op')
    
    # Load and train old graph
    
    def __load_saved_graph(self):
        
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.config['restore_dir'] + self.config['restore_name'])
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config['restore_dir']))
        self.graph = tf.get_default_graph()
        
        self.input_x1 = self.graph.get_tensor_by_name('input_x1:0')
        self.input_x2 = self.graph.get_tensor_by_name('input_x2:0')
        self.input_x1_length = self.graph.get_tensor_by_name('input_x1_length:0')
        self.input_x2_length = self.graph.get_tensor_by_name('input_x2_length:0')
        self.input_y = self.graph.get_tensor_by_name('input_y:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        
        if self.config['context']:
            self.input_ctx = self.graph.get_tensor_by_name('input_ctx:0')
            
        self.scores = self.graph.get_tensor_by_name('scores:0')
        self.predictions = self.graph.get_tensor_by_name('predictions:0')
        self.loss = self.graph.get_tensor_by_name('Loss/loss:0')
        self.accuracy = self.graph.get_tensor_by_name('Accuracy/accuracy:0')
        self.global_step = self.graph.get_tensor_by_name('global_step:0')
        self.train_op = self.graph.get_tensor_by_name('train_op:0') 

class DataBatcher:
    
    def __init__(self, data_dir, context=False):
        
        self.data_dir = data_dir
        self.context = context # tuple: (bool:context-or-not, context-length)
        
    def __generate_pair_batch(self, doc_a, doc_b, k):
        
        batch_x1, batch_x2, batch_y = [], [], []
        ys = [1,0,0,1]
        for _ in range(k): # 4 entries added per iteration.
            for i,(da,db) in enumerate(product([doc_a, doc_b], 
                                               [doc_a, doc_b])):
                batch_x1.append(random.choice(da))
                batch_x2.append(random.choice(db))
                batch_y.append(ys[i])
        return batch(batch_x1), batch(batch_x2), np.array(batch_y)
    
    def get_batch(self, filename, n=32):
        assert n%4==0
        doc_a, doc_b, doc_mix = dill.load(open(self.data_dir+filename, 'rb'))
        (batch_x1,batch_x1_len), (batch_x2,batch_x2_len), batch_y = self.__generate_pair_batch(doc_a,doc_b,int(n//4))
        if self.context[0]:
            CTX_LEN = self.context[1]
            doc_mix_flat = list(chain.from_iterable(doc_mix))
            doc_mix_len = len(doc_mix_flat)
            doc_mix_padded = np.array(doc_mix_flat[:CTX_LEN]) if doc_mix_len>=CTX_LEN \
                                 else np.array(doc_mix_flat+[0]*(CTX_LEN-doc_mix_len))
            batch_ctx = np.array([doc_mix_padded])
            return batch_x1,batch_x1_len,batch_x2,batch_x2_len,batch_ctx,batch_y
        return batch_x1,batch_x1_len,batch_x2,batch_x2_len,batch_y
        
def train_pairwise_clf(config):
    
    clf = PairwiseSentenceClassifier(config)
    
    dat = DataBatcher(clf.config['data_dir'], context=(clf.config['context'], clf.config['context_length']))
    
    track_dir, session_id = clf.config['track_dir'], clf.config['session_id']
    log_mode = 'w' if clf.config['new_track'] else 'a'
    with open(track_dir+session_id+'.txt', log_mode) as f:
        f.write('\n\n=== NEW SESSION ===\n\n')
    loss_track, accuracy_track = [], []
    start = time.time()
    try:
        for e in range(clf.config['n_epoch']):
            with open(track_dir+session_id+'.txt', 'a') as f:
                f.write('Epoch '+str(e+1)+'\n')
            file_indices = np.random.choice(list(range(len(clf.FILENAMES))), 
                                            size=clf.config['train_size'], replace=False)
            random.shuffle(file_indices)
            curr_loss_track, curr_accuracy_track = [], []
            for file_idx in file_indices:
                try:
                    if clf.config['context']:
                        batch_x1,batch_x1_length,batch_x2,batch_x2_length,batch_ctx,batch_y = dat.get_batch(clf.FILENAMES[file_idx],n=clf.config['batch_size'])
                    else:
                        batch_x1,batch_x1_length,batch_x2,batch_x2_length,batch_y = dat.get_batch(clf.FILENAMES[file_idx],n=clf.config['batch_size'])
                except:
                    continue
                fd = {clf.input_x1:batch_x1, clf.input_x1_length:batch_x1_length,
                      clf.input_x2:batch_x2, clf.input_x2_length:batch_x2_length,
                      clf.input_y:batch_y,
                      clf.keep_prob:clf.config['keep_prob']} 
                if clf.config['context']:
                    fd[clf.input_ctx] = batch_ctx
                _, step, loss_, accuracy_ = clf.sess.run([clf.train_op, clf.global_step, 
                                                          clf.loss, clf.accuracy], feed_dict=fd)
                curr_loss_track.append(loss_)
                curr_accuracy_track.append(accuracy_)
                if step % clf.config['save_freq'] == 0:
                    checkpoint_model(clf.config['save_dir'], clf.config['save_dir']+clf.config['save_name'],
                                     clf.saver, clf.sess)
                if step % clf.config['verbose'] == 0:
                    with open(track_dir+session_id+'.txt', 'a') as f:
                        avg_loss = np.mean(curr_loss_track)
                        avg_acc = np.mean(curr_accuracy_track)
                        loss_track.append(avg_loss)
                        accuracy_track.append(avg_acc)
                        f.write('loss & accuracy at step {}: <{}, {}> (time elapsed = {} secs)\n'.format(step, 
                                                                                        round(avg_loss,5),
                                                                                        round(avg_acc,5),
                                                                                        round(time.time()-start,2)))
                    start = time.time()
                    curr_loss_track, curr_accuracy_track = [], []
        with open(track_dir+session_id+'-final.txt', log_mode) as f:
            f.write('final avg loss & accuracy: <{}, {}>'.format(round(np.mean(loss_track),5),
                                                                 round(np.mean(accuracy_track),5)))
    except KeyboardInterrupt:
        print('Stopped!')
                        
    
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
              'context': args.context, 'context_length': args.context_length}

    train_pairwise_clf(config)
