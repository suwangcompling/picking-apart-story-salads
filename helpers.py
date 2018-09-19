import re
import os
import itertools
import numpy as np

from collections import Counter


class Indexer(object):
	"""
	@Author: gdurrett
	Vocab-Index bidirectional dictionary.
	"""
	def __init__(self):
		self.objs_to_ints = {}
		self.ints_to_objs = {}

	def __repr__(self):
		return str([str(self.get_object(i)) for i in range(0, len(self))])

	def __len__(self):
		return len(self.objs_to_ints)

	def get_object(self, index):
		if (index not in self.ints_to_objs):
			return None
		else:
			return self.ints_to_objs[index]

	def contains(self, object):
		return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
	def index_of(self, object):
		if (object not in self.objs_to_ints):
			return -1
		else:
			return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
	def get_index(self, object, add=True):
		if not add:
			return self.index_of(object)
		if (object not in self.objs_to_ints):
			new_idx = len(self.objs_to_ints)
			self.objs_to_ints[object] = new_idx
			self.ints_to_objs[new_idx] = object
		return self.objs_to_ints[object]


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def remove_all_files(target_dir):
    """Remove all existing files for saving new checkpoint."""
    for filename in os.listdir(target_dir):
        os.remove(os.path.abspath(os.path.join(target_dir, filename)))
        
def checkpoint_model(s_dir, s_path, svr, ss):
    """Check point model."""
    # save_dir, save_path, saver, sess
    remove_all_files(s_dir)
    svr.save(ss, s_path)   









































