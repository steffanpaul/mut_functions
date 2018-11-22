from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import mutagenesisfunctions as mf


import tensorflow as tf
import time as time


def load_dataset_hdf5(file_path, dataset_name=None, ss_type='seq', rbp_index=None):

	def prepare_data(train, ss_type=None):

		seq = train['inputs'][:,:,:,:4]

		if ss_type == 'pu':
			structure = train['inputs'][:,:,:,4:9]
			paired = np.expand_dims(structure[:,:,:,0], axis=3)
			unpaired = np.expand_dims(np.sum(structure[:,:,:,1:], axis=3), axis=3)
			seq = np.concatenate([seq, paired, unpaired], axis=3)

		elif ss_type == 'struct':
			structure = train['inputs'][:,:,:,4:9]
			paired = np.expand_dims(structure[:,:,:,0], axis=3)
			HIME = structure[:,:,:,1:]
			seq = np.concatenate([seq, paired, HIME], axis=3)

		train['inputs']  = seq
		return train

	# open dataset
	experiments = None
	dataset = h5py.File(file_path, 'r')
	if not dataset_name:
		# load set A data
		X_train = np.array(dataset['X_train']).astype(np.float32)
		Y_train = np.array(dataset['Y_train']).astype(np.float32)
		X_valid = np.array(dataset['X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
		X_test = np.array(dataset['X_test']).astype(np.float32)
		Y_test = np.array(dataset['Y_test']).astype(np.float32)

		# expand dims of targets
		if rbp_index is not None:
			Y_train = Y_train[:,rbp_index]
			Y_valid = Y_valid[:,rbp_index]
			Y_test = Y_test[:,rbp_index]
	else:
		X_train = np.array(dataset['/'+dataset_name+'/X_train']).astype(np.float32)
		Y_train = np.array(dataset['/'+dataset_name+'/Y_train']).astype(np.float32)
		X_valid = np.array(dataset['/'+dataset_name+'/X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['/'+dataset_name+'/Y_valid']).astype(np.float32)
		X_test = np.array(dataset['/'+dataset_name+'/X_test']).astype(np.float32)
		Y_test = np.array(dataset['/'+dataset_name+'/Y_test']).astype(np.float32)

	# expand dims of targets
	if len(Y_train.shape) == 1:
		Y_train = np.expand_dims(Y_train, axis=1)
		Y_valid = np.expand_dims(Y_valid, axis=1)
		Y_test = np.expand_dims(Y_test, axis=1)

	# add another dimension to make a 4d tensor
	X_train = np.expand_dims(X_train, axis=3).transpose([0, 2, 3, 1])
	X_test = np.expand_dims(X_test, axis=3).transpose([0, 2, 3, 1])
	X_valid = np.expand_dims(X_valid, axis=3).transpose([0, 2, 3, 1])

	# dictionary for each dataset
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	# parse secondary structure profiles
	train = prepare_data(train, ss_type)
	valid = prepare_data(valid, ss_type)
	test = prepare_data(test, ss_type)

	return train, valid, test


def process_data(train, valid, test, method='log_norm'):
	"""get the results for a single experiment specified by rbp_index.
	Then, preprocess the binding affinity intensities according to method.
	method:
		clip_norm - clip datapoints larger than 4 standard deviations from the mean
		log_norm - log transcormation
		both - perform clip and log normalization as separate targets (expands dimensions of targets)
	"""

	def normalize_data(data, method):
		if method == 'standard':
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			sigma = np.mean(data)
			data_norm = (data)/sigma
			params = sigma
		if method == 'clip_norm':
			# standard-normal transformation
			significance = 4
			std = np.std(data)
			index = np.where(data > std*significance)[0]
			data[index] = std*significance
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [mu, sigma]

		elif method == 'log_norm':
			# log-standard-normal transformation
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [MIN, mu, sigma]

		elif method == 'both':
			data_norm1, params = normalize_data(data, 'clip_norm')
			data_norm2, params = normalize_data(data, 'log_norm')
			data_norm = np.hstack([data_norm1, data_norm2])
		return data_norm, params


	# get binding affinities for a given rbp experiment
	Y_train = train['targets']
	Y_valid = valid['targets']
	Y_test = test['targets']

	# filter NaN
	train_index = np.where(np.isnan(Y_train) == False)[0]
	valid_index = np.where(np.isnan(Y_valid) == False)[0]
	test_index = np.where(np.isnan(Y_test) == False)[0]
	Y_train = Y_train[train_index]
	Y_valid = Y_valid[valid_index]
	Y_test = Y_test[test_index]
	X_train = train['inputs'][train_index]
	X_valid = valid['inputs'][valid_index]
	X_test = test['inputs'][test_index]

	# normalize intenensities
	if method:
		Y_train, params_train = normalize_data(Y_train, method)
		Y_valid, params_valid = normalize_data(Y_valid, method)
		Y_test, params_test = normalize_data(Y_test, method)

	# store sequences and intensities
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	return train, valid, test


def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir


def dataset_keys_hdf5(file_path):

	dataset = h5py.File(file_path, 'r')
	keys = []
	for key in dataset.keys():
		keys.append(str(key))

	return np.array(keys)


def get_experiments_hdf5(file_path):
	dataset = h5py.File(file_path, 'r')
	return np.array(dataset['experiment'])


def convert_seq_to_one_hot(sequences, alphabet):
    seq_length = len(sequences[0])
    num_alphabet = len(alphabet)
    one_hot = np.zeros((len(sequences), seq_length, num_alphabet))
    for i, seq in enumerate(sequences):
        for j, letter in enumerate(seq):
            one_hot[i,j,letter] = 1.0
    return one_hot


'''-----------------------------------------------------------------------------------------------
PETERS OTHER FUNCTIONS IN THE RNN NOTEBOOKS
--------------------------------------------------------------------------------------------'''

def filter_long_sequences(X_train, Y_train, MAX=300):
    
    X_new = []
    Y_new = []
    good_index = []
    for i, x in enumerate(X_train):
        if x.shape[0] <= MAX:
            X_new.append(x)
            Y_new.append(Y_train[i])
            good_index.append(i)
            
    return X_new, np.vstack(Y_new), good_index

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def roc(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        score = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
            score = auc(fpr, tpr)
            metric[i]= score
            curves.append((fpr, tpr))
    return metric, curves


def pr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves

def pad_inputs(X_train, MAX):
    X_train_padded = []
    X_length = []
    for x in X_train:
        seq_length, num_alphabet = x.shape
        X_length.append(seq_length)
        padding = np.zeros((MAX-seq_length, num_alphabet))

        X_train_padded.append([np.vstack([x, padding])])
    X_train_padded = np.vstack(X_train_padded)
    X_length = np.array(X_length)
    return X_train_padded, X_length

def get_maxlength(X_train):
    # calculate lengths of each sequence
    lengths = []
    for x in X_train:
        lengths.append(x.shape[0])
    lengths = np.array(lengths)
    return (np.max(lengths))


def batch_generator(X_train, batch_size=64, MAX=None, shuffle_data=True):
 
    # calculate lengths of each sequence
    lengths = []
    for x in X_train:
        lengths.append(x.shape[0])
    lengths = np.array(lengths)

    # add zero padding to all sequences
    if not MAX:
        MAX = np.max(lengths)
    X_train_padded, X_train_length = pad_inputs(X_train, MAX)

    # shuffle data
    num_data = len(X_train)
    if shuffle_data:
        shuffle = np.random.permutation(num_data)
    else:
        shuffle = np.array(range(num_data))
        
    # generate mini-batches of data
    num_batches = np.floor(num_data/batch_size).astype(int)
    batches = []
    for i in range(num_batches):
        indices = range(i*batch_size, (i+1)*batch_size)
        batches.append(X_train_padded[shuffle[indices]])
    remainder = num_data - batch_size*num_batches
    if remainder > 0:
        indices = range(num_batches*batch_size, num_data)
        batches.append(X_train_padded[shuffle[indices]])
    return batches
    
    
    
def bucket_generator(X_train, Y_train, batch_size=32, index=False):
    num_data = len(X_train)

    # add zero padding
    lengths = []
    for x in X_train:
        lengths.append(x.shape[0])
    lengths = np.array(lengths)

    # sort the lengths
    sort_index = np.argsort(lengths)

    num_buckets = np.floor(num_data/batch_size).astype(int)
    buckets = []
    for i in range(num_buckets):
        indices = sort_index[i*batch_size:(i+1)*batch_size]
        X = []
        for j in indices:
            X.append(X_train[j])

        MAX = len(X[-1])
        X_padded, _ = pad_inputs(X, MAX)
        buckets.append([X_padded, 
                        Y_train[indices,:]])

    if num_data > num_buckets*batch_size:
        indices = sort_index[num_buckets*batch_size:]
        X = []
        for j in indices:
            X.append(X_train[j])

        MAX = len(X[-1])
        X_padded, _ = pad_inputs(X, MAX)
        buckets.append([X_padded, 
                        Y_train[indices,:]])
    
    if not index:
        return buckets
    
    if index:
        return buckets, sort_index


'''-----------------------------------------------------------------------------------------------
RNN SPECIFIC SECOND ORDER MUTAGENESIS
--------------------------------------------------------------------------------------------'''

def som_average_ungapped_logodds(Xdict, ungapped_index_list, savepath, nntrainer, sess, progress='on', save=True, layer='output', 
                         normalize=False, normfactor=None, eps=0):

    num_summary, seqlen, _, dims = Xdict.shape

    starttime = time.time()

    sum_mut2_scores = []

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)
        
        epoch_starttime = time.time()
        
        #extract sequence
        X = np.expand_dims(Xdict[ii], axis=0)
        #extract ugidx
        ungapped_index = ungapped_index_list[ii]
        idxlen = len(ungapped_index)
        #Get WT score
        WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
        WT_score = nntrainer.get_activations(sess, WT, layer=layer)[0]

        X_mutsecorder = mf.double_mutate_ungapped(X, ungapped_index)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (idxlen*idxlen*dims*dims, seqlen, 1, dims))
        mutations = {'inputs': X_mutsecorder_reshape, 'targets': np.ones((X_mutsecorder_reshape.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer=layer)
        minscore = np.min(mut2_scores)
        
        #mut2_scores = np.log(np.clip(mut2_scores, a_min=0., a_max=1e7) + 1e-7) - np.log(WT_score+1e-7)
        mut2_scores = np.log(mut2_scores - minscore + 1e-7) - np.log(WT_score-minscore+1e-7)

        #Sum all the scores into a single matrix
        sum_mut2_scores.append(mut2_scores)

        epoch_endtime = time.time()
        
        if progress == 'on':

            print ('Epoch duration =' + mf.sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + mf.sectotime(epoch_endtime - starttime))
            print ()

        if progress == 'short':
            if ii%100 == 0:
                print (ii)
                print ('Epoch duration =' + mf.sectotime((epoch_endtime -epoch_starttime)*100))
                print ('Cumulative duration =' + mf.sectotime(epoch_endtime - starttime))
                print () 
            
    print ('----------------Summing complete----------------')
    
    mean_mut2_scores = np.nanmean(sum_mut2_scores, axis=0)    
    
    # Save the summed array for future use
    if save == True:
        np.save(savepath, mean_mut2_scores)
        print ('Saving scores to ' + savepath)


    return (mean_mut2_scores)


def normalize_mut_hol(hol_mut, normfactor=None):
    norm_hol_mut = np.copy(hol_mut)
    for one in range(hol_mut.shape[0]):
        for two in range(hol_mut.shape[0]):
            norm_hol_mut[one, two] = mf.normalize_hol(hol_mut[one, two], factor=normfactor)
    return norm_hol_mut


#This is used for varfam trials with Deepomics NNs
def som_average_ungapped_logodds_unalign(Xdict, ungapped_index, maxlength, savepath, nntrainer, sess, progress='on', save=True, layer='dense_1_bias', 
                         normalize=False, normfactor=None, eps=0):

    num_summary, seqlen, _, dims = Xdict.shape

    starttime = time.time()

    sum_mut2_scores = []

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)
        
        epoch_starttime = time.time()
        
        #extract sequence
        X = np.expand_dims(Xdict[ii], axis=0)

        idxlen = len(ungapped_index)
        #Get WT score
        X_unalign = [unalign(x) for x in X]
        X_unalign = np.expand_dims(pad_inputs(X_unalign, MAX=maxlength)[0], axis=2)
        WT = {'inputs': X_unalign, 'targets': np.ones((X_unalign.shape[0], 1))}
        WT_score = nntrainer.get_activations(sess, WT, layer=layer)[0]
        
        #Mutate
        X_mutsecorder = mf.double_mutate_ungapped(X, ungapped_index)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (idxlen*idxlen*dims*dims, seqlen, 1, dims))
        Xmut_unalign = [unalign(X) for X in X_mutsecorder_reshape]
        X_data_unalign = np.expand_dims(pad_inputs(Xmut_unalign, MAX=maxlength)[0], axis=2)
        mutations = {'inputs': X_data_unalign, 'targets': np.ones((X_data_unalign.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer=layer)
        
        #mut2_scores = np.log(np.clip(mut2_scores, a_min=0., a_max=1e7) + 1e-7) - np.log(WT_score+1e-7)
        mut2_scores = np.log(mut2_scores + 1e-7) - np.log(WT_score+1e-7)

        #Sum all the scores into a single matrix
        sum_mut2_scores.append(mut2_scores)

        epoch_endtime = time.time()
        
        if progress == 'on':

            print ('Epoch duration =' + mf.sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + mf.sectotime(epoch_endtime - starttime))
            print ()

        if progress == 'short':
            if ii%100 == 0:
                print (ii)
                print ('Epoch duration =' + mf.sectotime((epoch_endtime -epoch_starttime)*100))
                print ('Cumulative duration =' + mf.sectotime(epoch_endtime - starttime))
                print () 
                
                mean_mut2_scores = np.nanmean(sum_mut2_scores, axis=0)
                idx = np.where(mean_mut2_scores == np.nan)
                mean_mut2_scores[idx] = np.log(0.01)
                
                # Save the summed array for future use
                if save == True:
                    np.save(savepath, mean_mut2_scores)
            
    print ('----------------Summing complete----------------')
    
    mean_mut2_scores = np.nanmean(sum_mut2_scores, axis=0)
    idx = np.where(mean_mut2_scores == np.nan)
    mean_mut2_scores[idx] = np.log(0.01)
    
    # Save the summed array for future use
    if save == True:
        np.save(savepath, mean_mut2_scores)
        print ('Saving scores to ' + savepath)


    return (mean_mut2_scores)



def unalign(X):
    nuc_index = np.where(np.sum(X, axis=2)!=0)
    return (X[nuc_index])






















