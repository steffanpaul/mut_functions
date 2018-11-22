from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

import seaborn as sb
import time as time
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

#sequence generator function
def seq_generator(num_data, seq_length, dims, seed):

    np.random.seed(seed)
    Xsim = np.zeros((num_data, seq_length, 1, dims), np.float32)
    for d in range(num_data):
        Xsim_key = np.random.choice([0,1,2,3], seq_length, [0.25, 0.25, 0.25, 0.25])
        Xsim_hp = np.zeros((seq_length,1, dims))
        for (idx,nuc) in enumerate(Xsim_key):
            Xsim_hp[idx][0][nuc] = 1
        Xsim[d] = Xsim_hp
    return Xsim

def seq_generator_gaps(SS, num_data, seq_length, dims, pgaps=(0.05,0.8), seed=274, gapchar='.'):
    np.random.seed(seed)
    Xsim = np.zeros((num_data, seq_length, 1, dims+1))
    p,q = pgaps

    for d in range(num_data):

        #ungapped random nucleotides
        Xsim_seq = np.random.choice(dims+1, seq_length, p=[(1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4, p])
        #gapped random nucleotides
        Xsim_gaps = np.random.choice(dims+1, seq_length, p=[(1-q)/4, (1-q)/4, (1-q)/4, (1-q)/4, q])
        #merge
        gapidx = consensus_gaps(SS, gapchar)
        Xsim_seq[gapidx] = np.copy(Xsim_gaps[gapidx])


        Xsim_hp = np.zeros((seq_length,1, dims+1))
        for (idx,nuc) in enumerate(Xsim_seq):
            Xsim_hp[idx][0][nuc] = 1
        Xsim[d] = Xsim_hp

    return (Xsim)


def seq_generator_profile(profile, numdata, seqlen, dims, seed=274):
    np.random.seed(seed)
    Xsim = np.zeros((numdata, seqlen, 1, dims+1))

    for n in range(numdata):
        Xsim_seq = [np.random.choice(dims+1, p=profile[pos]) for pos in range(seqlen)]
        Xsim_hp = np.zeros((seqlen,1, dims+1))
        for (idx,nuc) in enumerate(Xsim_seq):
            Xsim_hp[idx][0][nuc] = 1
        Xsim[n] = Xsim_hp

    return (Xsim)


def seq_bunchshuffle(Xpos, numdata, seqlen, bunchsize=(10, 75)):

    #n = the number of bunches
    smallbunch, largebunch = bunchsize
    n_upper = seqlen//smallbunch
    n_lower = seqlen//largebunch

    Xshuffle = np.zeros((np.shape(Xpos)))
    ns = []
    for seq in range(numdata):
        Xcopy = np.copy(Xpos[seq])

        n = np.random.randint(n_lower, n_upper)

        bunchidx = [i*(seqlen//n) for i in range(n)]
        bunchidx.append(seqlen)

        start=0
        randidx = np.random.permutation(n)
        for i in range(n):
            idx = randidx[i]
            space = bunchidx[idx+1]-bunchidx[idx]
            Xshuffle[seq, start:start+space, :, :] = Xcopy[bunchidx[idx]:bunchidx[idx+1], :, :]
            start = start + space

    return (Xshuffle)

    #First order mutagenesis function
def mutate(sequence, seq_length, dims):
    import numpy as np
    num_mutations = seq_length * dims
    hotplot_mutations = np.zeros((num_mutations,seq_length,1,dims))

    for position in range(seq_length):
        for nuc in range(dims):
            mut_seq = np.copy(sequence)
            mut_seq[0, position, 0, :] = np.zeros(dims)
            mut_seq[0, position, 0, nuc] = 1.0

            hotplot_mutations[(position*dims)+nuc] = mut_seq
    return hotplot_mutations



#def secondorder_mutate(X):
def double_mutate(sequence, seq_length, dims):
    import numpy as np
    mutations_matrix = np.zeros((seq_length,seq_length, dims*dims, seq_length,1,dims))

    for position1 in range(seq_length):

        for position2 in range(seq_length):

            for nuc1 in range(dims):

                for nuc2 in range(dims):

                    mut_seq = np.copy(sequence)
                    mut_seq[0, position1, 0, :] = np.zeros(dims)
                    mut_seq[0, position1, 0, nuc1] = 1.0
                    mut_seq[0, position2, 0, :] = np.zeros(dims)
                    mut_seq[0, position2, 0, nuc2] = 1.0

                    mutations_matrix[position1, position2, (nuc1*dims)+nuc2, :] = mut_seq

    return mutations_matrix


def double_mutate_ungapped(X, ungapped_index):

    num_summary, seqlen, _, dims = X.shape
    idxlen = len(ungapped_index)

    mutations_matrix = np.zeros((idxlen,idxlen, dims*dims, seqlen,1,dims))

    for i1,position1 in enumerate(ungapped_index):

        for i2,position2 in enumerate(ungapped_index):

            for nuc1 in range(dims):

                for nuc2 in range(dims):

                    mut_seq = np.copy(X)
                    mut_seq[0, position1, 0, :] = np.zeros(dims)
                    mut_seq[0, position1, 0, nuc1] = 1.0
                    mut_seq[0, position2, 0, :] = np.zeros(dims)
                    mut_seq[0, position2, 0, nuc2] = 1.0

                    mutations_matrix[i1, i2, (nuc1*dims)+nuc2, :] = mut_seq

    return mutations_matrix



'--------------------------------------------------------------------------------------------------------------------------------'

''' ANALYSIS '''

''' FIRST ORDER MUTAGENESIS '''

def fom_saliency(X, layer, alphabet, nntrainer, sess, title='notitle', figsize=(15,2)):

    ''' requires that deepomics is being used and the appropriate architecture has already been constructed
    Must first initialize the session and set best parameters

    layer is the activation layer we want to use as a string
    figsize is the figure size we want to use'''

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T

    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - predictions[0]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    plt.figure(figsize=figsize)
    if title != 'notitle':
        plt.title(title)
    visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                        norm_heat_mut,
                                        alphabet=alphabet,
                                        nt_width=400)

def fom_saliency_mul(X, layer, alphabet, nntrainer, sess, ax, title='notitle'):

    ''' requires that deepomics is being used and the appropriate architecture has already been constructed
    Must first initialize the session and set best parameters

    layer is the activation layer we want to use as a string
    figsize is the figure size we want to use'''

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T

    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - predictions[0]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                        norm_heat_mut,
                                        alphabet=alphabet,
                                        nt_width=400)

def fom_heatmap(X, layer, alphabet, nntrainer, sess, eps=0):

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T

    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - predictions[0] + eps
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    return (norm_heat_mut)

def fom_neursal(X, layer, alphabet, neuron, nntrainer, sess, title='notitle', figsize=(15,2)
                , fig=None, pos=None, idx=None):

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get the neurons score for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)[:,neuron]

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    dense = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T

    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - dense[:, neuron]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    if fig:
        row, col = pos
        ax = fig.add_subplot(row, col, idx)
        if title != 'notitle':
            ax.set_title(title)
        ax = visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                        norm_heat_mut,
                                        alphabet=alphabet,
                                        nt_width=400)

    else:
        plt.figure(figsize=figsize)
        if title != 'notitle':
            plt.title(title)
        visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                            norm_heat_mut,
                                            alphabet=alphabet,
                                            nt_width=400)

def fom_convsal(X, layer, alphabet, convidx, nntrainer, sess, title='notitle', figsize=(15,2)
                , fig=None, pos=None, idx=None):

    eps = 1e-7

    #choose neuron coordinates within convolution output
    i2, i3, i4 = convidx

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get the neurons score for the mutations
    mut_scores = nntrainer.get_activations(sess, mutations, layer=layer)[:, i2, i3, i4]

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get activations for the WT sequence
    WT_score = nntrainer.get_activations(sess, WT, layer=layer)[:, i2, i3, i4]

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_scores.reshape(X.shape[1],4).T

    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = (heat_mut - WT_score) + eps
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    if fig:
        row, col = pos
        ax = fig.add_subplot(row, col, idx)
        if title != 'notitle':
            ax.set_title(title)
        ax = visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                        norm_heat_mut,
                                        alphabet=alphabet,
                                        nt_width=400)

    else:
        plt.figure(figsize=figsize)
        if title != 'notitle':
            plt.title(title)
        visualize.plot_seq_pos_saliency(np.squeeze(X).T,
                                            norm_heat_mut,
                                            alphabet=alphabet,
                                            nt_width=400)



'--------------------------------------------------------------------------------------------------------------------------------'

''' SECOND ORDER MUTAGENESIS '''



def som_average(X, savepath, nntrainer, sess, progress='on', save=True, layer='output',
                normalize=False, normfactor=None, eps=0):

    num_summary, seqlen, _, dims = X.shape

    sum_mut2_scores = np.zeros((seqlen*seqlen*dims*dims, 1))
    starttime = time.time()

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)

        epoch_starttime = time.time()

        #mutate the sequence
        X_mutsecorder = double_mutate(np.expand_dims(X[ii], axis=0), seqlen, dims)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (seqlen*seqlen*dims*dims, seqlen, 1, dims))
        mutations = {'inputs': X_mutsecorder_reshape, 'targets': np.ones((X_mutsecorder_reshape.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer=layer)

        #if normalize:
            #take the WT and put it into a dict for deepomics
            #WT = {'inputs': np.expand_dims(X[ii], axis=0), 'targets': np.ones((np.expand_dims(X[ii], axis=0).shape[0], 1))}
            #Get output or logits activations for the WT sequence
            #predictions = nntrainer.get_activations(sess, WT, layer=layer)

            #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
            #mut2_scores = mut2_scores - predictions[0] + eps
            #mut2_scores = normalize_mut_hol(mut2_scores, factor=normfactor)


        #Sum all the scores into a single matrix
        sum_mut2_scores += mut2_scores

        epoch_endtime = time.time()

        if progress == 'on':

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()

    if progress == 'off':
        print ('----------------Summing complete----------------')

    # Save the summed array for future use
    if save == True:
        np.save(savepath, sum_mut2_scores)
        print ('Saving scores to ' + savepath)

    return (sum_mut2_scores)

def som_average_large(Xdict, savepath, nntrainer, sess, progress='on', save=True, layer='output',
                      normalize=False, normfactor=None, eps=0):

    num_summary, seqlen, _, dims = Xdict.shape

    sum_mut2_scores = np.zeros((seqlen*seqlen*dims*dims, 1))
    starttime = time.time()

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)

        epoch_starttime = time.time()

        #extract sequence
        X = np.expand_dims(Xdict[ii], axis=0)

        #Mutate the sequence and put its activation into an array
        mut2_scores = np.zeros(shape=(seqlen, seqlen, dims, dims))

        for position1 in range(seqlen):

            for position2 in range(seqlen):

                for nuc1 in range(dims):

                    for nuc2 in range(dims):

                        mut_seq = np.copy(X)
                        mut_seq[0, position1, 0, :] = np.zeros(dims)
                        mut_seq[0, position1, 0, nuc1] = 1.0
                        mut_seq[0, position2, 0, :] = np.zeros(dims)
                        mut_seq[0, position2, 0, nuc2] = 1.0

                        #put mutant sequence into dict for deepomics
                        mutation = {'inputs': mut_seq, 'targets': np.ones((mut_seq.shape[0], 1))}

                        #Get output activations for the mutations
                        mut2_score= nntrainer.get_activations(sess, mutation, layer=layer)

                        mut2_scores[position1, position2, nuc1, nuc2] = mut2_score

        if normalize:
            #take the WT and put it into a dict for deepomics
            WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
            #Get output or logits activations for the WT sequence
            predictions = nntrainer.get_activations(sess, WT, layer=layer)

            #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
            mut2_scores = mut2_scores - predictions[0] + eps
            mut2_scores = normalize_hol(mut2_scores, factor=normfactor)

        #Sum all the scores into a single matrix
        sum_mut2_scores += mut2_scores.reshape(seqlen*seqlen*dims*dims, 1)

        epoch_endtime = time.time()

        if progress == 'on':

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()

    if progress == 'off':
        print ('----------------Summing complete----------------')

    # Save the summed array for future use
    if save == True:
        np.save(savepath, sum_mut2_scores)
        print ('Saving scores to ' + savepath)

    return (sum_mut2_scores)

#Generic
def som_average_ungapped(Xdict, ungapped_index, savepath, nntrainer, sess, progress='on', save=True, layer='output',
                         normalize=False):

    num_summary, seqlen, _, dims = Xdict.shape

    starttime = time.time()

    idxlen = len(ungapped_index)

    sum_mut2_scores = []

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)

        epoch_starttime = time.time()

        #extract sequence
        X = np.expand_dims(Xdict[ii], axis=0)
        #Get WT score
        WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
        WT_score = nntrainer.get_activations(sess, WT, layer=layer)[0]

        X_mutsecorder = double_mutate_ungapped(X, ungapped_index)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (idxlen*idxlen*dims*dims, seqlen, 1, dims))
        mutations = {'inputs': X_mutsecorder_reshape, 'targets': np.ones((X_mutsecorder_reshape.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer=layer)

        #logodds regime:
        if normalize == 'logodds':
            minscore = np.min(mut2_scores)

            #mut2_scores = np.log(np.clip(mut2_scores, a_min=0., a_max=1e7) + 1e-7) - np.log(WT_score+1e-7)
            mut2_scores = np.log(mut2_scores - minscore + 1e-7) - np.log(WT_score-minscore+1e-7)

        #Sum all the scores into a single matrix
        sum_mut2_scores.append(mut2_scores)

        epoch_endtime = time.time()

        if progress == 'on':

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()

        if progress == 'short':
            if ii%100 == 0:
                print (ii)
                print ('Epoch duration =' + sectotime((epoch_endtime -epoch_starttime)*100))
                print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
                print ()

    print ('----------------Summing complete----------------')

    mean_mut2_scores = np.nanmean(sum_mut2_scores, axis=0)

    # Save the summed array for future use
    if save == True:
        np.save(savepath, mean_mut2_scores)
        print ('Saving scores to ' + savepath)


    return (mean_mut2_scores)


#Implement log odds difference
def som_average_ungapped_logodds(Xdict, ungapped_index, savepath, nntrainer, sess, progress='on', save=True, layer='output',
                         normalize=False, normfactor=None, eps=0):

    num_summary, seqlen, _, dims = Xdict.shape

    starttime = time.time()

    idxlen = len(ungapped_index)

    sum_mut2_scores = []

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)

        epoch_starttime = time.time()

        #extract sequence
        X = np.expand_dims(Xdict[ii], axis=0)
        #Get WT score
        WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
        WT_score = nntrainer.get_activations(sess, WT, layer=layer)[0]

        X_mutsecorder = double_mutate_ungapped(X, ungapped_index)

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

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()

        if progress == 'short':
            if ii%100 == 0:
                print (ii)
                print ('Epoch duration =' + sectotime((epoch_endtime -epoch_starttime)*100))
                print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
                print ()

    print ('----------------Summing complete----------------')

    mean_mut2_scores = np.nanmean(sum_mut2_scores, axis=0)

    # Save the summed array for future use
    if save == True:
        np.save(savepath, mean_mut2_scores)
        print ('Saving scores to ' + savepath)


    return (mean_mut2_scores)

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



def normalize_mut_hol(hol_mut, WTmean=False, maxreduce=True, normfactor=None, withmean=False):
    norm_hol_mut = np.copy(hol_mut)
    for one in range(hol_mut.shape[0]):
        for two in range(hol_mut.shape[0]):
            if withmean:
                norm_hol_mut[one, two] = normalize_hol(hol_mut[one, two] - WTmean, maxreduce, factor=normfactor)
            else:
                norm_hol_mut[one, two] = normalize_hol(hol_mut[one, two], maxreduce, factor=normfactor)
    return norm_hol_mut


def som_neuronaverage(X, layer, neuron, savepath, nntrainer, sess, progress='on', save=True):

    num_summary, seqlen, _, dims = X.shape

    sum_mut2_scores = np.zeros((seqlen*seqlen*dims*dims,))
    starttime = time.time()

    for ii in range(num_summary):
        if progress == 'on':
            print (ii)

        epoch_starttime = time.time()

        #mutate the sequence
        X_mutsecorder = double_mutate(np.expand_dims(X[ii], axis=0), seqlen, dims)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (seqlen*seqlen*dims*dims, seqlen, 1, dims))
        mutations = {'inputs': X_mutsecorder_reshape, 'targets': np.ones((X_mutsecorder_reshape.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer=layer)[:,neuron]

        #Sum all the scores into a single matrix
        sum_mut2_scores += mut2_scores

        epoch_endtime = time.time()

        if progress == 'on':

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()

    if progress == 'off':
        print ('----------------Summing complete----------------')

    # Save the summed array for future use
    if save == True:
        np.save(savepath, sum_mut2_scores)
        print ('Saving scores to ' + savepath)

    return (sum_mut2_scores)






def square_holplot(mutations, num, alphabet, limits=(0., 1.0), title=False, cmap ='Blues',
                   figsize=(15,14), lines=True, start=(4,22), reverse=False, cbar=False):

    if alphabet == 'rna':
        nuc = ['A', 'C', 'G', 'U']
    if alphabet == 'dna':
        nuc = ['A', 'C', 'G', 'T']

    if lines == True:
        linewidths = 0.1
    else:
        linewidths = 0.

    if limits == False:
        vmin, vmax = (None, None)
    else:
        vmin, vmax = limits

    start_1, start_2 = start

    fig = plt.figure(figsize=figsize)
    for one in range(num):
        for two in range(num):
            ax = fig.add_subplot(num, num, ((one*num)+two)+1)

            #plot the 0th column with row labels and the num_th most row with column labels
            if two == 0:
                if one == (num-1):
                    xtick=nuc
                    ytick=nuc
                else:
                    xtick=[]
                    ytick=nuc
            else:
                if one == (num-1):
                    xtick=nuc
                    ytick=[]
                else:
                    xtick=[]
                    ytick=[]

            if reverse == True:
                yy = one + start_1
                xx = start_2 - two
            if reverse == False:
                yy = one + start_1
                xx = two + start_2

            ax = sb.heatmap(mutations[yy, xx], vmin=vmin, vmax=vmax, cmap=cmap,
                            linewidths=linewidths, linecolor='black', xticklabels=xtick, yticklabels=ytick,
                            cbar=cbar)
            #plot titles
            if title == True:
                if one == 0:
                    ax.set_title(str(xx))
                if two == 0:
                    ax.set_ylabel(str(yy))


def symlinear_holplot(mutations, figplot, alphabet, start_stop=(0,20), limits=(0., 1.),
                      cmap ='Blues', figsize=(10,7), lines=True):

    if alphabet == 'rna':
        nuc = ['A', 'C', 'G', 'U']
    if alphabet == 'dna':
        nuc = ['A', 'C', 'G', 'T']

    row, col = figplot
    start, end = start_stop

    if lines == True:
        linewidths = 0.1
    if lines == False:
        linewidths = 0.

    if limits == False:
        vmin, vmax = (None, None)
    else:
        vmin, vmax = limits

    fig = plt.figure(figsize=figsize)
    for ii in range(row*col):
        ax = fig.add_subplot(row,col,ii+1)
        ax.set_title(str(ii)+','+str(end-ii))
        ax = sb.heatmap(mutations[ii, end-ii], vmin=vmin, vmax=vmax, cmap='Blues', linewidths=linewidths, linecolor='black', xticklabels=nuc, yticklabels=nuc)


''' CONVOLUTIONS '''


def convprogression(data, L, layers, seq, order, sess, nntrainer, figsize=(10,2), exceptmean=None):
    for layer in L:
        layer = layers[layer]
        conv = nntrainer.get_activations(sess, data, layer=layer)
        conv_plot = np.squeeze(conv[seq])
        plt.figure(figsize=figsize)
        if layer == exceptmean:
            square = False
        else:
            square = True
        sb.heatmap(conv_plot[:, order], square=square, xticklabels=[])
        plt.title(layer)
        plt.xlabel('filters')
        plt.ylabel('Pooled sequence')
        plt.show()

'--------------------------------------------------------------------------------------------------------------------------------'

''' STOCKHOLM UTILITIES '''
#build model from seed
#! cmbuild ../../data_RFAM/RF01739.cm ../../data_RFAM/seeds/RF01739.sto

#emit sequences from alignment
#! cmemit -a -N 100 --seed 274 -o ../../data_RFAM/glnAsim_100.sto ../../data_RFAM/RF01739.cm

#open file
#from Bio import AlignIO
#filename = '../../data_RFAM/glnAsim_100.sto'
#alignment = AlignIO.read(open(filename), "stockholm")

#Save dictionaries into h5py files
#hdf5path = '../../../data_RFAM/glnAsim_100k_t2.hdf5'
#with h5py.File(hdf5path, 'w') as f:
#    f.create_dataset('X_data', data=X_data)
#    f.create_dataset('Y_data', data=Y_data)


def sto_onehot(simalign_file, alphabet, gaps=True):
    from Bio import AlignIO
    alignment = AlignIO.read(open(simalign_file), "stockholm")
    sequences = []
    for record in alignment:
        sequences.append(record.seq)
    X_data = seq_onehot(sequences=sequences, alphabet=alphabet, gaps=gaps)
    return (X_data)

def sto_sequences(simalign_file, alphabet, gaps=True):
    from Bio import AlignIO
    alignment = AlignIO.read(open(simalign_file), "stockholm")
    sequencesraw = []
    for record in alignment:
        sequencesraw.append(record.seq)
    sequences = []
    for seq in sequencesraw:
        seqstr = ''
        for j in seq:
            seqstr = seqstr + j
        sequences.append(seqstr)
    return (sequences)


#get SS consensus
def getSSconsensus(simalign_file):
    SS = ''
    with open(simalign_file) as f1:
        for line in f1:
            if '#=GC SS_cons' in line:
                line = line.strip()
                line = line.split()
                SS = SS + line[-1]
    return (SS)

def getSQconsensus(simalign_file):
    SQ = ''
    with open(simalign_file) as f1:
        for line in f1:
            if '#=GC RF' in line:
                line = line.strip()
                line = line.split()
                SQ = SQ + line[-1]
    return (SQ)

def rm_consensus_gaps(X_data, SS, gapchar=['.', '~']):
    idx = []
    ungappedSS = ''
    for i,s in enumerate(SS):
        if s not in  gapchar:
            idx.append(i)
            ungappedSS = ungappedSS + s

    return (X_data[:, idx, :], ungappedSS, idx)


def consensus_gaps(SS, gapchar=['.', '~']):
    idx = []
    for i,s in enumerate(SS):
        if s in gapchar:
            idx.append(i)
    return (idx)

def sigbasepair(SS, bpchar):
    bpchar = ['(',')','<','>','{','}','[',']']
    nonbpidx=[]
    bpidx = []
    bpSS = ''
    for i,s in enumerate(SS):
        if s in bpchar:
            bpidx.append(i)
            bpSS = bpSS + s
        else:
            nonbpidx.append(i)
    return (bpidx, bpSS, nonbpidx)

def bpug(ugidx, bpidx, SQ):
    bpugSQ = ['.' for i in range(len(ugidx))]
    bpugidx = []
    for i, ix in enumerate(ugidx):
        if ix in bpidx:
            bpugSQ[i] = SQ[ix]
            bpugidx.append(i)
    return (bpugSQ, bpugidx)



''' UTILITIES '''

rnadict = {'A':0, 'C':1, 'G':2, 'U':3, 'a':0, 'c':1, 'g':2, 'u':3, '-':4, '.':4}#, 'N':4, 'n':4, 'R':4}
dnadict = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3, '-':4, '.':4}#, 'N':4, 'n':4, 'R':4}

def seq_onehot(sequences, alphabet, gaps=True):
    if alphabet == 'rna':
        ndict = rnadict
        dims = 4
    if alphabet == 'dna':
        ndict = dnadict
        dims = 4
    onehot = np.zeros((len(sequences), len(sequences[0]), dims+1))
    for i1,seq in enumerate(sequences):
        for i2, n in enumerate(seq):
            try:
                onehot[i1][i2][ndict[n]] = 1.
            except KeyError:
                onehot[i1][i2][ndict[code_convert(n)]] = 1.
    if gaps == False:
            onehot = onehot[:, :, :dims]
    return onehot

def code_convert(n):
    if n == 'N' or n == 'n':
        newn = np.random.choice(['a','u','c','g'])
    else:
        newn = '-'
    return (newn)


def sectotime(t):
    t = np.around(t, 2)
    if t>=3600.:
        s = t%60
        m = ((t)%3600)//60
        hr = t//3600
        output = str(int(hr)) + 'hr ' + str(int(m)) + 'min ' + str(s) + 's'
    else:
        if t>=60.:
            s = t%60
            m = t//60
            output = str(int(m)) + 'min ' + str(s) + 's'
        else:
            s = np.copy(t)
            output = str(s) + 's'
    return(output)


def showlayers(nnmodel):
    return (list(enumerate(nnmodel.network.keys())))


def nucpresence(Xdict, ii):
    #creates an array with vectors holding the presence or absence of a nucleotide
    #to be used easily with a list comprehension
    X = np.expand_dims(Xdict['inputs'][plot_index[ii]], axis=0)
    nuc_index = np.where(np.sum(X, axis=0)!=0)[0]
    nucpos = np.zeros(shape=(1,seqlen))
    nucpos[:, nuc_index] = 1.
    return nucpos


def normalize_hol(hol, maxreduce=True, factor=None):

    if maxreduce:
        MAX = np.max(np.abs(hol))
        hol = hol/MAX
    if factor:
        hol = np.exp(hol*factor)
    norm = np.sum(np.abs(hol))
    return (hol/norm)



'''  HOLISTIC TO BLOCK

blocklen = np.sqrt(np.product(meanhol_mut2.shape)).astype(int)
S = np.zeros((blocklen, blocklen))
i,j,k,l = meanhol_mut2.shape

for ii in range(i):
    for jj in range(j):
        for kk in range(k):
            for ll in range(l):
                S[(4*ii)+kk, (4*jj)+ll] = meanhol_mut2[ii,jj,kk,ll]

plt.figure(figsize=(15,15))
plt.imshow(S,  cmap='RdPu')
plt.colorbar()

'''








#bpugSQ = ['.' for i in range(len(ugidx))]
#for i, ix in enumerate(ugidx):
#    if ix in bpidx:
#        bpugSQ[i] = SQ[ix]
