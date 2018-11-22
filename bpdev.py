from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import sys
sys.path.append('../../..')
import mutagenesisfunctions as mf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency
import time as time
import pandas as pd

def bp_coords(ugSS):
    '''
    Function that takes in an ungapped Sequence string and
    outputs a list of lists with the coordinates base pairs.
    Optionally it can also output the list extended with the
    reflections of the coordinates for use with holistics
    plots.
    '''

    #identify if we're dealing with pk or nested
    if type(ugSS) == str:
        ugSS = [ugSS]
    if type(ugSS) == list:
        ugSS = ugSS

    bp_openers = ['(', '<', '{']
    bp_closers = [')', '>', '}']

    basepairs = [] #list to hold the base pair coords

    for SS in ugSS:
        opened = np.array([]) # holds the integers of chars and keeps track of how close they are to being closed
        counter = 0
        for char in SS:

            if char in bp_openers:
                #open a base pair and start counting till its closed
                opened = np.append(opened, 0)
                opened += 1

            elif char in bp_closers:
                #get closer to closing if we find a closing bracket
                opened -= 1
                if 0 in opened:
                    #check if we've successfuly closed a pair
                    op = np.where(opened ==0)[0][0]
                    basepairs.append([op, counter]) #add the pair to our list
                    opened[np.where(opened ==0)] = 1000 # make the recently closed char negligible
                opened = np.append(opened, 1000) #treat closing brackets as negligible


            else:
                opened = np.append(opened, 1000) #non-base-paired chars are negligible

            counter += 1

    basepairs = np.asarray(basepairs)

    #reflect
    reflect = basepairs[:, ::-1]
    basepairs = np.vstack([basepairs, reflect])

    return (basepairs)


def KLD(hol, ref):
    S = np.ravel(hol)
    R = np.ravel(ref)
    dkl = np.sum([S[i]*(np.log(S[i]+1e-15)-np.log(R[i]+1e-15)) for i in range(len(S))])
    return (dkl)

def KLD_hol(hol_mut, ref):
    KLD_scores = np.zeros((hol_mut.shape[0], hol_mut.shape[0]))
    for one in range(hol_mut.shape[0]):
        for two in range(hol_mut.shape[0]):
            KLD_scores[one, two] = KLD(makeprob(hol_mut[one, two]), ref)
    return (KLD_scores)

def makeprob(hol):
    norm = np.sum(np.abs(hol))
    return (hol/norm)

def bp_probmx():
	bpfilter = np.ones((4,4))*0
	for i,j in zip(range(4), range(4)):
	    bpfilter[i, -(j+1)] = 0.25
	return (bpfilter)


#Plots and returns the average holistic matrix of all the truly base paired nucs.
def avgholbp(ugSS, numbp, dims, meanhol_mut2):
    #Get the base pair coords
    bp_rc = bp_coords(ugSS)

    #pull out the base pairs from the holistic scores array
    bp_hols = np.zeros((numbp, dims, dims))

    for i,r in enumerate(bp_rc):
        bp_hols[i] = meanhol_mut2[r[0],r[1]]

    bp_hols_avg = np.mean(bp_hols, axis=0)
    plt.figure()
    sb.heatmap(bp_hols_avg)
    plt.show()
    return (bp_hols_avg)


#extract the wc scores from a savefile and optionally plot them
def get_wc(savefile, numug, dims, bpugSQ, plotit=True, normalize=True, cmap='RdPu'):
    mean_mut2 = np.load(savefile)
    #Reshape into a holistic tensor organizing the mutations into 4*4 matrices
    try:
        meanhol_mut2 = mean_mut2.reshape(numug,numug,dims,dims)
    except ValueError:
        print ('SoM scores already reshaped')
        meanhol_mut2 = mean_mut2

    if normalize:
        meanhol_mut2 = mf.normalize_mut_hol(meanhol_mut2, maxreduce=True, normfactor = 1)

    bpfilter = np.ones((4,4))*0
    for i,j in zip(range(4), range(4)):
        bpfilter[i, -(j+1)] = 1.

    C = np.sum((meanhol_mut2*bpfilter).reshape(numug,numug,dims*dims), axis=2)

    if plotit:
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        sb.heatmap(C, xticklabels=bpugSQ, yticklabels=bpugSQ, vmin=None, cmap=cmap, linewidth=0.0)
        plt.title('Base Pair score for the ungapped consensus regions given by infernal')
        plt.xlabel('Ungapped nucleotides: pos 1')
        plt.ylabel('Ungapped nucleotides: pos 2')
    return (C, meanhol_mut2)

#returns a matrix in the shape wc of all the true base pairs
def get_true(ugSS, numug):
    #Get the base pair coords
    bp_rc = bp_coords(ugSS)
    #make sure that these are what we expect
    s = np.ones((numug, numug))*0
    for r in bp_rc:
        s[r[0], r[1]] = 1
    return (s)

#Plots and returns (s) all the truly base paired nucs
def plot_true(ugSS, bpugSQ, numug):
    s = get_true(ugSS, numug)

    #visualize in the same shape as a wc plot
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sb.heatmap(s, xticklabels=bpugSQ, yticklabels=bpugSQ, vmin=None, cmap='RdPu', linewidth=0.0)
    plt.title('True base pairs taken from stockholm SS')
    plt.xlabel('Ungapped nucleotides: pos 1')
    plt.ylabel('Ungapped nucleotides: pos 2')

#Plot the highest numbp ranked wc scores in the same shape as wc to see where they are
def plot_wcrank(C, numug, numbp, cmap='Greens'):
    #unravel the wc plot
    bp_stretch = np.ravel(C)
    #get the highest scores indices
    bp_index = np.argsort(bp_stretch)[::-1]
    #get the ranking of each score
    bp_rank = np.argsort(bp_index)
    #reshape into the wc shape
    bp_rank = bp_rank.reshape(C.shape)
    #open a new matrix that will show the coloring of the ranks in two groups
    bp_shade = np.zeros_like(bp_rank)
    #shade in the high ranked
    bp_shade[np.where(bp_rank<numbp)] = 1.

    #visualize
    plt.figure()
    sb.heatmap(bp_shade, cmap=cmap)
    plt.title('The top numbp scores from the model')
    plt.show()


#Plot a graph that shows how many true base pairs the model has learned at a particular rank
def plot_rankprogress(C, ugSS, numbp, numug):
    s = get_true(ugSS, numug)
    bp_stretch = np.ravel(C)
    ac_stretch = np.ravel(s)
    #sort the unraveled wc scores
    bp_index = np.argsort(bp_stretch)[::-1]
    #get the cummulative score of the ranks
    hits = np.cumsum(ac_stretch[bp_index])
    plt.figure()
    plt.plot(range(len(bp_stretch)), hits)
    plt.title('Number of true base pairs the model has found at that rank')
    plt.xlabel('Rank of the WC scores')
    plt.ylabel('Number of true base pairs found')
    plt.show()


def plotKLDscores_toy(meanhol_mut2, cmap='RdPu'):
	K = KLD_hol(meanhol_mut2, bp_probmx())

	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	sb.heatmap(K, cmap=cmap, linewidth=0.0)
	plt.title('Base Pair score for the ungapped consensus regions given by infernal')
	plt.xlabel('Ungapped nucleotides: pos 1')
	plt.ylabel('Ungapped nucleotides: pos 2')
	plt.show()



def plotKLDscores(meanhol_mut2, bpugSQ, bpSS, cmap='RdPu'):
	K = KLD_hol(meanhol_mut2, bp_probmx())

	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	sb.heatmap(K, xticklabels=bpugSQ, yticklabels=bpugSQ, vmax=None, cmap=cmap, linewidth=0.0)
	plt.title('Base Pair score for the ungapped consensus regions given by infernal')
	plt.xlabel('Ungapped nucleotides: pos 1')
	plt.ylabel('Ungapped nucleotides: pos 2')
	plt.show()
	plt.subplot(1,2,2)
	sb.heatmap(K[bpugidx][:, bpugidx], xticklabels=bpSS, yticklabels=bpSS, vmax=None, cmap=cmap, linewidth=0.)
	plt.title('Base Pair score for the base paired consensus regions given by infernal')
	plt.xlabel('Base Paired nucleotides: pos 1')
	plt.ylabel('Base Paired nucleotides: pos 2')
	plt.show

#Method 1: in the places where there are supposed to be bp, what is the total score
#given a watson crick plot, pull out the bpcoords and add up the total KLD_scores
def bp_totscore(ugSS, C, numug):
    #first normalize the scores by dividing them by the max
    C = C - np.mean(C)
    C = C/np.max(C)

    totscore = 0
    #Now pull out the bpcoords
    bp_rc = bp_coords(ugSS)
    for r,c in bp_rc:
        totscore += C[r,c]
    return (totscore/numug)

#Method 1: in the top numbp scores, how many of those are actually bps
#Given a wc plot, order it score, are the top numbp scores actually base paired?
def bp_ppv(C, ugSS, numbp, numug, trans=False):
    s = get_true(ugSS, numug)
    bp_stretch = np.ravel(C)
    ac_stretch = np.ravel(s)
    #sort the unraveled wc scores
    bp_index = np.argsort(bp_stretch)[::-1]
    #get the total number of true positives from the first numbp (what we call positive)
    #divide it by the number of positives (numbp)
    PPV = np.sum(ac_stretch[bp_index][:numbp])/(numbp)
    if trans:
        PPV = np.sum(ac_stretch[bp_index][20:20+numbp])/(numbp)
    return (PPV)
