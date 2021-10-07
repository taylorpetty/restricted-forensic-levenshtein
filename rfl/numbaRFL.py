# -*- coding: utf-8 -*-

import pandas as pd
import itertools
from weighted_levenshtein import lev
import time
import os
from rfl_module import rfl
from lsdp_module import lsdp
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.typed import Dict

lettertodigit = {'A':1, 'C':2, 'G':3, 'T':4}
digittoletter = {1:'A', 2:'C', 3:'G', 4:'T'}

def encode(dna):
    '''Turns a DNA string into a single integer.
    
    The letters A-T encode as the integers 1-4.'''
    
    return int(''.join([str(lettertodigit[letter]) for letter in dna]))

def encodeasarray(dna,dtype = 'np.int64'):
    '''Turns a DNA string into a NumPy array of ints.
    
    E.g. ACTTG returns as array([1,2,4,4,3]).
    
    The dtype option almost certainly isn't needed.'''
    if dtype == 'np.int64':
        return np.asarray([int(lettertodigit[letter]) for letter in dna],
                          dtype = np.int64)
    elif dtype == 'np.float64':
        return np.asarray([int(lettertodigit[letter]) for letter in dna],
                          dtype = np.float64)
    else:
        raise ValueError("Invalid dtype.")

#this needs to be @njit-ified to be compatible with the full function later
@njit
def arraytoint(intarray):
    '''Numba-compatible: turn an array of ints
    into a concatenated int.
    
    Do not enter or expect floats, unless floor(x) == x.
    
    This will easily overflow the computer for even moderately
    long arrays. We only need it to go up to 9 characters.'''
    ndigits = len(intarray)
    val = 0
    for i in range(ndigits):
        val = val + intarray[i]*pow(10,ndigits-i-1)
    return val

def lsdp_encoded(motif, costs, fwstcost, bwstcost):
    '''Standard LSDP but with encoded keys for numba.
    
    Motif is a string.
    Costs is a 5x5 panda with indices and columns named
    '', 'A', 'C', 'G', and 'T', so costs.loc[x,y] is the
    cost of SNPing, inserting, or deleting from x to y.
    
    fwstcost and bwstcosts are floats.
    
    Returns tuple of (Numba-compatible)
    forward and backward stutter dictionaries for all strings
    of length 1 to 2k-1, where k = len(motif).'''
    
    dfw = Dict.empty(key_type=types.int64,
                      value_type=types.float64)
    dbw = Dict.empty(key_type=types.int64,
                      value_type=types.float64)
    # dbw = Dict.empty(key_type=types.int64,
    #                  value_type=types.float64)
    icosts = np.ones(128, dtype=np.float64)
    dcosts = np.ones(128, dtype=np.float64)
    scosts = np.ones((128, 128), dtype=np.float64)

    alphabet = ['A', 'C', 'G', 'T']
    
    for letter in alphabet:
        
        icosts[ord(letter)] = costs.loc['',letter]
        dcosts[ord(letter)] = costs.loc[letter,'']
        
        for letter2 in alphabet:
            if letter2 != letter:
                scosts[ord(letter),ord(letter2)] = costs.loc[letter,letter2]
                
    for i in range(1,2*len(motif)):
        for item in itertools.product(alphabet,repeat=i):
            sewn = ''.join(item)            
            fwcost = fwstcost + lev(motif,sewn,insert_costs=icosts,delete_costs=dcosts,substitute_costs=scosts)
            dfw[encode(sewn)] = fwcost            
            bwcost = bwstcost + lev(sewn,motif,insert_costs=icosts,delete_costs=dcosts,substitute_costs=scosts)
            dbw[encode(sewn)] = bwcost
    return dfw, dbw

##################

###BABYRFLOPT IS SINGLE-MOTIF ONLY

##################

@njit
def babyrflopt(parent,child,peek,singlecharcosts,
               costdictfw,costdictbw):
    '''For a single-repeat locus, this returns the Restricted
    Forensic Levenshtein quasi-distance from parent to child/
    
    parent and child are NumPy arrays of ints.
    singlecharcosts is a 5x5 NumPy array of floats.
    The cost dicts are the outputs of lsdp_encoded.'''
    lp = len(parent)
    lc = len(child)
    d = np.zeros((lp+1,lc+1),dtype=np.float64)
    
    #d[0,0] will always be 0 (the cost of going from
    #an empty string to an empty string)
    
    ###FIRST ROW
    #d[0,i] = insert cost of first i child chars
    #i range from 1 to len(child) inclusive
    for i in range(1,lc+1): #for every column in the 1st row, build d[0,i].  
        #build d[0,i]
        
        #we look back no further than peek # of characters.
        #if peek is 5 (motif length 3) and we're at index 10,
        #we want to look back 5.
        #if peek is 5 and we're at index 5, we want to look back
        #5. At index 2 we want to look back 2, since
        #it'll be d[0,0] + cost of inserting child[0:2]
        lookback = min(peek,i)   
        
        #j goes from 1 to lookback+1 and we select child[i-j:i]
        #so if lookback = i then we'd have child[i-i:i]
        
        #cost possibilities: either insert directly (+1),
        #or use the insert dictionary. The number of possible
        #costs from dictionary to check is lookback.
        #(here we rely on single-motif assumption)
        #(if more than one motif, how to know whether
        #substring will be in dict? it'll be based on length.
        #if len(substring) <= len(2*motif-1), access it.
        #problem later for numba: dict of dicts)
        options = np.zeros(lookback+1, dtype=np.float64)#, dtype=np.uint8)
        
        #first option is insertion. look at previous entry
        #in row, then add cost of inserting the ith child char
        options[0] = d[0,i-1]+singlecharcosts[0,int(child[i-1])]
        
        #ranges from 1 up to min(lookahead,i), inclusive
        for j in range(1,lookback+1): #for every trailing string up to either the beginning of the string or the peek amount, whichever is smaller
        
        ###INDEXING NOTE
        #at d[0,3], we're working with first 3 chars of child
        #i.e. child[0], child[1], child[2],
        #i.e. child[0:3]
        ###
        
            #select the previous j characters,
            #INCLUDING THE CURRENT WORKING STATE at column i
            
            #j=1: s = child[i-1:i] (add ith char via dict)
            #j=2: s = child[i-2:i] (add prev 2 chars via dict)
            #...
            #j=n: s = child[i-n:i] (add last n...)
            
            s = arraytoint(child[i-j:i])
            options[j] = costdictfw[s] + d[0,i-j]
    
        d[0,i] = np.amin(options) #take the minimum cost
    
    ###FIRST COLUMN
    #d[i,0] = delete cost of first i chars of parent
    #       = delete cost of parent[0:i]
    #       = del cost of parent[0]...parent[i-1]
    for i in range(1,lp+1): #for every column in the 1st row, build d[0,i].  
        #build d[i,0]
        
        lookback = min(peek,i)    
        
        options = np.zeros(lookback+1)#, dtype=np.uint8)
        
        #first option is insertion. look at previous entry
        #in row, then add cost of inserting the ith child char
        options[0] = d[i-1,0]+singlecharcosts[int(parent[i-1]),0]
        
        #ranges from 1 up to min(lookahead,i), inclusive
        for j in range(1,lookback+1): #for every trailing string up to either the beginning of the string or the peek amount, whichever is smaller
        
        ###INDEXING NOTE
        #at d[0,3], we're working with first 3 chars of child
        #i.e. child[0], child[1], child[2],
        #i.e. child[0:3]
        ###
        
            #select the previous j characters,
            #INCLUDING THE CURRENT WORKING STATE at column i
            
            #j=1: s = child[i-1:i] (add ith char via dict)
            #j=2: s = child[i-2:i] (add prev 2 chars via dict)
            #...
            #j=n: s = child[i-n:i] (add last n...)
            
            s = arraytoint(parent[i-j:i])
            options[j] = costdictbw[s] + d[i-j,0]
    
        d[i,0] = np.amin(options) #take the minimum cost
        
    ###FILLING IN THE REST OF THE MATRIX
    for m in range(1,lp+1):
        for n in range(1,lc+1):
            #build d[m,n]
            
            #we can look at SNPing, inserting, deleting,
            #or inserting up to peek # previous
            #or deleting up peek # previous
            
            rowlook = min(peek,m)
            collook = min(peek,n)
            
            rowlookoptions = np.zeros(rowlook, dtype=np.float64)
            collookoptions = np.zeros(collook, dtype=np.float64)
            standardoptions = np.zeros(3)
            
            #SNP
            standardoptions[0] = d[m-1,n-1] + singlecharcosts[int(parent[m-1]),int(child[n-1])]
            #insert
            standardoptions[1] = d[m,n-1] + singlecharcosts[0,int(child[n-1])]
            #delete
            standardoptions[2] = d[m-1,n] + singlecharcosts[int(parent[m-1]),0]
                
            #insert quasi motif (first k chars of child[:n])
            #check back a few columns, same row
            for k in range(1,collook+1):
                s = arraytoint(child[n-k:n])
                cost = d[m,n-k] + costdictfw[s]
                collookoptions[k-1] = cost
                
            #delete quasi motif
            #check back a few rows, same column
            for k in range(1,rowlook+1):
                s = arraytoint(parent[m-k:m])
                cost = d[m-k,n] + costdictbw[s]
                rowlookoptions[k-1] = cost
            
            mins = np.array([np.amin(standardoptions),
                             np.amin(collookoptions),
                             np.amin(rowlookoptions)])
            d[m,n] = np.amin(mins)
    
    return d[lp,lc]        
