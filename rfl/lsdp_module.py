# -*- coding: utf-8 -*-

import numpy as np
import itertools
from weighted_levenshtein import lev

def lsdp(motif, costs, fwstcost, bwstcost):
    '''Returns a dictionary of insert and delete costs
    for all strings from length 1 to 2k-1, where k = len(motif).
    
    The cost of insertion is assumed to be based on a first edit
    of forward stutter of the motif, followed by SNPs and indels.
    
    The cost of deletion is assumed to be based on edits of the
    entry back to the motif, then lastly backward stutter of the
    motif.'''
    dfw = dict()
    dbw = dict()
    icosts = np.ones(128,dtype=np.float64)
    dcosts = np.ones(128,dtype=np.float64)
    scosts = np.ones((128, 128), dtype=np.float64)

    icosts[ord('A')] = costs.loc['','A'] 
    icosts[ord('C')] = costs.loc['','C']
    icosts[ord('G')] = costs.loc['','G']
    icosts[ord('T')] = costs.loc['','T']
    
    dcosts[ord('A')]= costs.loc['A','']
    dcosts[ord('C')]= costs.loc['C','']
    dcosts[ord('G')]= costs.loc['G','']
    dcosts[ord('T')]= costs.loc['T','']
    
    scosts[ord('A'), ord('C')] = costs.loc['A','C'] 
    scosts[ord('A'), ord('G')] = costs.loc['A','G']
    scosts[ord('A'), ord('T')] = costs.loc['A','T']
    
    scosts[ord('C'), ord('A')] = costs.loc['C','A']
    scosts[ord('C'), ord('G')] = costs.loc['C','G']
    scosts[ord('C'), ord('T')] = costs.loc['C','T']
    
    scosts[ord('G'), ord('A')] = costs.loc['G','A']
    scosts[ord('G'), ord('C')] = costs.loc['G','C']
    scosts[ord('G'), ord('T')] = costs.loc['G','T']
    
    scosts[ord('T'), ord('A')] = costs.loc['T','A']
    scosts[ord('T'), ord('C')] = costs.loc['T','C']
    scosts[ord('T'), ord('G')] = costs.loc['T','G']
    alphabet = ['A', 'C', 'G', 'T']
    for i in range(1,2*len(motif)):
        for item in itertools.product(alphabet,repeat=i):
            sewn = ''.join(item)            
            fwcost = fwstcost + lev(motif,sewn,insert_costs=icosts,delete_costs=dcosts,substitute_costs=scosts)
            dfw[sewn] = fwcost            
            bwcost = bwstcost + lev(sewn,motif,insert_costs=icosts,delete_costs=dcosts,substitute_costs=scosts)
            dbw[sewn] = bwcost
    return {'insert cost':dfw,'delete cost':dbw}
