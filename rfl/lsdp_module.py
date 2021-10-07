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
    
    alphabet = ['A','C','G','T']
    
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
            dfw[sewn] = fwcost            
            bwcost = bwstcost + lev(sewn,motif,insert_costs=icosts,delete_costs=dcosts,substitute_costs=scosts)
            dbw[sewn] = bwcost
            
    return {'insert cost':dfw,'delete cost':dbw}
