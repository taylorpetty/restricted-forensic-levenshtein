# -*- coding: utf-8 -*-

###IN THIS VERSION:
    #indexing is backwards.
    #matrix entry d[i,j] is distance between LAST i characters
    #of parent and LAST j characters of child. Final answer
    #will be the same, but the full matrix will look different,
    #if printed.

def rfl(parent,child,motifs_and_st_costs,single_char_costs,costdict):
    '''Returns Restricted Forensic Levenshtein quasi-distance
    directed from parent to child.
    
    Parent and child are strings.
    Motifs_and_st_costs is a dict of dicts. The keys are motifs,
    and the values are dictionaries of fwst costs and bwst costs.
    
    Costdict is output from lsdp.
    
    single_char_costs is a 5x5 panda with named indices.'''
    lp = len(parent)
    lc = len(child)
    max_motif_length = max([len(motif) for motif in motifs_and_st_costs])
    lookahead = 2*max_motif_length-1    
    d = np.zeros((lp+1,lc+1))
    ###FILL IN FIRST ROW WITH INSERT COSTS
    for i in range(1,lc+1): #for every column in the 1st row, build d[0,i].  
        options = [] #build a list of cost possibilities
        options.append(d[0,i-1]+single_char_costs.loc['',child[lc-i]])
        for j in range(1,min(lookahead,i)+1): #for every trailing string up to either the beginning of the string or the lookahead amount, whichever is smaller
            s = child[lc-i:lc-i+j] #starts at ith from last, goes j chars
            for motif in motifs_and_st_costs: #for every motif we have,
                if s in costdict[motif]['insert cost']:
                    options.append(costdict[motif]['insert cost'][s]
                                   + d[0,i-j])
        d[0,i] = min(options) #take the minimum cost    
    ###FILL IN FIRST COLUMN WITH DELETE COSTS
    for i in range(1,lp+1): #for every row in the dist matrix,
        options = [] #build a list of possible costs
        options.append(d[i-1,0]+single_char_costs.loc[parent[lp-i],''])
        for j in range(1,min(lookahead,i)+1): #for every trailing string up to either the beginning of the string or the lookahead amount, whichever is smaller
            s = parent[lp-i:lp-i+j] #starts at ith from end, goes j chars
            for motif in motifs_and_st_costs: #for every motif we have,
                if s in costdict[motif]['delete cost']:
                    options.append(costdict[motif]['delete cost'][s]
                                   + d[i-j,0])
        d[i,0] = min(options) #take the minimum cost
    ###FILL IN THE REST OF THE MATRIX
    for m in range(1,lp+1):
        for n in range(1,lc+1):
            options = []        
            if parent[lp-m] == child[lc-n]:
                snpcost = 0
            else:
                snpcost = single_char_costs.loc[parent[lp-m],child[lc-n]]
            
            options.append(d[m-1,n-1]+snpcost)
            options.append(d[m,n-1]+single_char_costs.loc['',child[lc-n]])
            options.append(d[m-1,n]+single_char_costs.loc[parent[lp-m],''])
            for i in range(1,min(lookahead,n)+1): #for all lengths of look-back
                s = child[lc-n:lc-n+i]
                for motif in motifs_and_st_costs: #for every motif,
                    if s in costdict[motif]['insert cost']: #if the string isn't too long for the motif,
                        options.append(d[m,n-i]+costdict[motif]['insert cost'][s]) #add the cost
            for i in range(1,min(lookahead,m)+1):
                s = parent[lp-m:lp-m+i]
                for motif in motifs_and_st_costs:
                    if s in costdict[motif]['delete cost']:
                        options.append(d[m-i,n]+costdict[motif]['delete cost'][s])
            d[m,n] = min(options)
    return d[lp,lc]
