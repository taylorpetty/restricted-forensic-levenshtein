# restricted-forensic-levenshtein
Code for a generalization of the standard Levenshtein distance, allowing the adding or dropping of entire words at once as a separate edit type. This is motivated by forensic applications in STR analysis, where stutter occurs as a result of PCR.

The alphabet throughout this code is the genetic alphabet {A,C,G,T}. Each edit type is allowed to have a separate cost (e.g. A --> T can have a different cost than T --> A).

The name `lsdp` stands for Levenshtein Stutter Dictionary Pair. It takes a motif of length k and computes the standard weighted Levenshtein distance from every possible word of length 1 to 2k-1 to the motif, and from the motif to every possible word of length 1 to 2k-1. It returns a pair of dictionaries, one for forward stutter and one for backward, and the total cost of building every possible word with the given lengths if the motif stutters forward or backward exactly once.

The function `rfl` is the actual distance computation. It takes a data frame with rows and columns named '', 'A', 'C', 'G', 'T', where the i,j entry is the cost to edit from i to j (including insertions and deletions). It proceeds through the dynamic programming algorithm as normal, except at every step, it additionally looks back 1 to 2k-1 spaces for both insertion and deletion, using the precomputed `lsdp` output to see if the cost would be smaller by taking one of those words. Each motif can have its own stutter cost, and forward and backward can differ.

The `numbaRFL` version currently only accepts a single motif instead of a dictionary of multiple motifs, because of the way `numba` optimizes code, but it's about 28 times faster.
