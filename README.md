# Sqra.jl

This "package" grew from the work on and with the 'Square Root Approximation' (SQRA).
During that time it 'evolved' into a collection of different applications.
The main use case was the comparison of the *committors* of a *Lennard-Jones cluster* system computed via the *SQRA* on either *SparseBoxes* or
*Voronoi tesselations*.

It furthermore contains the code for the convergence analysis of the SQRA in the Paper

## Summary of contents:
- Lennard-Jones Clusters
- SQRA
- SparseBoxes
- Committors, classically via Q and via neural networks
- ISOKANN, clasically via power iterations and via residual loss
- error computation on voronoi/sparsebox discretizations

## Contents:

- permadict.jl - HD memoization implemented by a Dict forwarding read/writes to the hard disk
- eulermaruyama.jl - An (adaptive!) Euler Maruyama SDE solver with a maximal dy threshold 
- picking.jl - Implementation of the Picking algorithm (by Vedat Durmaz)
- voronoi_lp.jl - Linear Program to determine the adjacency of Voronoi cells (by Han Cheng Lie)
- sparseboxes.jl - this was sbmatrix.jl once, now switched to the Dict representation
- spdict.jl - Representation of SparseBoxes via Dict
- spdistancs.jl - Calculate $\int f-g dx$ where $f,g$ are defined on different SparseBox discretizations
- errors.jl - Calculate $\int f-g dx$ for $f,g$ Voronoi-Voronoi / Voronoi-SparseBox discretizations
- sqra_core.jl - The SQRA core algorithm
- committor.jl - Solver for the committor problems given a generator matrix Q
- models.jl - TripleWell and Lennard-Jones cluster models (with LJ Normal Form plot)
- experiment.jl / convergence.jl - comparison of SparseBoxes and Voronoi-Picking SQRA committor results 
- experiment2.jl - simpler rewrite of experiment.jl Experiments
- mcerror.jl - Adaptive Voronoi MC Volume computation (to a specified confidence) (not yet in VoronoiGraph.jl)
- mcfinvol.jl - Alternative to SQRA by using the Voronoi MC method to compute the boundary fluxes
- martin/*.jl - Code for the co

Further contents:
- archive/batch.jl - batch computation and comparison of the different discretization resolutions
- archive/boxtrie.jl - attempt of writing a *Trie* datastructure for the *SparseBoxes*
- archive/isokann.jl - *ISOKANN*, either classically via the *Fixed Point Iteration* or via *Residual Minimization*
- archive/lennardjones.jl - first attempt at implementing the Lennard Jones Clusters
- archive/metasgd.jl - SGD on the SGD learning rate by linear regression on past improvements
- archive/molly.jl - attempt to run Molecular Dynamics via Molly.jl
- archive/nestedgrad*.jl - Tests to single out AD problems with nested gradients
- archive/neurcomm.jl - A custom approach to compute committors via neural networks, inspired by (Khoo et al and Li et al) (actually this was quite interesting, worth looking into again)
- archive/sbmatrix.jl - the first implementation of sparseboxes
- archive/trie.jl - and another trie implementation

Notable efforts went into the code for the SparseBoxes (i.e. a structured "matrix" approach, then tries, finally the Dict) and their utility functions (cartesiancoords, merge, dists).
This was also the birthplace of the VoronoiGraph.jl package, which got outsourced, and a testbed for neural committors (neurcomm.jl) and ISOKANN.
