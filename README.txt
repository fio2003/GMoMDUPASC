1. Make sure you have next file in your folder: 
	clusters.dat - cluster assignments obtained with MDSCTK's spectral
	clustering algorithm
	sincos.dat - file containing information about points stored in
	phi-psi angles as described in paper.
	PCA_generative.py - python3 implementation of the algorithm described
	in paper(make sure this file has execution permission and first line
	points to you python3 path). This script assumes that you have all
	needed libraries installed - please go through the code and check all
	import statements.
2. Run PCA_generative.py - 10 files will be created, each will contain
points(in angle space) that were defined as one cluster.
3. Run PCA_generative.py -g 1000 to generate 1000 points.
