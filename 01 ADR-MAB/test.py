#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:05:43 2019

@author: tuyenta

Edit on Thursday, 5 August 2021
Farzad Azizi 
Email: farzad.azizi@aut.ac.ir 

Benyamin Teymuri
Email: benyamin.teymuri@aut.ac.ir 

"""
from lora.utils import print_params, sim

nrNodes = int(100)
nrIntNodes = int(100)

nrBS = int(1)
initial = "RANDOM"
radius = float(4500)

distribution = [0.1, 0.1, 0.3, 0.4, 0.05, 0.05]

avgSendTime = int(4*60*1000)

horTime = int(3e6)

packetLength = int(50)

sfSet = [7, 8, 9, 10, 11, 12]

#868100,868300,868500
freqSet = [868100]
    #minfreq = 867100
    #maxfreq = 868500
# 8, 11, 14  
powSet = [14]
captureEffect = True
interSFInterference = True
info_mode = 'NO'

# learning algorithm
algo = 'ADR-MAB'

# folder
exp_name = 'E1'
logdir = 'Result'


# print simulation parameters
print("\n=================================================")
print_params(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime, packetLength, 
            sfSet, freqSet, powSet, captureEffect, interSFInterference, info_mode, algo)
assert initial in ["UNIFORM", "RANDOM"], "Initial mode must be UNIFORM, or RANDOM."
assert info_mode in ["NO", "PARTIAL", "FULL"], "Initial mode must be NO, PARTIAL, or FULL."
assert algo in ["ADR-MAB"], "Learning algorithm ADR-MAB."

# running simulation
bsDict, nodeDict = sim(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, 
                                horTime, packetLength, sfSet, freqSet, powSet, 
                                captureEffect, interSFInterference, info_mode, algo, logdir, exp_name)
