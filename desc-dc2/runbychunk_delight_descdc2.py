#!/usr/bin/env python
# coding: utf-8

# # Test Delight on DESC-DC2 simulation  in the context of  Vera C. Rubin Obs (LSST) 
# ## Run by chunk
# 
# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : January 22 2022
# 
# 
# 
# - run at NERSC with **desc-python** python kernel.
# 
# 
# Instruction to have a **desc-python** environnement:
# - https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Anaconda+Python+at+NERSC
# 
# 
# This environnement is a clone from the **desc-python** environnement where package required in requirements can be addded according the instructions here
# - https://github.com/LSSTDESC/desc-python/wiki/Add-Packages-to-the-desc-python-environment

# We will use the parameter file "tmps/parametersTestRail.cfg".
# This contains a description of the bands and data to be used.
# In this example we will generate mock data for the ugrizy LSST bands,
# fit each object with our GP using ugi bands only and see how it predicts the rz bands.
# This is an example for filling in/predicting missing bands in a fully bayesian way
# with a flexible SED model quickly via our photo-z GP.


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys,os
sys.path.append('../')
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP


import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')


print(sys.executable)
print(sys.version)
#print(sys.version_info)



# # Initialisation

workdir = "tmp"


# # Configuration parameters
# 
# - now parameters are generated in a dictionnary


list_of_files = os.listdir(workdir)
list_of_files.remove('data') 
list_of_files.remove('delight_data') 
if '.ipynb_checkpoints' in list_of_files:
    list_of_files.remove('.ipynb_checkpoints')
    
list_of_configfiles = sorted(list_of_files)


print(list_of_configfiles)




NCHUNKS = len(list_of_configfiles)


# # Filters

# - First, we must **fit the band filters with a gaussian mixture**. 
# This is done with this script:


from delight.interfaces.rail.processFilters import processFilters


configfilename = list_of_configfiles[0]
configfullfilename = os.path.join(workdir,configfilename) 
processFilters(configfullfilename)


# # SED

# - Second, we will process the library of SEDs and project them onto the filters,
# (for the mean fct of the GP) with the following script (which may take a few minutes depending on the settings you set):


from delight.interfaces.rail.processSEDs import processSEDs


configfilename = list_of_configfiles[0]
configfullfilename = os.path.join(workdir,configfilename) 
processSEDs(configfullfilename)


# # Train and apply
# Run the scripts below. There should be a little bit of feedback as it is going through the lines.
# For up to 1e4 objects it should only take a few minutes max, depending on the settings above.

# ## Template Fitting


from delight.interfaces.rail.templateFitting import templateFitting


for idx_file in range(1,NCHUNKS):
    theconfigfile = list_of_configfiles[idx_file]
    configfullfilename = os.path.join(workdir,theconfigfile) 
    templateFitting(configfullfilename)


# ## Gaussian Process

# ### Trainning


from delight.interfaces.rail.delightLearn import delightLearn



delightLearn(configfullfilename)


# ## Predictions


from delight.interfaces.rail.delightApply import delightApply


for idx_file in range(1,NCHUNKS):
    theconfigfile = list_of_configfiles[idx_file]
    configfullfilename = os.path.join(workdir,theconfigfile) 
    delightApply(configfullfilename)


