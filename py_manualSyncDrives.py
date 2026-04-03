################################################################
# Jon Stingel
# 05022023
################################################################
'''
script to take my local files and sync them to a google drive
A helper function version of this is built into the 
helperOsimFunctions, this is for manually calling outside as
a standalone. 
'''

# # % imports
import os
import helperOsimFunctions
import pdb

destDir = 'G:\\Shared drives\\Exotendon\\predictiveSim\\testingMaterials\\PredictRunning_2d';
localDir = 'C:\\Users\\jonstingel\\code\\predictiveSim\\testingMaterials\\PredictRunning_2d';
# pdb.set_trace()
# copy all from local to G and overwrite anything
helperOsimFunctions.syncDrives(localDir, destDir)



# to wipe G and replace with local
# helperOsimFunctions.wipeSyncDrives(localDir, destDir)


