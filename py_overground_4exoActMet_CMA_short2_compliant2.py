# # % -------------------------------------------------------------------------- %
# # % working on getting python versions of my own scripts going.
# # % Author: Jon Stingel
# # % 20230907
# # % -------------------------------------------------------------------------- %

# # % imports
import os

from matplotlib import table
# os.add_dll_directory("C:/OpenSim 4.4/bin")
# os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
# os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")
# os.add_dll_directory('C:/Users/jonstingel/opensim/opensim-core-4.5-2024-05-15-a1a2282/bin')
os.add_dll_directory('C:/Users/jonstingel/opensim-core-4.5.1-2024-08-23-cf3ef35/bin')

import opensim as osim
import re
import numpy as np
import pdb
import helperOsimFunctions
import bilevelTools
import time


# clear;
destDir = 'G:\\Shared drives\\Exotendon\\predictiveSim\\testingMaterials\\PredictRunning_2d';
localDir = 'C:\\Users\\jonstingel\\code\\predictiveSim\\testingMaterials\\PredictRunning_2d';
# % Load the Moco libraries

# % ---------------------------------------------------------------------------
# % Set up a coordinate tracking problem where the goal is to minimize the
# % difference between provided and simulated coordinate values and speeds (and
# % ground reaction forces), as well as to minimize an effort cost (squared
# % controls). The provided data represents half a gait cycle. Endpoint
# % constraints enforce periodicity of the coordinate values (except for
# % pelvis tx) and speeds, coordinate actuator controls, and muscle activations.


# % Set the weights for the terms in the objective function.

# % Note: We set GRFTrackingWeight to 0 so GRFs will not be tracked, but left 
# % this in the code in case other researchers wanted to track experimental GRF.
# % Setting GRFTrackingWeight to 1 will cause the total tracking error (states + GRF) to
# % have about the same magnitude as control effort in the final objective value.
# cmaX = [5.65627447e+00, 4.88501232e+01, 4.42395394e+00, 3.19657882e+00, 2.56978218e+01, 3.84954755e+01, 6.25325640e+01, 3.50247660e+01, 4.34307998e+01, 1.45299279e+01, 5.44550447e+00, 1.16829446e-07, 4.36006879e+00, 1.96495369e+01]
# actualx = [0, 0.6, 6e-1, 2.68e3, 2.68e0] # testing time
# x = [8.13639581e+00, 1.45953691e-04, 7.06217313e+00, 9.86183447e+01]
# # x = [8.13639581e+00, 1.45953691e-04, 7.06217313e+00, 5.86183447e+02]
# # x = [8.13639581e+00, 1.45953691e-04, 7.06217313e+00, 2.86183447e+02]
# # x = [8.13639581e+00, 1.45953691e-04, 7.06217313e+00, 1.86183447e+02]
# # x = [8.13639581e+00, 1.45953691e-04, 7.06217313e+00, 3.86183447e+02]

# x = [8.13639581e+00, 1.45953691e-04, 7.06217313e-2, 9.86183447e+01]
# x = [8.13639581e+00, 1.45953691e-04, 7.06217313e-4, 9.86183447e+01]
# x = [8.13639581e+00, 1.45953691e-04, 7.06217313e-2, 9.86183447e+01]
# x = [1.613639581e+01, 2.45953691e-04, 7.06217313e-2, 9.86183447e+01]
# x = [2.613639581e+01, 8.45953691e-04, 7.06217313e-2, 9.86183447e+01]
# x = [2.613639581e+01, 2.45953691e-04, 1.06217313e-1, 9.86183447e+01]
# x = [2.613639581e+01, 2.45953691e-04, 1.06217313e-3, 9.86183447e+01] # GRF way down to get dev.... 

# # g = 0.1

# controlEffortWeight = 0; 
# effortExponent = 2 
# stateTrackingWeight = 5e-6 # 10**(x[0]) # * g #* 5e-5
# GRFTrackingWeight   = x[2] #* 1e-3 # 10**(x[2]) # * g #x[1] * g 

# activationWeight = x[0] # 10**(0.9) # (x[1]) # * g #* 0.01
# activationWeightEach = 1e0

# metabolicsWeight = x[1] # 10**(x[1]) # * g #* 0.05
# metabolicsExponent = 2


# # ## testing something here
# # # GRF magnitudes
# # forceweight = 3e-6
# # heelForceWeight = forceweight
# # heelForceExponent = 2;
# # toeForceWeight = forceweight
# # toeForceExponent = 2;

# # head tracking
# # headTrackWeight = x[3]

# # head accelerations
# # headWeight = 1e-3
# # vitalWeight = 1e-3

# # heel acceleration
# # heelAccWeight = 3e-4
# # heelAccExponent = 2;
########################################################
### here is the new CMA setup

## here is the new CMA pass 
x = [1.03638490e-06, 8.09317398e+00, 1.00004734e+02, 9.99580997e+01]
# state track, grf track, head track, ty weight
# act = 8.0
# met = 2.4e-4

# g = 0.1

controlEffortWeight = 0
effortExponent = 2
stateTrackingWeight = x[0] * 2 * 0.5 # normally stops at 0.5
GRFTrackingWeight   = x[1] * 5e-2 * 0.5 * 73 * 2 * 0.5 # stops at 0.5

activationWeight = 8.0
activationWeightEach = 1e0

metabolicsWeight = 2.4e-4 * 15 * 0.5 * 0.5
metabolicsExponent = 2

# head tracking 
headTrackWeight = x[2] * 0

# pelvis ty tracking
tyweight = x[3] * 2 * 100

# joint moment tracking
# momenttrackweight = 1e-3

# ## testing something here
# # GRF magnitudes
# forceweight = 3e-5

# heelForceWeight = forceweight
# heelForceExponent = 2;
# toeForceWeight = forceweight
# toeForceExponent = 2;

# head accelerations
# headWeight = 1e-3
# vitalWeight = 1e-3

# heel acceleration
# heelAccWeight = 3e-4
# heelAccExponent = 2;

# guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
# # guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_met/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
# guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_newmet/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
guessfile = './goodresults/4ms_mk13_poly5/015_4ActMet_684_tight/4ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
# guessfile = './results/4.0/684/4ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'

implicitWeight = 6e-2 # 6e3

convergeTolerance   = 1e-2
constraintTolerance = 1e-3
fractionExtraBoundSize = 0.45

stepsize = .015
maxiterations = 1000

initialTime = 0.0

finalTime = 0.564 / 2 #  0.604, 0.644, 0.684, 0.724
startendtime = 0.25
endendtime = 0.74/2

guess = True
wantguess = False

trackIK = True
# trackIK = False

trackedfile = './4CMA_exo_short2_compliant2_2D3D_muscle_GaitTracking_tracked_states.sto'
resultspath = './results/4.0/' + str(finalTime*2)[2:] + '/'
if not os.path.exists(resultspath):
    os.makedirs(resultspath)

# % Define the optimal control problem
# % ==================================
track = osim.MocoTrack();
track.setName('4CMA_exo_short2_compliant2_2D3D_muscle_GaitTracking');


# % Set the OpenSim Model and give it a name
# TreadmillModel = 'strong_mk13_rv1_dgf.osim'
# TreadmillModel = 'strong_mk14_rv1_dgf.osim'
# TreadmillModel = 'strong_mk14_rv1_dgf_kneedampremoved_exo_short2_compliant2.osim'
TreadmillModel = 'strong_mk15_rv1_dgf_kneedampremoved_exo_short2_compliant2.osim'
modelProcessor = osim.ModelProcessor(TreadmillModel)
functionBasedPathsFile = './pathResults/HOBLingApoorva-scaled_FunctionBasedPathSet.xml'
modelProcessor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(functionBasedPathsFile))
model = modelProcessor.process();

# % Reference data for tracking problem
if trackIK:
    # tableProcessor = osim.TableProcessor('./expData/2D2Darms/27_IK_nat_mk12_rv1_1.mot');
    tableProcessor = osim.TableProcessor('./expData/Ham19_4ms/4_IK_mk13_rv1_nat_trim.mot')
    tableProcessor.append(osim.TabOpLowPassFilter(20))
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
else:
    # new method for stretch/shrinking input values for tracking based on the duration.
    # tableProcessor = osim.TableProcessor('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
    # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
    # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_tight/3ActMet_kinematicsValues_solution_6835.sto')
    basekin = osim.TimeSeriesTable('./goodresults/4ms_mk13_poly5/015_4ActMet_684_tight/4ActMet_kinematicsValues_solution_684.sto')
    basekintime = basekin.getIndependentColumn();
    # get a new time vector the length of the original one, but with new values for our desired duration. 
    newtime = np.linspace(0,finalTime*2,len(basekintime)); # length depends on the input here


    # stretching the cycle
    if newtime[-1] > basekintime[-1]:
        newidx = len(newtime) - 1
        for i in range(len(basekintime)):
            basekin.setIndependentValueAtIndex(newidx, newtime[-(i+1)])
            newidx -= 1 

    if newtime[-1] < basekintime[-1]:
        for i in range(len(basekintime)):
            basekin.setIndependentValueAtIndex(i, newtime[i])

    # osim.STOFileAdapter.write(basekin, 'basekin.sto')
    tableProcessor = osim.TableProcessor(basekin)
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());


#'''
# if doing metabolics in the problem tweak the model
# if 'metabolicsWeight' in locals():
## for Bhargava
# adding metabolics effort to the cost
modelProcessor = osim.ModelProcessor(model);
premetmodel = modelProcessor.process()
premetmodel.initSystem()
muscles = premetmodel.getMuscles()
numMuscles = muscles.getSize()
metabolics = osim.Bhargava2004SmoothedMuscleMetabolics()
premetmodel.addComponent(metabolics)  
metabolics.setName('metabolic_cost')
metabolics.set_use_smoothing(True)
metabolics.set_enforce_minimum_heat_rate_per_muscle(True)
metabolics.set_forbid_negative_total_power(True)
metabolics.set_include_negative_mechanical_work(True)
metabolics.set_basal_coefficient(0)
metabolics.set_basal_exponent(1.0)
##
# other thing is to set fast and slow twitches for recruitment
##
# loop and add all the muscles to the model
for m in range(numMuscles):
    muscle = muscles.get(m)
    muscleName = muscle.getName()
    musclePath = muscle.getAbsolutePathString()
    ratio = helperOsimFunctions.getMuscleFiberRatios(muscleName, 'short2')
    metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 150000)
    # also add a metabolics component for each of the muscles individually
    musclemet = osim.Bhargava2004SmoothedMuscleMetabolics()
    musclemet.setName(muscleName + '_metabolic_cost')
    musclemet.set_use_smoothing(True)
    musclemet.set_enforce_minimum_heat_rate_per_muscle(True)
    musclemet.set_forbid_negative_total_power(True)
    musclemet.set_include_negative_mechanical_work(True)
    musclemet.set_basal_coefficient(0)
    musclemet.set_basal_exponent(1.0)
    musclemet.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 150000)
    premetmodel.addComponent(musclemet)
# premetmodel.addComponent(metabolics)
premetmodel.finalizeConnections()
premetmodel.printToXML('strong_mk14_exo_short2_compliant2_post.osim')
modelProcessor = osim.ModelProcessor(premetmodel)


# modelProcessor = osim.ModelProcessor(model)
# make sure our tendons are compliant
modelProcessor.append(osim.ModOpTendonComplianceDynamicsModeDGF('implicit'));
# test widening the active force length curve 
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

# modelProcessor = osim.ModelProcessor(TreadmillModel);
track.setModel(modelProcessor);
track.setStatesReference(tableProcessor);
track.set_states_global_tracking_weight(stateTrackingWeight);
track.set_allow_unused_references(True);
track.set_track_reference_position_derivatives(True);
# track.set_apply_tracked_states_to_guess(True);
track.set_initial_time(0.0);
# track.set_mesh_interval(stepsize)
track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
study = track.initialize();
problem = study.updProblem();

# % Goals
# % =====

# % # % Set different tracking weights for states (weights for states not 
# % # % explicitly set here have a default value of 1.0). The values below
# % # % were obtained by trial and error.
stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
# stateTrackingGoal.setDivideByDisplacement(True)

'''
best individual
[16.4642132  26.17749431  2.40395633  1.40346929  5.81808907  9.45995736
9.9811225  19.10767123  0.17034188  6.38006386  7.76837126  4.61191839
8.47651309  2.5277297 ]
'''
cmaX = [5.65627447e+00, 4.88501232e+01, 4.42395394e+00, 3.19657882e+00, 2.56978218e+01, 3.84954755e+01, 6.25325640e+01, 3.50247660e+01, 4.34307998e+01, 1.45299279e+01, 5.44550447e+00, 1.16829446e-07, 4.36006879e+00, 1.96495369e+01]

# ## new way using the max averaged differences between conditions for the tracking weights. 
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 500/((2.2*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/speed', 500/((2.2*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 50/((2.6*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed', 50/((2.6*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/value', 50/((2.3*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/speed', 50/((2.3*np.pi/180)**2));

# # bounds of 1e7 to 1e10
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0);
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1e2);

# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
# stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);

# stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 100/((4.5*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 100/((4.5*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 100/((4.5*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 100/((4.5*np.pi/180)**2))

# stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 5/((18.8*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 5/((18.8*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 5/((18.8*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 5/((18.8*np.pi/180)**2))

# stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 1e2/((12.9*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 1e2/((12.9*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 1e2/((12.9*np.pi/180)**2))
# stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 1e2/((12.9*np.pi/180)**2))

# stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
# stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
# stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
# stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);

# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 1000/((3.0*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 1000/((3.0*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 100/((4.9*np.pi/180)**2)); # 0.5
# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 100/((4.9*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/value', 50/((3.6*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/speed', 50/((3.6*np.pi/180)**2));

# stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 50/((2.1*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/speed', 50/((2.1*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 50/((2.1*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/speed', 50/((2.1*np.pi/180)**2));

# stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 10/((1.4*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/speed', 10/((1.4*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 10/((1.4*np.pi/180)**2));
# stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/speed', 10/((1.4*np.pi/180)**2));

########################################################
# CMA new setup
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 1000/((2.2*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/speed',0.1* 1000/((2.2*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 50/((2.6*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed',0.1* 50/((2.6*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/value', 50/((2.3*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/speed',0.1* 50/((2.3*np.pi/180)**2));

# bounds of 1e7 to 1e10
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 4*tyweight);
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 4*tyweight);

stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed',0.001* 0.0);

stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 1000* 1000/((4.5*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 0.001*1000/((4.5*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 1000* 1000/((4.5*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 0.001*1000/((4.5*np.pi/180)**2))

stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 3* 5e3/((18.8*np.pi/180)**2)) # 5e4 too small, 7e4 too big
stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed',0.1* 3* 5e3/((18.8*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 1e3/((18.8*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed',0.001* 1e3/((18.8*np.pi/180)**2))

stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 8e6/((12.9*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed',0.001* 0 * 8e6/((12.9*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 4e4/((12.9*np.pi/180)**2))
stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed',0.001* 0 * 4e4/((12.9*np.pi/180)**2))

stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed',0.001* 0.0);
stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed',0.001* 0.0);

stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 3* 100/((2.1*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/speed', 3* 100/((2.1*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 3* 100/((2.1*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/speed', 3* 100/((2.1*np.pi/180)**2));

stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 100/((1.4*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/speed', 100/((1.4*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 100/((1.4*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/speed', 100/((1.4*np.pi/180)**2));

stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 10000/((3.0*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 10000/((3.0*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10000/((4.9*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10000/((4.9*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/value', 5000/((3.6*np.pi/180)**2));
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/speed', 5000/((3.6*np.pi/180)**2));



# activation goal setup
activationGoal = osim.MocoSumSquaredStateGoal('activationGoal')
if 'activationWeight' in locals():
    activationGoal.setWeight(activationWeight)
    activationGoal.setDivideByDisplacement(True)
    problem.addGoal(activationGoal)

# initial Activation endpoint constraint goal
initActivationGoal = osim.MocoInitialActivationGoal('initialActivationGoal')
problem.addGoal(initActivationGoal)


# % Symmetry (to permit simulating only one step)
symmetryGoal = osim.MocoPeriodicityGoal('symmetryGoal');
problem.addGoal(symmetryGoal);
model = modelProcessor.process();
model.initSystem();


# % Symmetric coordinate values (except for pelvis_tx) and speeds
for i in range(model.getNumStateVariables()):
    currentStateName = str(model.getStateVariableNames().getitem(i));
    if str.startswith(currentStateName , '/jointset') and 'beta' not in currentStateName:
        # print('\niiii')
        # print(currentStateName)

        # take joints out of the state squared goal, only want activations
        activationGoal.setWeightForState(currentStateName, 0)

        # right and left limb coordinates
        if currentStateName.endswith('_r/value') or currentStateName.endswith('_r/speed'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_r', '_l', currentStateName));
            symmetryGoal.addStatePair(pair);
            # print('1 - rights and lefts - pair')
            # print(currentStateName)
            # print(re.sub('_r', '_l', currentStateName))
        if currentStateName.endswith('_l/value') or currentStateName.endswith('_l/speed'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_l', '_r', currentStateName)); 
            symmetryGoal.addStatePair(pair);
            # print('2 - lefts and rights - pair')
            # print(currentStateName)
            # print(re.sub('_l', '_r', currentStateName))
        # pelvis tilt
        if currentStateName.endswith('_tilt/value') or currentStateName.endswith('_tilt/speed'):
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
            # print('3 - pelvis tilt - pair value and speed')
            # print(currentStateName)
        # pelvis list
        if currentStateName.endswith('_list/value') or currentStateName.endswith('_list/speed'):
            if currentStateName.endswith('/value'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('4 - pelvis list - negated pair value')
                # print(currentStateName)
            if currentStateName.endswith('/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('4 - pelvis list - speed pair')
                # print(currentStateName)
        # pelvis rotation
        if currentStateName.endswith('pelvis_rotation/value') or currentStateName.endswith('pelvis_rotation/speed'):
            if currentStateName.endswith('/value'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('5 - pelvis rotation - negated pair value')
                # print(currentStateName)
            if currentStateName.endswith('/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('5 - pelvis rotation - pair speed')
                # print(currentStateName)
        # pelvis ty symmetry
        if currentStateName.endswith('_ty/value') or currentStateName.endswith('_ty/speed'):
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
            # print('6 - pelvis ty - pair value and speed')
            # print(currentStateName)
        # pelvis tx
        if currentStateName.endswith('_tx/speed'): # overground so not value
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
            # print('7 - pelvis tx - pair speed')
            # print(currentStateName)
        # # pelvis Tz
        # if currentStateName.endswith('_tz/value') or currentStateName.endswith('_tz/speed'):
        #     if currentStateName.endswith('value'):
        #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
        #         print('8 - pelvis tz - pair value')
        #         print(currentStateName)
        #     if currentStateName.endswith('speed'):
        #         symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
        #         print('8 - pelvis tz - negated pair speed')
        #         print(currentStateName) 
        # lumbar extension 
        if currentStateName.endswith('_extension/value') or currentStateName.endswith('_extension/speed'):
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
            # print('9 - lumbar extension - pair value and speed')
            # print(currentStateName)
        # lumbar bending 
        if currentStateName.endswith('_bending/value') or currentStateName.endswith('_bending/speed'):
            if currentStateName.endswith('/value'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('10 - lumbar bending - negated value pair')
                # print(currentStateName)
            if currentStateName.endswith('/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('10 - lumbar bending - speed pair')
                # print(currentStateName)
        # lumbar rotation
        if currentStateName.endswith('lumbar_rotation/value') or currentStateName.endswith('lumbar_rotation/speed'):
            if currentStateName.endswith('/value'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('11 - lumbar rotation - negated pair value')
                # print(currentStateName)
            if currentStateName.endswith('_rotation/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                # print('11 - lumbar rotation - pair speed')
                # print(currentStateName)

    if 'beta' in currentStateName:
        # print('\niiii')
        # print(currentStateName)
        activationGoal.setWeightForState(currentStateName, 0)

print('\n\naaaaaaaaaaaaaaaaaaaahahaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
# % Symmetric muscle activations
for i in range(model.getNumStateVariables()):
    currentStateName = str(model.getStateVariableNames().getitem(i));
    
    if str.endswith(currentStateName, '/normalized_tendon_force'):
        # print('\nhhhhhh')
        # print(currentStateName)
        activationGoal.setWeightForState(currentStateName, 0)
    # TODO tendon forces symmetry
        if str.endswith(currentStateName, '_r/normalized_tendon_force'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_r','_l', currentStateName));
            symmetryGoal.addStatePair(pair);
            # print('norm tendon pairs')
            # print(re.sub('_r', '_l', currentStateName))
        if str.endswith(currentStateName, '_l/normalized_tendon_force'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_l','_r', currentStateName));
            symmetryGoal.addStatePair(pair);
            # print('norm tendon pairs')
            # print(re.sub('_l', '_r', currentStateName))

    if str.endswith(currentStateName,'/activation'):
        # print('\naaaa')
        # print(currentStateName)
        # activations squared for this actuator
        activationGoal.setWeightForState(currentStateName, activationWeightEach)

        if currentStateName.endswith('_r/activation'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_r','_l', currentStateName));
            symmetryGoal.addStatePair(pair);
            # print('a1 - rights and lefts - pair')
            # # print(currentStateName)
            # print(re.sub('_r', '_l', currentStateName))
        if currentStateName.endswith('_l/activation'):
            pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_l', '_r', currentStateName));
            symmetryGoal.addStatePair(pair);
            # print('a2 - lefts and rights - pair')
            # # print(currentStateName)
            # print(re.sub('_l', '_r', currentStateName))
        # bending , rotation , tz, list - these are gonna be different 
        # if 'Bend' in currentStateName or 'Rot' in currentStateName:
        #     symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
        #     print('a3 - lumbar bending, lumbar rotation actuators - negated pair')
        #     print(currentStateName)
        # if 'Ext' in currentStateName:
        #     symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
        #     print('a3 - lumbar extension actuator -  pair')
        #     print(currentStateName)

print('\nccccccccccccccccccccccccccccccccccccccccccccccccccc')
# now get the controls and do symmetry for them as well
# controlTab = model.getControlsTable();
# controlnames = controlTab.getColumnLabels();
# for i in range(len(controlnames)):
#     currentcontrol = controlnames[i]
#     print('/controllerset/' + currentcontrol)
#     if str.endswith(currentcontrol, '_r'):
#         pair = Moco
forceSet = model.getForceSet();
for i in range(forceSet.getSize()):
    currentforce = forceSet.get(i).getAbsolutePathString();
    # print('\nccccc')
    # print(currentforce)
    if str.endswith(currentforce, '_r') and 'Passive' not in currentforce and 'mtp' not in currentforce:
        pair = osim.MocoPeriodicityGoalPair(currentforce, re.sub('_r', '_l', currentforce));
        symmetryGoal.addControlPair(pair);
        # print(re.sub('_r', '_l', currentforce))
    if str.endswith(currentforce, '_l') and 'Passive' not in currentforce and 'mtp' not in currentforce:
        pair = osim.MocoPeriodicityGoalPair(currentforce, re.sub('_l', '_r', currentforce));
        symmetryGoal.addControlPair(pair);
        # print(re.sub('_l', '_r', currentforce))



# % Get a reference to the MocoControlGoal that is added to every MocoTrack
# % problem by default and change the weight
effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
effort.setWeight(controlEffortWeight);
effort.setDivideByDisplacement(True);
effort.setExponent(effortExponent)
if controlEffortWeight == 0:
    effort.setEnabled(False)
else:
    effort.setEnabled(True)


speedGoal = osim.MocoAverageSpeedGoal('speed');
speedGoal.set_desired_average_speed(4.0);
speedGoal.setMode('endpoint_constraint');
problem.addGoal(speedGoal);


# tendon velocity bounding
bounds = osim.MocoBounds(-0.8, 0.8)
boundsVec = osim.StdVectorMocoBounds()
boundsVec.append(bounds)

tenVelWeight = 1

tenGoal_bfsh_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_r', tenVelWeight)
tenGoal_bfsh_r.setMode('endpoint_constraint')
tenGoal_bfsh_r.setOutputPath('/forceset/bfsh_r|tendon_velocity')
tenGoal_bfsh_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_bfsh_r)
tenGoal_gasmed_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_r', tenVelWeight)
tenGoal_gasmed_r.setMode('endpoint_constraint')
tenGoal_gasmed_r.setOutputPath('/forceset/gasmed_r|tendon_velocity')
tenGoal_gasmed_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_gasmed_r)
tenGoal_soleus_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_r', tenVelWeight)
tenGoal_soleus_r.setMode('endpoint_constraint')
tenGoal_soleus_r.setOutputPath('/forceset/soleus_r|tendon_velocity')
tenGoal_soleus_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_soleus_r)
tenGoal_tibant_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_r', tenVelWeight)
tenGoal_tibant_r.setMode('endpoint_constraint')
tenGoal_tibant_r.setOutputPath('/forceset/tibant_r|tendon_velocity')
tenGoal_tibant_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_tibant_r)
tenGoal_vasint_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_r', tenVelWeight)
tenGoal_vasint_r.setMode('endpoint_constraint')
tenGoal_vasint_r.setOutputPath('/forceset/vasint_r|tendon_velocity')
tenGoal_vasint_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_vasint_r)
tenGoal_recfem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_r', tenVelWeight)
tenGoal_recfem_r.setMode('endpoint_constraint')
tenGoal_recfem_r.setOutputPath('/forceset/recfem_r|tendon_velocity')
tenGoal_recfem_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_recfem_r)
tenGoal_psoas_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_r', tenVelWeight)
tenGoal_psoas_r.setMode('endpoint_constraint')
tenGoal_psoas_r.setOutputPath('/forceset/psoas_r|tendon_velocity')
tenGoal_psoas_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_psoas_r)
tenGoal_semimem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_r', tenVelWeight)
tenGoal_semimem_r.setMode('endpoint_constraint')
tenGoal_semimem_r.setOutputPath('/forceset/semimem_r|tendon_velocity')
tenGoal_semimem_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_semimem_r)
tenGoal_glmax2_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_r', tenVelWeight)
tenGoal_glmax2_r.setMode('endpoint_constraint')
tenGoal_glmax2_r.setOutputPath('/forceset/glmax2_r|tendon_velocity')
tenGoal_glmax2_r.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_glmax2_r)
tenGoal_bfsh_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_l', tenVelWeight)
tenGoal_bfsh_l.setMode('endpoint_constraint')
tenGoal_bfsh_l.setOutputPath('/forceset/bfsh_l|tendon_velocity')
tenGoal_bfsh_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_bfsh_l)
tenGoal_gasmed_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_l', tenVelWeight)
tenGoal_gasmed_l.setMode('endpoint_constraint')
tenGoal_gasmed_l.setOutputPath('/forceset/gasmed_l|tendon_velocity')
tenGoal_gasmed_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_gasmed_l)
tenGoal_soleus_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_l', tenVelWeight)
tenGoal_soleus_l.setMode('endpoint_constraint')
tenGoal_soleus_l.setOutputPath('/forceset/soleus_l|tendon_velocity')
tenGoal_soleus_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_soleus_l)
tenGoal_tibant_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_l', tenVelWeight)
tenGoal_tibant_l.setMode('endpoint_constraint')
tenGoal_tibant_l.setOutputPath('/forceset/tibant_l|tendon_velocity')
tenGoal_tibant_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_tibant_l)
tenGoal_vasint_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_l', tenVelWeight)
tenGoal_vasint_l.setMode('endpoint_constraint')
tenGoal_vasint_l.setOutputPath('/forceset/vasint_l|tendon_velocity')
tenGoal_vasint_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_vasint_l)
tenGoal_recfem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_l', tenVelWeight)
tenGoal_recfem_l.setMode('endpoint_constraint')
tenGoal_recfem_l.setOutputPath('/forceset/recfem_l|tendon_velocity')
tenGoal_recfem_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_recfem_l)
tenGoal_psoas_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_l', tenVelWeight)
tenGoal_psoas_l.setMode('endpoint_constraint')
tenGoal_psoas_l.setOutputPath('/forceset/psoas_l|tendon_velocity')
tenGoal_psoas_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_psoas_l)
tenGoal_semimem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_l', tenVelWeight)
tenGoal_semimem_l.setMode('endpoint_constraint')
tenGoal_semimem_l.setOutputPath('/forceset/semimem_l|tendon_velocity')
tenGoal_semimem_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_semimem_l)
tenGoal_glmax2_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_l', tenVelWeight)
tenGoal_glmax2_l.setMode('endpoint_constraint')
tenGoal_glmax2_l.setOutputPath('/forceset/glmax2_l|tendon_velocity')
tenGoal_glmax2_l.setEndpointConstraintBounds(boundsVec)
problem.addGoal(tenGoal_glmax2_l)


# if metabolics in the problem
if 'metabolicsWeight' in locals():
    metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight / 22)
    metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
    metabolicsGoal.setDivideByDisplacement(True)
    metabolicsGoal.setDivideByMass(True)
    metabolicsGoal.setExponent(metabolicsExponent)
    problem.addGoal(metabolicsGoal)

    # # test additional term on the hip flexors to see if we shorten the stride up better. 
    # metabolicspsoas = osim.MocoOutputGoal('metabolicspsoas', metabolicsWeight*10)
    # metabolicspsoas.setOutputPath('/psoas_r_metabolic_cost|total_metabolic_rate')
    # metabolicspsoas.setDivideByDisplacement(True)
    # metabolicspsoas.setDivideByMass(True)
    # metabolicspsoas.setExponent(metabolicsExponent)
    # problem.addGoal(metabolicspsoas)

    # # test additional term on the hip flexors to see if we shorten the stride up better. 
    # metabolicspsoas_l = osim.MocoOutputGoal('metabolicspsoas_l', metabolicsWeight*10)
    # metabolicspsoas_l.setOutputPath('/psoas_l_metabolic_cost|total_metabolic_rate')
    # metabolicspsoas_l.setDivideByDisplacement(True)
    # metabolicspsoas_l.setDivideByMass(True)
    # metabolicspsoas_l.setExponent(metabolicsExponent)
    # problem.addGoal(metabolicspsoas_l)

    # # test additional term on the hamstrings to see if we shorten the stride up better. 
    # metabolicssemimem_r = osim.MocoOutputGoal('metabolicssemimem_r', metabolicsWeight*10)
    # metabolicssemimem_r.setOutputPath('/semimem_r_metabolic_cost|total_metabolic_rate')
    # metabolicssemimem_r.setDivideByDisplacement(True)
    # metabolicssemimem_r.setDivideByMass(True)
    # metabolicssemimem_r.setExponent(metabolicsExponent)
    # problem.addGoal(metabolicssemimem_r)

    # # test additional term on the hamstrings to see if we shorten the stride up better. 
    # metabolicssemimem_l = osim.MocoOutputGoal('metabolicssemimem_l', metabolicsWeight*10)
    # metabolicssemimem_l.setOutputPath('/semimem_l_metabolic_cost|total_metabolic_rate')
    # metabolicssemimem_l.setDivideByDisplacement(True)
    # metabolicssemimem_l.setDivideByMass(True)
    # metabolicssemimem_l.setExponent(metabolicsExponent)
    # problem.addGoal(metabolicssemimem_l)

if 'headTrackWeight' in locals():
    # % track the head positions
    # get the states and stretch/shrink them
    # headkinTraj = osim.MocoTrajectory('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto')
    headkinTraj = osim.MocoTrajectory('./goodresults/4ms_mk13_poly5/015_4ActMet_684_tight/4ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto')
    headkin = headkinTraj.exportToStatesTable()
    headkintime = headkin.getIndependentColumn();
    # get a new time vector the length of the original one, but with new values for our desired duration. 
    newtime = np.linspace(0,finalTime*2,len(headkintime)); # length depends on the input here
    # stretching the cycle
    if newtime[-1] > headkintime[-1]:
        newidx = len(newtime) - 1
        for i in range(len(headkintime)):
            headkin.setIndependentValueAtIndex(newidx, newtime[-(i+1)])
            newidx -= 1 
    if newtime[-1] < headkintime[-1]:
        for i in range(len(headkintime)):
            headkin.setIndependentValueAtIndex(i, newtime[i])
    # osim.STOFileAdapter.write(basekin, 'basekin.sto')
    headtableProcessor = osim.TableProcessor(headkin)
    headtableProcessor.append(osim.TabOpUseAbsoluteStateNames());
    headPosGoal = osim.MocoTranslationTrackingGoal('headpos');
    headPosGoal.setWeight(headTrackWeight); # type: ignore
    headPosGoal.setStatesReference(headtableProcessor)
    headPosGoal.setFramePaths(['/bodyset/torso/head'])
    problem.addGoal(headPosGoal);

if 'headWeight' in locals():
    # % head accelerations
    headGoal = osim.MocoOutputGoal('headacc');
    headGoal.setOutputPath('bodyset/torso/head|acceleration');
    headGoal.setExponent(2);
    headGoal.setWeight(headWeight); # type: ignore
    problem.addGoal(headGoal);


if 'vitalWeight' in locals():
    # % head accelerations
    vitalGoal = osim.MocoOutputGoal('vitalacc');
    vitalGoal.setOutputPath('bodyset/torso/vitals|acceleration');
    vitalGoal.setExponent(2);
    vitalGoal.setWeight(vitalWeight); # type: ignore
    problem.addGoal(vitalGoal);

if 'momenttrackweight' in locals():
    # Add a joint moment tracking goal to the problem.
    jointMomentTracking = osim.MocoGeneralizedForceTrackingGoal('joint_moment_tracking', momenttrackweight) # type: ignore
    # low-pass filter the data at 10 Hz. The reference data should use the 
    # same column label format as the output of the Inverse Dynamics Tool.
    jointMomentRef = osim.TableProcessor('./IDtesting/inverse_dynamics15_short_rebase.sto')
    # jointMomentRef.append(osim.TabOpLowPassFilter(10))
    jointMomentTracking.setReference(jointMomentRef)
    # Set the force paths that will be applied to the model to compute the
    # generalized forces. Usually these are the external loads and actuators 
    # (e.g., muscles) should be excluded, but any model force can be included 
    # or excluded. Gravitational force is applied by default.
    # Regular expression are supported when setting the force paths.
    forcePaths = osim.StdVectorString()
    forcePaths.append('.*externalloads.*')
    forcePaths.append('.*contact.*')
    jointMomentTracking.setForcePaths(forcePaths)
    # Allow unused columns in the reference data.
    jointMomentTracking.setAllowUnusedReferences(True)
    # Normalize the tracking error for each generalized for by the maximum 
    # absolute value in the reference data for that generalized force.
    jointMomentTracking.setNormalizeTrackingError(True)
    # Ignore coordinates that are locked, prescribed, or coupled to other
    # coordinates via CoordinateCouplerConstraints (true by default).
    jointMomentTracking.setIgnoreConstrainedCoordinates(True)
    # Do not track generalized forces associated with pelvis residuals.
    jointMomentTracking.setWeightForGeneralizedForcePattern('.*pelvis.*', 0)
    jointMomentTracking.setWeightForGeneralizedForcePattern('.*mtp.*', 0)
    # Encourage better tracking of the ankle joint moments.
    # jointMomentTracking.setWeightForGeneralizedForce('ankle_angle_r_moment', 100)
    # jointMomentTracking.setWeightForGeneralizedForce('ankle_angle_l_moment', 100)
    # Add the joint moment tracking goal to the problem.
    problem.addGoal(jointMomentTracking)

if 'heelForceWeight' in locals():
    # heel goal right
    heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
    heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
    heelForceGoal_r.setExponent(heelForceExponent); # type: ignore
    heelForceGoal_r.setWeight(heelForceWeight); # type: ignore
    problem.addGoal(heelForceGoal_r);
    # toe goal right
    toeForceGoal_r = osim.MocoOutputGoal('toeforce_r');
    toeForceGoal_r.setOutputPath('contactLateralMidfoot_r|sphere_force');
    toeForceGoal_r.setExponent(toeForceExponent); # type: ignore
    toeForceGoal_r.setWeight(toeForceWeight); # type: ignore
    problem.addGoal(toeForceGoal_r);

    toeForceGoal2_r = osim.MocoOutputGoal('toeforce2_r');
    toeForceGoal2_r.setOutputPath('contactMedialToe_r|sphere_force');
    toeForceGoal2_r.setExponent(toeForceExponent); # type: ignore
    toeForceGoal2_r.setWeight(toeForceWeight); # type: ignore
    problem.addGoal(toeForceGoal2_r);

    toeForceGoal3_r = osim.MocoOutputGoal('toeforce3_r');
    toeForceGoal3_r.setOutputPath('contactMedialMidfoot_r|sphere_force');
    toeForceGoal3_r.setExponent(toeForceExponent); # type: ignore
    toeForceGoal3_r.setWeight(toeForceWeight); # type: ignore
    problem.addGoal(toeForceGoal3_r);

if 'expressionForceWeight' in locals():
    # minimize the expression based coordinate forces on the knee.
    expressionForceGoal_kneer = osim.MocoOutputGoal('expressionForceGoal_kneer');
    expressionForceGoal_kneer.setOutputPath('/forceset/PassiveKneeDamping_r|force_magnitude');
    expressionForceGoal_kneer.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_kneer.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_kneer.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_kneer)
    expressionForceGoal_kneel = osim.MocoOutputGoal('expressionForceGoal_kneel');
    expressionForceGoal_kneel.setOutputPath('/forceset/PassiveKneeDamping_l|force_magnitude');
    expressionForceGoal_kneel.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_kneel.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_kneel.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_kneel)

    # minimize the expression based coordinate forces on the hips.
    expressionForceGoal_hipr = osim.MocoOutputGoal('expressionForceGoal_hipr');
    expressionForceGoal_hipr.setOutputPath('/forceset/PassiveHipDamping_l|force_magnitude');
    expressionForceGoal_hipr.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_hipr.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_hipr.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_hipr)
    expressionForceGoal_hipl = osim.MocoOutputGoal('expressionForceGoal_hipl');
    expressionForceGoal_hipl.setOutputPath('/forceset/PassiveHipDamping_l|force_magnitude');
    expressionForceGoal_hipl.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_hipl.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_hipl.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_hipl)

    # minimize the expression based coordinate forces on the ankles. 
    expressionForceGoal_ankler = osim.MocoOutputGoal('expressionForceGoal_ankler');
    expressionForceGoal_ankler.setOutputPath('/forceset/PassiveAnkleDamping_r|force_magnitude');
    expressionForceGoal_ankler.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_ankler.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_ankler.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_ankler)
    expressionForceGoal_anklel = osim.MocoOutputGoal('expressionForceGoal_anklel');
    expressionForceGoal_anklel.setOutputPath('/forceset/PassiveAnkleDamping_l|force_magnitude');
    expressionForceGoal_anklel.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_anklel.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_anklel.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_anklel)

    # minimize the expression based coordinate forces on the toes. 
    expressionForceGoal_toer = osim.MocoOutputGoal('expressionForceGoal_toer');
    expressionForceGoal_toer.setOutputPath('/forceset/PassiveToeDamping_r|force_magnitude');
    expressionForceGoal_toer.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_toer.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_toer.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_toer)
    expressionForceGoal_toel = osim.MocoOutputGoal('expressionForceGoal_toel');
    expressionForceGoal_toel.setOutputPath('/forceset/PassiveToeDamping_l|force_magnitude');
    expressionForceGoal_toel.setExponent(expressionForceExponent); # type: ignore
    expressionForceGoal_toel.setWeight(expressionForceWeight); # type: ignore
    expressionForceGoal_toel.setDivideByDisplacement(True)
    problem.addGoal(expressionForceGoal_toel)

if 'heelAccWeight' in locals():
    # try heel acclerations
    heelGoalr = osim.MocoOutputGoal('heelracc');
    heelGoalr.setOutputPath('bodyset/calcn_r/heelr|acceleration');
    heelGoalr.setExponent(heelAccExponent); # type: ignore
    heelGoalr.setWeight(heelAccWeight); # type: ignore
    heelGoalr.setDivideByDisplacement(True)
    problem.addGoal(heelGoalr)

    # heelGoall = osim.MocoOutputGoal('heellacc');
    # heelGoall.setOutputPath('bodyset/calcn_l/heell|acceleration');
    # heelGoall.setExponent(heelAccExponent);
    # heelGoall.setWeight(heelAccWeight);
    # heelGoalr.setDivideByDisplacement(True)
    # problem.addGoal(heelGoall)
    ## temp remove, since it is in swing, shouldn't matter as much... 


# % Optionally, add a contact tracking goal.
if GRFTrackingWeight != 0:
    # % Track the right and left vertical and fore-aft ground reaction forces.
    contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight)
    # what data are we tracking, GRF exp, or from tight tracking results



    if trackIK:
        contactTracking.setExternalLoadsFile('grf_walk_nat_4ms.xml');
    else:
        # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight.xml'); # grf_walk - Copy
        ## current work around for time changing... not easy way to access the xml and adjust the names and things... 
        # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight_poly_' + str(finalTime*2)[2:] + '.xml')
        contactTracking.setExternalLoadsFile('grf_walk_nat_4ms_tight_poly_' + str(finalTime*2)[2:] + '.xml')

    forceNamesRightFoot = osim.StdVectorString();
    forceNamesRightFoot.append('/contactHeel_r');
    # forceNamesRightFoot.append('/forceset/contactLateralRearfoot_r');
    forceNamesRightFoot.append('/contactLateralMidfoot_r');
    # forceNamesRightFoot.append('/contactLateralToe_r');
    forceNamesRightFoot.append('/contactMedialToe_r');
    forceNamesRightFoot.append('/contactMedialMidfoot_r');
    # contactTracking.addContactGroup(forceNamesRightFoot, 'Right_GRF');
    contactTrackingSplitRight = osim.MocoContactTrackingGoalGroup(forceNamesRightFoot, 'Right_GRF');
    contactTrackingSplitRight.append_alternative_frame_paths('/bodyset/toes_r')
    contactTracking.addContactGroup(contactTrackingSplitRight);

    forceNamesLeftFoot = osim.StdVectorString();
    forceNamesLeftFoot.append('/contactHeel_l');
    # forceNamesLeftFoot.append('/forceset/contactLateralRearfoot_l');
    forceNamesLeftFoot.append('/contactLateralMidfoot_l');
    # forceNamesLeftFoot.append('/contactLateralToe_l');
    forceNamesLeftFoot.append('/contactMedialToe_l');
    forceNamesLeftFoot.append('/contactMedialMidfoot_l');
    # contactTracking.addContactGroup(forceNamesLeftFoot, 'Left_GRF');
    contactTrackingSplitLeft = osim.MocoContactTrackingGoalGroup(forceNamesLeftFoot, 'Left_GRF');
    contactTrackingSplitLeft.append_alternative_frame_paths('/bodyset/toes_l')
    contactTracking.addContactGroup(contactTrackingSplitLeft);
    
    contactTracking.setProjection('plane');
    contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));

    # contactTracking.setDivideByDuration(True)
    contactTracking.setDivideByMass(True)
    problem.addGoal(contactTracking);



# % Bounds
# % ======
helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, trackedfile) # trackedfile

# with initial value bounds
# problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
# problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);
# problem.setStateInfo('/jointset/groundPelvis/pelvis_list/value', [-20*np.pi/180, 20*np.pi/180], [-10*np.pi/180,-2*np.pi/180]) #  
# problem.setStateInfo('/jointset/groundPelvis/pelvis_rotation/speed', [-2.3, 2.3]) # [0]
# problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-0.5, 0.1], [-23*np.pi/180, -10*np.pi/180])
# problem.setStateInfo('/jointset/back/lumbar_bending/value', [-20*np.pi/180, 20*np.pi/180], [0, 5*np.pi/180]) # , [0]
# problem.setStateInfo('/jointset/back/lumbar_rotation/value', [-0.75, 0.75], [-25*np.pi/180, -7*np.pi/180]);
# problem.setStateInfo('/jointset/back/lumbar_rotation/speed', [-9, 9], [0, 9]);
# # have to set custom mtp
# # problem.setStateInfo('/jointset/mtp_r/mtp_angle_r/value', [-0.5, 0.5])
# # problem.setStateInfo('/jointset/mtp_l/mtp_angle_l/value', [-0.5, 0.5])
# # problem.setStateInfo('/jointset/mtp_r/mtp_angle_r/speed', [-10, 10])
# # problem.setStateInfo('/jointset/mtp_l/mtp_angle_l/speed', [-10, 10])
# # problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
# # problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-30*np.pi/180, 60*np.pi/180]);
# # problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-30*np.pi/180, 60*np.pi/180]);
# problem.setStateInfo('/jointset/walker_knee_l/knee_angle_l/value', [0, 2.443]);
# problem.setStateInfo('/jointset/walker_knee_r/knee_angle_r/value', [0, 2.443]);
# problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-40*np.pi/180, 30*np.pi/180]);
# problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-40*np.pi/180, 30*np.pi/180], [-5*np.pi/180,0]);
# # problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);
# problem.setStateInfo('/jointset/acromial_l/arm_flex_l/value', [-70*np.pi/180, 35*np.pi/180])
# problem.setStateInfo('/jointset/acromial_r/arm_flex_r/value', [-70*np.pi/180, 35*np.pi/180])
# problem.setStateInfo('/jointset/elbow_l/elbow_flex_l/value', [45*np.pi/180, 160*np.pi/180])
# problem.setStateInfo('/jointset/elbow_r/elbow_flex_r/value', [45*np.pi/180, 160*np.pi/180])


# without initial value bounds
problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-0.5, 0.1]) #, [-23*np.pi/180, -10*np.pi/180])

problem.setStateInfo('/jointset/groundPelvis/pelvis_list/value', [-20*np.pi/180, 20*np.pi/180]) #, [-10*np.pi/180,-2*np.pi/180]) #  

problem.setStateInfo('/jointset/groundPelvis/pelvis_rotation/value', [-0.24, 0.24]) # [0]
problem.setStateInfo('/jointset/groundPelvis/pelvis_rotation/speed', [-2.3, 2.3]) # [0]

problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);

problem.setStateInfo('/jointset/back/lumbar_bending/value', [-20*np.pi/180, 20*np.pi/180]) # , [0, 5*np.pi/180]) # , [0]
problem.setStateInfo('/jointset/back/lumbar_rotation/value', [-0.75, 0.75]) # , [-25*np.pi/180, -7*np.pi/180]);
problem.setStateInfo('/jointset/back/lumbar_rotation/speed', [-9, 9]) # , [0, 9]);
# have to set custom mtp
# problem.setStateInfo('/jointset/mtp_r/mtp_angle_r/value', [-0.5, 0.5])
# problem.setStateInfo('/jointset/mtp_l/mtp_angle_l/value', [-0.5, 0.5])
# problem.setStateInfo('/jointset/mtp_r/mtp_angle_r/speed', [-10, 10])
# problem.setStateInfo('/jointset/mtp_l/mtp_angle_l/speed', [-10, 10])
# problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
# problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-30*np.pi/180, 60*np.pi/180]);
# problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-30*np.pi/180, 60*np.pi/180]);
problem.setStateInfo('/jointset/walker_knee_l/knee_angle_l/value', [0, 2.443]);
problem.setStateInfo('/jointset/walker_knee_r/knee_angle_r/value', [0, 2.443]);
problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-40*np.pi/180, 30*np.pi/180]);
problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-40*np.pi/180, 30*np.pi/180]) #, [-5*np.pi/180,0]);
# problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);
problem.setStateInfo('/jointset/acromial_l/arm_flex_l/value', [-70*np.pi/180, 35*np.pi/180])
problem.setStateInfo('/jointset/acromial_r/arm_flex_r/value', [-70*np.pi/180, 35*np.pi/180])
problem.setStateInfo('/jointset/elbow_l/elbow_flex_l/value', [45*np.pi/180, 160*np.pi/180])
problem.setStateInfo('/jointset/elbow_r/elbow_flex_r/value', [45*np.pi/180, 160*np.pi/180])

# problem.setTimeBounds(0, [startendtime, endendtime]);

# % Configure the solver
# % ====================
solver = study.initCasADiSolver();
solver.set_optim_finite_difference_scheme('forward')
solver.set_parameters_require_initsystem(False);
duration = endendtime - track.get_initial_time();
num_mesh = round(duration/stepsize);
solver.set_num_mesh_intervals(num_mesh);
solver.set_verbosity(2);
solver.set_optim_solver('ipopt');
solver.set_optim_convergence_tolerance(convergeTolerance);
solver.set_optim_constraint_tolerance(constraintTolerance);
solver.set_optim_max_iterations(maxiterations);
solver.set_scale_variables_using_bounds(True);
solver.set_minimize_implicit_auxiliary_derivatives(True);
solver.set_implicit_auxiliary_derivatives_weight(implicitWeight);

if guess:
    randomguess = solver.createGuess('bounds')
    newguess = helperOsimFunctions.fillGuess(randomguess, guessfile)
    solver.setGuess(newguess);



# %{%
# % Solve the problem
# % =================
gaitTrackingSolution = study.solve();
pdb.set_trace()
testobj = gaitTrackingSolution.getObjective()
gaitTrackingSolution.write(resultspath + '4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
# gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
# % keyboard
# gaitTrackingSolution = osim.MocoTrajectory(resultspath + '4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')
# testobj = 5
# % Create a full stride from the periodic single step solution.
# % For details, navigate to
# % User Guide > Utilities > Model and trajectory utilities
# % in the Moco Documentation.
addPatterns = osim.StdVectorString();
addPatterns.append('.*pelvis_tx/value');

negatePatterns = osim.StdVectorString();
negatePatterns.append('.*pelvis_list.*');
negatePatterns.append(".*pelvis_rotation.*")
negatePatterns.append(".*lumbar_bending.*")
negatePatterns.append(".*lumbar_rotation.*")

negateAndShiftPatterns = osim.StdVectorString()
# negateAndShiftPatterns.append(".*pelvis_tz/value")


# addPatterns = {".*pelvis_tx/value"},
# std::vector< std::string >  negatePatterns = { ".*pelvis_list.*", ".*pelvis_rotation.*", ".*pelvis_tz(?!/value).*", ".*lumbar_bending(?!/value).*", ".*lumbar_rotation.*"}
# std::vector< std::string >  negateAndShiftPatterns = {  ".*pelvis_tz/value", ".*lumbar_bending/value"}

fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution, addPatterns, negatePatterns, negateAndShiftPatterns);
fullStride.write(resultspath + './4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_FullStride.sto');
# write a controls file as well
osim.STOFileAdapter.write(fullStride.exportToControlsTable(), resultspath + '4CMA_exo_short2_compliant2_controls.sto')

## run some analysis
analyzeStrings_vel = osim.StdVectorString()
analyzeStrings_vel.append('.*normalized_fiber_velocity')
table_vel = study.analyze(gaitTrackingSolution, analyzeStrings_vel)
osim.STOFileAdapter.write(table_vel, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_Normalized_Fiber_Velocity.sto')

analyzeStrings_len = osim.StdVectorString()
analyzeStrings_len.append('.*normalized_fiber_length')
table_len = study.analyze(gaitTrackingSolution, analyzeStrings_len)
osim.STOFileAdapter.write(table_len, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_Normalized_Fiber_Length.sto')

analyzeStrings_probe = osim.StdVectorString()
analyzeStrings_probe.append('/metabolic_cost.*')
table_probe = study.analyze(gaitTrackingSolution, analyzeStrings_probe)
osim.STOFileAdapter.write(table_probe, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_metabolics.sto')

analyzeStrings_tenvel = osim.StdVectorString()
analyzeStrings_tenvel.append('.*tendon_velocity')
table_tenvel = study.analyze(fullStride, analyzeStrings_tenvel)
osim.STOFileAdapter.write(table_tenvel, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_tendon_velocity.sto')

analyzeStrings_mtu = osim.StdVectorString(); 
analyzeStrings_mtu.append('.*length'); 
table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
osim.STOFileAdapter.write(table_mtu, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_mtu.sto')

analyzeStrings_pass = osim.StdVectorString(); 
analyzeStrings_pass.append('.*passive_fiber_force'); 
table_pass = study.analyze(fullStride, analyzeStrings_pass); 
osim.STOFileAdapter.write(table_pass, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_passive_fiber_force.sto')

analyzeStrings_activ = osim.StdVectorString(); 
analyzeStrings_activ.append('.*active_fiber_force'); 
table_activ = study.analyze(fullStride, analyzeStrings_activ); 
osim.STOFileAdapter.write(table_activ, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_active_fiber_force.sto')

analyzeStrings_head = osim.StdVectorString(); 
analyzeStrings_head.append('/bodyset/torso/head\\|position');
table_head = osim.analyzeVec3(model, gaitTrackingSolution.exportToStatesTable(), gaitTrackingSolution.exportToControlsTable(), analyzeStrings_head)
osim.STOFileAdapter.write(table_head.flatten(), resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_head_pos.sto') 

analyzeStrings_ToeL = osim.StdVectorString(); 
analyzeStrings_ToeL.append('/bodyset/toes_l/bigToeL\\|position');
table_ToeL = osim.analyzeVec3(model, gaitTrackingSolution.exportToStatesTable(), gaitTrackingSolution.exportToControlsTable(), analyzeStrings_ToeL)
osim.STOFileAdapter.write(table_ToeL.flatten(), resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_bigToeL_pos.sto')

analyzeStrings_ToeR = osim.StdVectorString(); 
analyzeStrings_ToeR.append('/bodyset/toes_r/bigToeR\\|position');
table_ToeR = osim.analyzeVec3(model, gaitTrackingSolution.exportToStatesTable(), gaitTrackingSolution.exportToControlsTable(), analyzeStrings_ToeR)
osim.STOFileAdapter.write(table_ToeR.flatten(), resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_bigToeR_pos.sto')

analyzeStrings_probeind = osim.StdVectorString()
analyzeStrings_probeind.append('.*metabolic_cost.*')
table_probeind = study.analyze(fullStride, analyzeStrings_probeind)
osim.STOFileAdapter.write(table_probeind, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_metabolics_individuals.sto')

analyzeStrings_forcePaths = osim.StdVectorString();
analyzeStrings_forcePaths.append('.*externalloads.*');
# analyzeStrings_forcePaths.append('/contactHeel_r');
# analyzeStrings_forcePaths.append('/contactLateralMidfoot_r');
# analyzeStrings_forcePaths.append('/contactMedialToe_r');
# analyzeStrings_forcePaths.append('/contactMedialMidfoot_r');
# analyzeStrings_forcePaths.append('/contactHeel_l');
# analyzeStrings_forcePaths.append('/contactLateralMidfoot_l');
# analyzeStrings_forcePaths.append('/contactMedialToe_l');
# analyzeStrings_forcePaths.append('/contactMedialMidfoot_l');
analyzeStrings_forcePaths.append('.*contact.*');
table_jointMoments = study.calcGeneralizedForces(fullStride, analyzeStrings_forcePaths);
osim.STOFileAdapter.write(table_jointMoments, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_joint_moments.sto');

analyzeStrings_exoten = osim.StdVectorString(); 
analyzeStrings_exoten.append('/forceset/HOBL.*'); 
table_exoten = study.analyze(fullStride, analyzeStrings_exoten); 
osim.STOFileAdapter.write(table_exoten, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_HOBL.sto')

analyzeStrings_mtu = osim.StdVectorString(); 
analyzeStrings_mtu.append('.*'); 
table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
osim.STOFileAdapter.write(table_mtu, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_all.sto')

analyzeStrings_ma = osim.StdVectorString();
analyzeStrings_ma.append('.*moment_arm');
table_ma = study.analyze(fullStride, analyzeStrings_ma);
osim.STOFileAdapter.write(table_ma, resultspath + '4CMA_exo_short2_compliant2_quickAnalysis_moment_arms.sto')


# % Extract ground reaction forces
# % ==============================
contact_r = osim.StdVectorString();
contact_l = osim.StdVectorString();
contact_r.append('/contactHeel_r');
# contact_r.append('/contactLateralRearfoot_r');
contact_r.append('/contactLateralMidfoot_r');
# contact_r.append('/contactLateralToe_r');
contact_r.append('/contactMedialToe_r');
contact_r.append('/contactMedialMidfoot_r');

contact_l.append('/contactHeel_l');
# contact_l.append('/contactLateralRearfoot_l');
contact_l.append('/contactLateralMidfoot_l');
# contact_l.append('/contactLateralToe_l');
contact_l.append('/contactMedialToe_l');
contact_l.append('/contactMedialMidfoot_l');

externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, 
                                 fullStride,contact_r,contact_l);
osim.STOFileAdapter.write(externalForcesTableFlat, 
                             resultspath + './4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto');
fullstrideGRF = fullStride;
fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
fullstrideGRF.write(resultspath + './4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
print('gaitTrackingSolution to fullstrideGRF:  \n4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n')

helperOsimFunctions.simMetCost(table_probe, TreadmillModel)


state = model.initSystem();
modelmass = model.getTotalMass(state);

tag = '4CMA_exo_short2_compliant2'
## evaluate the errors using the bilevel function
totalerr = bilevelTools.objective_sweep_nat(externalForcesTableFlat, \
                                            fullStride, \
                                            osim.TimeSeriesTable('./expData/Ham19_4ms/GRF_mk13_rv1_4ms_1.mot'), \
                                            osim.TimeSeriesTable('./expData/Ham19_4ms/4_IK_mk13_rv1_nat_trim.mot'), \
                                            testobj, \
                                            x, tag, modelmass, resultspath)
                                            # osim.TimeSeriesTable('./expData/nat_1_GRF.mot'), \ 

# totalerr = bilevelTools.objective_sweep_nat(externalForcesTableFlat, \
#                                             fullStride, \
#                                             osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solutionGRF_FullStride_6835.sto'), \
#                                             osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_kinematicsValues_solution_6835.sto'), \
#                                             testobj, \
#                                             x, tag, modelmass, resultspath)
# generate a report
output = resultspath + '4CMA_exo_short2_compliant2_report.pdf'
report = osim.report.Report(model,
                            resultspath + '4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto',
                            bilateral=True,
                            output=output)
reportFilePath = report.generate()

# optional for creating a comparison report based on whenever both nat and exo scripts are run
# comparisonreport = input("\n\nDo we want to run the comparison report? \nNote: need to have the other solutions written.\n\n 0 for no, 1 for yes.")
comparisonreport = '1'
if comparisonreport != '1':
    comparisonreport = 0
else:
    comparisonreport = int(comparisonreport)

if comparisonreport:
    print(comparisonreport)
    # out2 = resultspath + '4ActMet_3exoActMet_compare_report.pdf'
    out2 = resultspath + '4CMA_exo_short2_compliant2_Naturaltight_compare_report.pdf'
    # ref_files = [
    #         './results/2.7/648/3exoActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #         './results/2.7/648/3exoActMet_controls.sto']
    ref_files = [
            './goodresults/4ms_mk13_poly5/015_4ActMet_684_tight/4ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
            './goodresults/4ms_mk13_poly5/015_4ActMet_684_tight/4ActMet_controls.sto']
    report = osim.report.Report(model,
                                    resultspath + '4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
                                    output=out2, bilateral=True,
                                    ref_files=ref_files)
    # The PDF is saved to the working directory.
    report.generate()



























# now compare to the ik results
comparisonreport = '1'
if comparisonreport != '1':
    comparisonreport = 0
else:
    comparisonreport = int(comparisonreport)

if comparisonreport:
    print(comparisonreport)
    # out2 = resultspath + '4CMA_exo_short2_compliant2_3exoActMet_compare_report.pdf'
    out2 = resultspath + '4CMA_exo_short2_compliant2_NaturalIK_compare_report.pdf'
    # ref_files = [
    #         './results/2.7/648/3exoActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #         './results/2.7/648/3exoActMet_controls.sto']
    ref_files = [
            './expData/Ham19_4ms/4_IK_nat_FullStride.sto',
            './expData/Ham19_4ms/4_IK_nat_controls.sto']
    report = osim.report.Report(model,
                                    resultspath + '4CMA_exo_short2_compliant2_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
                                    output=out2, bilateral=True,
                                    ref_files=ref_files)
    # The PDF is saved to the working directory.
    report.generate()


# % Visualize the solution
pdb.set_trace()
study.visualize(gaitTrackingSolution)

# helperOsimFunctions.syncDrives(localDir, destDir)