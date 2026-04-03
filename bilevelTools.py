## creating a module of helper functions for opensim scripts
# jon stingel
# 20230410
##################################################################
import os
# from turtle import pd

# os.add_dll_directory("C:/OpenSim 4.4/bin")
# os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
# os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")
# os.add_dll_directory('C:/Users/jonstingel/opensim/opensim-core-4.5-2024-05-15-a1a2282/bin')
os.add_dll_directory('C:/Users/jonstingel/opensim-core-4.5.1-2024-08-23-cf3ef35/bin')


import opensim as osim
import pdb
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.optimize import minimize
import helperOsimFunctions
from multiprocessing import Pool
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

# TODO
'''
clean up the different fucntions and calls to them. Can probably simplify down to the one simulation function again, they are the same, 
just whether or not using the met term. need to figure out the all the different objectives and when I want them. and also which moco they 
call. and which CMA functions I am using and which they each call.  
'''

'''
# Define the objective function to be minimized in inverse optimal control problem - fitness function
def objective_bilevel_GA(x):
    # going to try and recreate Umberger paper 2019
    
    # e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results

        # get the outlog file
        outlogfilename = open('GA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable = py_overgroundGait2D_bi(x, 'GA_')
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];

        IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # get vectors of the GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy();
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy();
        # flatten size
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()

        # get predicted hip, knee, ankle
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy())*180/np.pi;
        IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy();

        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy())*180/np.pi;
        IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy();

        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy())*180/np.pi;
        IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy();

        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy();


        # flatten size
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()

        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()

        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()

        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()


        ####################################
        # integrate through time for differences

        # # get the DTW RMSE 
        # distance, path = fastdtw(predvec2, IDvec2) #, dist=euclidean)
        # path_x, path_y = zip(*path)
        # aligned_x = np.array([predvec2[p] for p in path_x])
        # aligned_y = np.array([IDvec2[p] for p in path_y])
        # rmse = np.sqrt(np.mean((aligned_x - aligned_y) ** 2))

        # use DTW to get RMSE across time, and divide by experimental mean
        hgrferr = 10 * helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        vgrferr = 10 * helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        hiperr = 15*helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        kneeerr = 15*helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        ankleerr = 15 * helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        tyvalueerr = 15 * helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)

        # TODO - revisit this as the error function

        #################################
        # then want cost of total duration for the gait cycle
        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\nerr: %f' % err)

        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nerr: %f' % err)
        outlog.close()

        # TODO add all terms
        tot_err = vgrferr + hgrferr + hiperr + kneeerr + ankleerr + tyvalueerr + err 

    except Exception as e:
        # the moco simulation likely failed
        # outlog = open(outlogfile)
        # outlog.write('\n\nInputs:\n')
        # outlog.write(str(x))
        # outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        # outlog.write(str(e))
        # outlog.close()
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err
'''

###########################################################################
# tracking problems for the base (natural) and exotendon - note return respective tight tracking for error. 
###########################################################################
def py_overgroundGait2D_3basetrack(x, tag): #[effortWeight, effortExponent, activationWeight, headWeight, headExponent, implicitAuxWeight]):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % Author: Jon Stingel
    # # % 20230407
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/OpenSim 4.4/bin")
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

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
    actualx = [0, 0.6, 6e-1, 2.68e3, 2.68e0] # testing time

    # g = 0.1

    controlEffortWeight = 0; 
    effortExponent = 2 
    stateTrackingWeight = x[0] # 5e-5 # 10**(x[0]) # * g #* 5e-5
    GRFTrackingWeight   = x[1] # 10**(x[2]) # * g #x[1] * g 

    activationWeight = 8.0 # x[0] # 10**(0.9)   # (x[1]) # * g #* 0.01
    activationWeightEach = 1e0

    metabolicsWeight = 2.4e-4 #x[1] # 10**(x[1]) # * g #* 0.05
    metabolicsExponent = 2


    # ## testing something here
    # # GRF magnitudes
    # forceweight = 3e-5

    # heelForceWeight = forceweight
    # heelForceExponent = 2;
    # toeForceWeight = forceweight
    # toeForceExponent = 2;

    # head tracking 
    headTrackWeight = x[2]

    # pelvis ty tracking
    tyweight = x[3]

    # head accelerations
    # headWeight = 1e-3
    # vitalWeight = 1e-3

    # heel acceleration
    # heelAccWeight = 3e-4
    # heelAccExponent = 2;

    # guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './results/cma/n_fromCMA/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight_intermediate/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'

    implicitWeight = 6e3 # 6e4

    convergeTolerance   = 1e-2
    constraintTolerance = 1e-3
    fractionExtraBoundSize = 0.45

    stepsize = .015
    maxiterations = 1000

    initialTime = 0.0

    finalTime = 0.6835 / 2 #  0.7235  0.6835  0.648  0.608
    startendtime = 0.25
    endendtime = 0.74/2

    guess = True
    wantguess = False

    # trackIK = True
    trackIK = False

    trackedfile = './3CMA_nat_2D3D_muscle_GaitTracking_tracked_states.sto'

    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('3CMA_nat_2D3D_muscle_GaitTracking');


    
    # % Set the OpenSim Model and give it a name
    # TreadmillModel = 'strong_mk13_rv1_dgf.osim'
    TreadmillModel = 'strong_mk14_rv1_dgf.osim'
    modelProcessor = osim.ModelProcessor(TreadmillModel)
    functionBasedPathsFile = './pathResults/HOBLingApoorva-scaled_FunctionBasedPathSet.xml'
    modelProcessor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(functionBasedPathsFile))
    model = modelProcessor.process();

    # % Reference data for tracking problem
    if trackIK:
        tableProcessor = osim.TableProcessor('./expData/2D2Darms/27_IK_nat_mk12_rv1_1.mot'); ########################### OG_test_nat_1_IK
        tableProcessor.append(osim.TabOpLowPassFilter(20));
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames());
    else:
        # new method for stretch/shrinking input values for tracking based on the duration.
        # tableProcessor = osim.TableProcessor('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_tight/3ActMet_kinematicsValues_solution_6835.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_kinematicsValues_solution_6835.sto')
        basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight/3ActMet_kinematicsValues_solution_6835.sto')
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
    ##
    # other thing is to set fast and slow twitches for recruitment
    ##
    # loop and add all the muscles to the model
    for m in range(numMuscles):
        muscle = muscles.get(m)
        muscleName = muscle.getName()
        musclePath = muscle.getAbsolutePathString()
        ratio = helperOsimFunctions.getMuscleFiberRatios(muscleName, 'short2')
        metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 350000)
    # premetmodel.addComponent(metabolics)
    premetmodel.finalizeConnections()
    # premetmodel.printToXML('mk13_rv1_dgf_met.osim')
    modelProcessor = osim.ModelProcessor(premetmodel)
    # else:
    #     modelProcessor = osim.ModelProcessor(model)
    #'''


    # modelProcessor = osim.ModelProcessor(model)
    # make sure our tendons are compliant
    modelProcessor.append(osim.ModOpTendonComplianceDynamicsModeDGF('implicit'));


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
    stateTrackingGoal.setDivideByDisplacement(False)

    '''
    best individual
    [16.4642132  26.17749431  2.40395633  1.40346929  5.81808907  9.45995736
    9.9811225  19.10767123  0.17034188  6.38006386  7.76837126  4.61191839
    8.47651309  2.5277297 ]
    '''
    cmaX = [5.65627447e+00, 4.88501232e+01, 4.42395394e+00, 3.19657882e+00, 2.56978218e+01, 3.84954755e+01, 6.25325640e+01, 3.50247660e+01, 4.34307998e+01, 1.45299279e+01, 5.44550447e+00, 1.16829446e-07, 4.36006879e+00, 1.96495369e+01]

    
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 100/((2.2*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/speed', 100/((2.2*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 50/((2.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed', 50/((2.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/value', 50/((2.3*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/speed', 50/((2.3*np.pi/180)**2));

    # bounds of 1e7 to 1e10
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', tyweight);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', tyweight);

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);

    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 1000/((4.5*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 1000/((18.8*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 8e4/((12.9*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);

    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 1000/((3.0*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 1000/((3.0*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 1000/((4.9*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 1000/((4.9*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/value', 50/((3.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/speed', 50/((3.6*np.pi/180)**2));

    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/speed', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/speed', 50/((2.1*np.pi/180)**2));

    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/speed', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/speed', 100/((1.4*np.pi/180)**2));







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
    speedGoal.set_desired_average_speed(2.67);
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    # tendon velocity bounding
    bounds = osim.MocoBounds(-0.8, 0.8)
    boundsVec = osim.StdVectorMocoBounds()
    boundsVec.append(bounds)

    tenGoal_bfsh_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_r', 10)
    tenGoal_bfsh_r.setMode('endpoint_constraint')
    tenGoal_bfsh_r.setOutputPath('/forceset/bfsh_r|tendon_velocity')
    tenGoal_bfsh_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_r)
    tenGoal_gasmed_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_r', 10)
    tenGoal_gasmed_r.setMode('endpoint_constraint')
    tenGoal_gasmed_r.setOutputPath('/forceset/gasmed_r|tendon_velocity')
    tenGoal_gasmed_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_r)
    tenGoal_soleus_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_r', 10)
    tenGoal_soleus_r.setMode('endpoint_constraint')
    tenGoal_soleus_r.setOutputPath('/forceset/soleus_r|tendon_velocity')
    tenGoal_soleus_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_r)
    tenGoal_tibant_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_r', 10)
    tenGoal_tibant_r.setMode('endpoint_constraint')
    tenGoal_tibant_r.setOutputPath('/forceset/tibant_r|tendon_velocity')
    tenGoal_tibant_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_r)
    tenGoal_vasint_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_r', 10)
    tenGoal_vasint_r.setMode('endpoint_constraint')
    tenGoal_vasint_r.setOutputPath('/forceset/vasint_r|tendon_velocity')
    tenGoal_vasint_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_r)
    tenGoal_recfem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_r', 10)
    tenGoal_recfem_r.setMode('endpoint_constraint')
    tenGoal_recfem_r.setOutputPath('/forceset/recfem_r|tendon_velocity')
    tenGoal_recfem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_r)
    tenGoal_psoas_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_r', 10)
    tenGoal_psoas_r.setMode('endpoint_constraint')
    tenGoal_psoas_r.setOutputPath('/forceset/psoas_r|tendon_velocity')
    tenGoal_psoas_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_r)
    tenGoal_semimem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_r', 10)
    tenGoal_semimem_r.setMode('endpoint_constraint')
    tenGoal_semimem_r.setOutputPath('/forceset/semimem_r|tendon_velocity')
    tenGoal_semimem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_r)
    tenGoal_glmax2_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_r', 10)
    tenGoal_glmax2_r.setMode('endpoint_constraint')
    tenGoal_glmax2_r.setOutputPath('/forceset/glmax2_r|tendon_velocity')
    tenGoal_glmax2_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_r)
    tenGoal_bfsh_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_l', 10)
    tenGoal_bfsh_l.setMode('endpoint_constraint')
    tenGoal_bfsh_l.setOutputPath('/forceset/bfsh_l|tendon_velocity')
    tenGoal_bfsh_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_l)
    tenGoal_gasmed_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_l', 10)
    tenGoal_gasmed_l.setMode('endpoint_constraint')
    tenGoal_gasmed_l.setOutputPath('/forceset/gasmed_l|tendon_velocity')
    tenGoal_gasmed_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_l)
    tenGoal_soleus_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_l', 10)
    tenGoal_soleus_l.setMode('endpoint_constraint')
    tenGoal_soleus_l.setOutputPath('/forceset/soleus_l|tendon_velocity')
    tenGoal_soleus_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_l)
    tenGoal_tibant_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_l', 10)
    tenGoal_tibant_l.setMode('endpoint_constraint')
    tenGoal_tibant_l.setOutputPath('/forceset/tibant_l|tendon_velocity')
    tenGoal_tibant_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_l)
    tenGoal_vasint_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_l', 10)
    tenGoal_vasint_l.setMode('endpoint_constraint')
    tenGoal_vasint_l.setOutputPath('/forceset/vasint_l|tendon_velocity')
    tenGoal_vasint_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_l)
    tenGoal_recfem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_l', 10)
    tenGoal_recfem_l.setMode('endpoint_constraint')
    tenGoal_recfem_l.setOutputPath('/forceset/recfem_l|tendon_velocity')
    tenGoal_recfem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_l)
    tenGoal_psoas_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_l', 10)
    tenGoal_psoas_l.setMode('endpoint_constraint')
    tenGoal_psoas_l.setOutputPath('/forceset/psoas_l|tendon_velocity')
    tenGoal_psoas_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_l)
    tenGoal_semimem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_l', 10)
    tenGoal_semimem_l.setMode('endpoint_constraint')
    tenGoal_semimem_l.setOutputPath('/forceset/semimem_l|tendon_velocity')
    tenGoal_semimem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_l)
    tenGoal_glmax2_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_l', 10)
    tenGoal_glmax2_l.setMode('endpoint_constraint')
    tenGoal_glmax2_l.setOutputPath('/forceset/glmax2_l|tendon_velocity')
    tenGoal_glmax2_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_l)


    # if metabolics in the problem
    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight / 9)
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        metabolicsGoal.setExponent(metabolicsExponent)
        problem.addGoal(metabolicsGoal)

        # test out additional shortening
        # metabolicsshort = osim.MocoOutputGoal('metabolicsshort', 100 * metabolicsWeight / 9)
        # metabolicsshort.setOutputPath('/metabolic_cost|total_shortening_rate')
        # metabolicsshort.setDivideByDisplacement(True)
        # metabolicsshort.setDivideByMass(True)
        # metabolicsshort.setExponent(metabolicsExponent)
        # problem.addGoal(metabolicsshort)

    if 'headTrackWeight' in locals():
        # % track the head positions
        # get the states and stretch/shrink them
        headkinTraj = osim.MocoTrajectory('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight/3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto')
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
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight)
        

        
        
        # what data are we tracking, GRF exp, or from tight tracking results
        if trackIK:
            contactTracking.setExternalLoadsFile('grf_walk_nat_1.xml')
        else:
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight.xml'); # grf_walk - Copy
            ## current work around for time changing... not easy way to access the xml and adjust the names and things... 
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight_poly_' + str(finalTime*2)[2:] + '.xml')
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_extratight_poly_' + str(finalTime*2)[2:] + '.xml')
            contactTracking.setExternalLoadsFile('grf_walk_nat_1_9tight_poly_' + str(finalTime*2)[2:] + '.xml')
        
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
        problem.addGoal(contactTracking);


   # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, trackedfile) # trackedfile

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
    # pdb.set_trace()
    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write('3CMA_nat_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard
    # gaitTrackingSolution = osim.MocoTrajectory('3CMA_nat_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')
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
    fullStride.write('./3CMA_nat_2D3D_OG_muscles_Tracking_solution_FullStride.sto');
    # write a controls file as well 
    osim.STOFileAdapter.write(fullStride.exportToControlsTable(), '3CMA_nat_controls.sto')

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);

    # ## run some analysis
    # analyzeStrings_vel = osim.StdVectorString()
    # analyzeStrings_vel.append('.*normalized_fiber_velocity')
    # table_vel = study.analyze(gaitTrackingSolution, analyzeStrings_vel)
    # osim.STOFileAdapter.write(table_vel, './analysesTools/mk3/quickAnalysis_3CMA_nat_Normalized_Fiber_Velocity.sto')

    # analyzeStrings_len = osim.StdVectorString()
    # analyzeStrings_len.append('.*normalized_fiber_length')
    # table_len = study.analyze(gaitTrackingSolution, analyzeStrings_len)
    # osim.STOFileAdapter.write(table_len, './analysesTools/mk3/quickAnalysis_3CMA_nat_Normalized_Fiber_Length.sto')

    analyzeStrings_probe = osim.StdVectorString()
    analyzeStrings_probe.append('/metabolic_cost.*')
    table_probe = study.analyze(gaitTrackingSolution, analyzeStrings_probe)
    osim.STOFileAdapter.write(table_probe, './analysesTools/mk3/quickAnalysis_3CMA_nat_metabolics.sto')

    # analyzeStrings_tenvel = osim.StdVectorString()
    # analyzeStrings_tenvel.append('.*tendon_velocity')
    # table_tenvel = study.analyze(fullStride, analyzeStrings_tenvel)
    # osim.STOFileAdapter.write(table_tenvel, './analysesTools/mk3/quickAnalysis_3CMA_nat_tendon_velocity.sto')

    # analyzeStrings_mtu = osim.StdVectorString(); 
    # analyzeStrings_mtu.append('.*length'); 
    # table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    # osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_3CMA_nat_mtu.sto')

    # analyzeStrings_mtu = osim.StdVectorString(); 
    # analyzeStrings_mtu.append('.*'); 
    # table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    # osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_3ActMet_mtu.sto')


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
                                 './3CMA_nat_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write('./3CMA_nat_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('gaitTrackingSolution to fullstrideGRF:  \n3CMA_nat_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n')

    metcost = helperOsimFunctions.simMetCost(table_probe, TreadmillModel)

    state = model.initSystem();
    modelmass = model.getTotalMass(state);

    tag = '3CMA_nat'
    ## evaluate the errors using the bilevel function
    # totalerr = bilevelTools.objective_sweep_nat(externalForcesTableFlat, \
    #                                             fullStride, \
    #                                             osim.TimeSeriesTable('./expData/nat_1_GRF.mot'), \
    #                                             osim.TimeSeriesTable('./expData/2D2Darms/27_IK_nat_mk12_rv1_1.mot'), \
    #                                             testobj, \
    #                                             x, tag, modelmass)


    # print('\nWARNING: these analyses have the wrong GRF, so loads and JRA will be wrong... probably...\n\n')
    # solutionFile = '3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # fullSolutionFile = '3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto'
    # # TreadmillModel
    # # statesFile = 'testingmetcoststates.sto'
    # # controlsFile = 'testingmetcostControls.sto'
    # grfFile = 'grf_walk_nat_1.xml'
    # probeFile = './analysesTools/quickAnalysis_3ActMet_ProbeReporter_probes.sto'

    # # helperOsimFunctions.quickAnalyze(solutionFile, TreadmillModel, '', '3ActMet')
    # # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # # pdb.set_trace()

    # helperOsimFunctions.quickAnalyze(fullSolutionFile, TreadmillModel, '', '3ActMet')
    # time.sleep(1)
    # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # generate a report
    # output = './analysesReports/3CMA_nat_report.pdf'
    # report = osim.report.Report(model,
    #                             './3CMA_nat_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto',
    #                             bilateral=True,
    #                             output=output)
    # reportFilePath = report.generate()

    # optional for creating a comparison report based on whenever both nat and exo scripts are run
    # comparisonreport = input("\n\nDo we want to run the comparison report? \nNote: need to have the other solutions written.\n\n 0 for no, 1 for yes.")
    # if comparisonreport != '1':
    #     comparisonreport = 0
    # else:
    #     comparisonreport = int(comparisonreport)

    # if comparisonreport:
    #     print(comparisonreport)
    #     out2 = './analysesReports/comparisons/3ActMet_3exoActMet_compare_report.pdf'
    #     ref_files = [
    #             '3exoActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #             '3exoActMet_controls.sto']
    #     report = osim.report.Report(model,
    #                                     '3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #                                     output=out2, bilateral=True,
    #                                     ref_files=ref_files)
        # # The PDF is saved to the working directory.
        # report.generate()

    # pdb.set_trace()
    # study.visualize(gaitTrackingSolution)

    # helperOsimFunctions.syncDrives(localDir, destDir)

    # pdb.set_trace()
    # helperOsimFunctions.syncDrives(localDir, destDir)
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \  # TODO check on this
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./expData/nat_1_GRF.mot'), \
    osim.TimeSeriesTable('./expData/2D2Darms/27_IK_nat_mk12_rv1_1.mot'), \
    metcost, \
    modelmass
    # testobj
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \ # TODO fix this to go from the inputs, not the guess
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \    # TODO fix this to go from the inputs, not the guess

def py_overgroundGait2D_3exotrack(x, tag):
        # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % Author: Jon Stingel
    # # % 20230407
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/OpenSim 4.4/bin")
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

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
    actualx = [0, 0.6, 6e-1, 2.68e3, 2.68e0] # testing time

    # g = 0.1

    controlEffortWeight = 0; 
    effortExponent = 2 
    stateTrackingWeight = x[0] # 5e-5 # 10**(x[0]) # * g #* 5e-5
    GRFTrackingWeight   = x[1] # 10**(x[2]) # * g #x[1] * g 

    activationWeight = 8.0 # x[0] # 10**(0.9)   # (x[1]) # * g #* 0.01
    activationWeightEach = 1e0

    metabolicsWeight = 2.4e-4 #x[1] # 10**(x[1]) # * g #* 0.05
    metabolicsExponent = 2


    # ## testing something here
    # # GRF magnitudes
    # forceweight = 3e-5

    # heelForceWeight = forceweight
    # heelForceExponent = 2;
    # toeForceWeight = forceweight
    # toeForceExponent = 2;

    # head tracking 
    headTrackWeight = x[2]

    # pelvis ty tracking
    tyweight = x[3]

    # head accelerations
    # headWeight = 1e-3
    # vitalWeight = 1e-3

    # heel acceleration
    # heelAccWeight = 3e-4
    # heelAccExponent = 2;

    # guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './results/cma/n_fromCMA/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    guessfile = './goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight_intermediate/3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'

    implicitWeight = 6e3 # 6e4

    convergeTolerance   = 1e-2;
    constraintTolerance = 1e-3;
    fractionExtraBoundSize = 0.45

    stepsize = .015;
    maxiterations = 1000;

    initialTime = 0.0

    finalTime = 0.648 / 2 #  0.7235  0.6835  0.648  0.608
    startendtime = 0.25
    endendtime = 0.74/2

    guess = True;
    wantguess = False;
    
    # trackIK = True
    trackIK = False

    trackedfile = './3CMA_exo_2D3D_muscle_GaitTracking_tracked_states.sto'

    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('3CMA_exo_2D3D_muscle_GaitTracking');


    
    # % Set the OpenSim Model and give it a name
    # TreadmillModel = 'strong_mk13_rv1_dgf_exo.osim'
    TreadmillModel = 'strong_mk14_rv1_dgf_exo.osim'
    modelProcessor = osim.ModelProcessor(TreadmillModel)
    functionBasedPathsFile = './pathResults/HOBLingApoorva-scaled_FunctionBasedPathSet.xml'
    modelProcessor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(functionBasedPathsFile))
    model = modelProcessor.process();

    
    # % Reference data for tracking problem
    if trackIK:
        tableProcessor = osim.TableProcessor('./expData/2D2Darms/27_IK_nat_mk12_rv1_1.mot'); ########################### OG_test_nat_1_IK
        tableProcessor.append(osim.TabOpLowPassFilter(20));
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames());
    else:
        # new method for stretch/shrinking input values for tracking based on the duration.
        # tableProcessor = osim.TableProcessor('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever_symmetry_2/3ActMet_notendon_kinematicsValues_solution.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_tight/3ActMet_kinematicsValues_solution_6835.sto')
        # basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_extraTight/3ActMet_kinematicsValues_solution_6835.sto')
        basekin = osim.TimeSeriesTable('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight/3ActMet_kinematicsValues_solution_6835.sto')
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
    ##
    # other thing is to set fast and slow twitches for recruitment
    ##
    # loop and add all the muscles to the model
    for m in range(numMuscles):
        muscle = muscles.get(m)
        muscleName = muscle.getName()
        musclePath = muscle.getAbsolutePathString()
        ratio = helperOsimFunctions.getMuscleFiberRatios(muscleName, 'short2')
        metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 350000)
    # premetmodel.addComponent(metabolics)
    premetmodel.finalizeConnections()
    # premetmodel.printToXML('mk13_rv1_dgf_met.osim')
    modelProcessor = osim.ModelProcessor(premetmodel)
    # else:
    #     modelProcessor = osim.ModelProcessor(model)
    #'''


    # modelProcessor = osim.ModelProcessor(model)
    # make sure our tendons are compliant
    modelProcessor.append(osim.ModOpTendonComplianceDynamicsModeDGF('implicit'));


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
    stateTrackingGoal.setDivideByDisplacement(False)

    '''
    best individual
    [16.4642132  26.17749431  2.40395633  1.40346929  5.81808907  9.45995736
    9.9811225  19.10767123  0.17034188  6.38006386  7.76837126  4.61191839
    8.47651309  2.5277297 ]
    '''
    cmaX = [5.65627447e+00, 4.88501232e+01, 4.42395394e+00, 3.19657882e+00, 2.56978218e+01, 3.84954755e+01, 6.25325640e+01, 3.50247660e+01, 4.34307998e+01, 1.45299279e+01, 5.44550447e+00, 1.16829446e-07, 4.36006879e+00, 1.96495369e+01]

    ## new way using the max averaged differences between conditions for the tracking weights. 
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 100/((2.2*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/speed', 100/((2.2*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 50/((2.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed', 50/((2.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/value', 50/((2.3*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_rotation/speed', 50/((2.3*np.pi/180)**2));

    # bounds of 1e7 to 1e10
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', tyweight);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', tyweight);

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);

    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 1000/((4.5*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 1000/((4.5*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 1000/((18.8*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 1000/((18.8*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 8e4/((12.9*np.pi/180)**2))
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 8e4/((12.9*np.pi/180)**2))

    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);

    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 1000/((3.0*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 1000/((3.0*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 1000/((4.9*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 1000/((4.9*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/value', 50/((3.6*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_rotation/speed', 50/((3.6*np.pi/180)**2));

    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/speed', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 50/((2.1*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/speed', 50/((2.1*np.pi/180)**2));

    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/speed', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 100/((1.4*np.pi/180)**2));
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/speed', 100/((1.4*np.pi/180)**2));






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
    # model.printToXML('post_2Dmodel_copy_stiff.osim')

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
    speedGoal.set_desired_average_speed(2.67);
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    # tendon velocity bounding
    bounds = osim.MocoBounds(-0.8, 0.8)
    boundsVec = osim.StdVectorMocoBounds()
    boundsVec.append(bounds)

    tenGoal_bfsh_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_r', 10)
    tenGoal_bfsh_r.setMode('endpoint_constraint')
    tenGoal_bfsh_r.setOutputPath('/forceset/bfsh_r|tendon_velocity')
    tenGoal_bfsh_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_r)
    tenGoal_gasmed_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_r', 10)
    tenGoal_gasmed_r.setMode('endpoint_constraint')
    tenGoal_gasmed_r.setOutputPath('/forceset/gasmed_r|tendon_velocity')
    tenGoal_gasmed_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_r)
    tenGoal_soleus_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_r', 10)
    tenGoal_soleus_r.setMode('endpoint_constraint')
    tenGoal_soleus_r.setOutputPath('/forceset/soleus_r|tendon_velocity')
    tenGoal_soleus_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_r)
    tenGoal_tibant_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_r', 10)
    tenGoal_tibant_r.setMode('endpoint_constraint')
    tenGoal_tibant_r.setOutputPath('/forceset/tibant_r|tendon_velocity')
    tenGoal_tibant_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_r)
    tenGoal_vasint_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_r', 10)
    tenGoal_vasint_r.setMode('endpoint_constraint')
    tenGoal_vasint_r.setOutputPath('/forceset/vasint_r|tendon_velocity')
    tenGoal_vasint_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_r)
    tenGoal_recfem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_r', 10)
    tenGoal_recfem_r.setMode('endpoint_constraint')
    tenGoal_recfem_r.setOutputPath('/forceset/recfem_r|tendon_velocity')
    tenGoal_recfem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_r)
    tenGoal_psoas_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_r', 10)
    tenGoal_psoas_r.setMode('endpoint_constraint')
    tenGoal_psoas_r.setOutputPath('/forceset/psoas_r|tendon_velocity')
    tenGoal_psoas_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_r)
    tenGoal_semimem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_r', 10)
    tenGoal_semimem_r.setMode('endpoint_constraint')
    tenGoal_semimem_r.setOutputPath('/forceset/semimem_r|tendon_velocity')
    tenGoal_semimem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_r)
    tenGoal_glmax2_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_r', 10)
    tenGoal_glmax2_r.setMode('endpoint_constraint')
    tenGoal_glmax2_r.setOutputPath('/forceset/glmax2_r|tendon_velocity')
    tenGoal_glmax2_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_r)
    tenGoal_bfsh_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_l', 10)
    tenGoal_bfsh_l.setMode('endpoint_constraint')
    tenGoal_bfsh_l.setOutputPath('/forceset/bfsh_l|tendon_velocity')
    tenGoal_bfsh_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_l)
    tenGoal_gasmed_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_l', 10)
    tenGoal_gasmed_l.setMode('endpoint_constraint')
    tenGoal_gasmed_l.setOutputPath('/forceset/gasmed_l|tendon_velocity')
    tenGoal_gasmed_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_l)
    tenGoal_soleus_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_l', 10)
    tenGoal_soleus_l.setMode('endpoint_constraint')
    tenGoal_soleus_l.setOutputPath('/forceset/soleus_l|tendon_velocity')
    tenGoal_soleus_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_l)
    tenGoal_tibant_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_l', 10)
    tenGoal_tibant_l.setMode('endpoint_constraint')
    tenGoal_tibant_l.setOutputPath('/forceset/tibant_l|tendon_velocity')
    tenGoal_tibant_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_l)
    tenGoal_vasint_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_l', 10)
    tenGoal_vasint_l.setMode('endpoint_constraint')
    tenGoal_vasint_l.setOutputPath('/forceset/vasint_l|tendon_velocity')
    tenGoal_vasint_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_l)
    tenGoal_recfem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_l', 10)
    tenGoal_recfem_l.setMode('endpoint_constraint')
    tenGoal_recfem_l.setOutputPath('/forceset/recfem_l|tendon_velocity')
    tenGoal_recfem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_l)
    tenGoal_psoas_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_l', 10)
    tenGoal_psoas_l.setMode('endpoint_constraint')
    tenGoal_psoas_l.setOutputPath('/forceset/psoas_l|tendon_velocity')
    tenGoal_psoas_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_l)
    tenGoal_semimem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_l', 10)
    tenGoal_semimem_l.setMode('endpoint_constraint')
    tenGoal_semimem_l.setOutputPath('/forceset/semimem_l|tendon_velocity')
    tenGoal_semimem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_l)
    tenGoal_glmax2_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_l', 10)
    tenGoal_glmax2_l.setMode('endpoint_constraint')
    tenGoal_glmax2_l.setOutputPath('/forceset/glmax2_l|tendon_velocity')
    tenGoal_glmax2_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_l)


    # if metabolics in the problem
    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight / 9)
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        metabolicsGoal.setExponent(metabolicsExponent)
        problem.addGoal(metabolicsGoal)

        # test out additional shortening
        # metabolicsshort = osim.MocoOutputGoal('metabolicsshort', 100 * metabolicsWeight / 9)
        # metabolicsshort.setOutputPath('/metabolic_cost|total_shortening_rate')
        # metabolicsshort.setDivideByDisplacement(True)
        # metabolicsshort.setDivideByMass(True)
        # metabolicsshort.setExponent(metabolicsExponent)
        # problem.addGoal(metabolicsshort)

    if 'headTrackWeight' in locals():
        # % track the head positions
        # get the states and stretch/shrink them
        headkinTraj = osim.MocoTrajectory('./goodresults/27ms_mk13_poly5/015_3ActMet_6835_9tight/3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto')
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
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight)
        

        
        
        # what data are we tracking, GRF exp, or from tight tracking results
        if trackIK:
            contactTracking.setExternalLoadsFile('grf_walk_nat_1.xml')
        else:
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight.xml'); # grf_walk - Copy
            ## current work around for time changing... not easy way to access the xml and adjust the names and things... 
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_tight_poly_' + str(finalTime*2)[2:] + '.xml')
            # contactTracking.setExternalLoadsFile('grf_walk_nat_1_extratight_poly_' + str(finalTime*2)[2:] + '.xml')
            contactTracking.setExternalLoadsFile('grf_walk_nat_1_9tight_poly_' + str(finalTime*2)[2:] + '.xml')
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
        problem.addGoal(contactTracking);



    # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, trackedfile) # trackedfile

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
    # pdb.set_trace()
    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write('3CMA_exo_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard
    # gaitTrackingSolution = osim.MocoTrajectory('3CMA_exo_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')
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
    fullStride.write('./3CMA_exo_2D3D_OG_muscles_Tracking_solution_FullStride.sto');
    # write a controls file as well 
    osim.STOFileAdapter.write(fullStride.exportToControlsTable(), '3CMA_exo_controls.sto')

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);

    # ## run some analysis
    # analyzeStrings_vel = osim.StdVectorString()
    # analyzeStrings_vel.append('.*normalized_fiber_velocity')
    # table_vel = study.analyze(gaitTrackingSolution, analyzeStrings_vel)
    # osim.STOFileAdapter.write(table_vel, './analysesTools/mk3/quickAnalysis_3CMA_exo_Normalized_Fiber_Velocity.sto')

    # analyzeStrings_len = osim.StdVectorString()
    # analyzeStrings_len.append('.*normalized_fiber_length')
    # table_len = study.analyze(gaitTrackingSolution, analyzeStrings_len)
    # osim.STOFileAdapter.write(table_len, './analysesTools/mk3/quickAnalysis_3CMA_exo_Normalized_Fiber_Length.sto')

    analyzeStrings_probe = osim.StdVectorString()
    analyzeStrings_probe.append('/metabolic_cost.*')
    table_probe = study.analyze(gaitTrackingSolution, analyzeStrings_probe)
    osim.STOFileAdapter.write(table_probe, './analysesTools/mk3/quickAnalysis_3CMA_exo_metabolics.sto')

    # analyzeStrings_tenvel = osim.StdVectorString()
    # analyzeStrings_tenvel.append('.*tendon_velocity')
    # table_tenvel = study.analyze(fullStride, analyzeStrings_tenvel)
    # osim.STOFileAdapter.write(table_tenvel, './analysesTools/mk3/quickAnalysis_3CMA_exo_tendon_velocity.sto')

    # analyzeStrings_mtu = osim.StdVectorString(); 
    # analyzeStrings_mtu.append('.*length'); 
    # table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    # osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_3CMA_exo_mtu.sto')

    # # analyzeStrings_mtu = osim.StdVectorString(); 
    # # analyzeStrings_mtu.append('.*'); 
    # # table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    # # osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_3ActMet_mtu.sto')


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
                                 './3CMA_exo_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write('./3CMA_exo_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('gaitTrackingSolution to fullstrideGRF:  \n3CMA_exo_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n')

    metcost = helperOsimFunctions.simMetCost(table_probe, TreadmillModel)

    state = model.initSystem();
    modelmass = model.getTotalMass(state);

    tag = '3CMA_exo'
    ## evaluate the errors using the bilevel function
    # totalerr = bilevelTools.objective_sweep_exo(externalForcesTableFlat, \
    #                                             fullStride, \
    #                                             osim.TimeSeriesTable('./expData/exo_1_GRF.mot'), \
    #                                             osim.TimeSeriesTable('./expData/2D2Darms/27_exo_IK_mk12_rv1_1.mot'), \
    #                                             testobj, \
    #                                             x, tag, modelmass)


    # print('\nWARNING: these analyses have the wrong GRF, so loads and JRA will be wrong... probably...\n\n')
    # solutionFile = '3ActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # fullSolutionFile = '3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto'
    # # TreadmillModel
    # # statesFile = 'testingmetcoststates.sto'
    # # controlsFile = 'testingmetcostControls.sto'
    # grfFile = 'grf_walk_nat_1.xml'
    # probeFile = './analysesTools/quickAnalysis_3ActMet_ProbeReporter_probes.sto'

    # # helperOsimFunctions.quickAnalyze(solutionFile, TreadmillModel, '', '3ActMet')
    # # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # # pdb.set_trace()

    # helperOsimFunctions.quickAnalyze(fullSolutionFile, TreadmillModel, '', '3ActMet')
    # time.sleep(1)
    # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # # generate a report
    # output = './analysesReports/3CMA_exo_report.pdf'
    # report = osim.report.Report(model,
    #                             './3CMA_exo_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto',
    #                             bilateral=True,
    #                             output=output)
    # reportFilePath = report.generate()

    # optional for creating a comparison report based on whenever both nat and exo scripts are run
    # comparisonreport = input("\n\nDo we want to run the comparison report? \nNote: need to have the other solutions written.\n\n 0 for no, 1 for yes.")
    # if comparisonreport != '1':
    #     comparisonreport = 0
    # else:
    #     comparisonreport = int(comparisonreport)

    # if comparisonreport:
    #     print(comparisonreport)
    #     out2 = './analysesReports/comparisons/3ActMet_3exoActMet_compare_report.pdf'
    #     ref_files = [
    #             '3exoActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #             '3exoActMet_controls.sto']
    #     report = osim.report.Report(model,
    #                                     '3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto',
    #                                     output=out2, bilateral=True,
    #                                     ref_files=ref_files)
        # # The PDF is saved to the working directory.
        # report.generate()

    # pdb.set_trace()
    # study.visualize(gaitTrackingSolution)

    # helperOsimFunctions.syncDrives(localDir, destDir)

    # pdb.set_trace()
    # helperOsimFunctions.syncDrives(localDir, destDir)
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \  # TODO check on this
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./expData/exo_1_GRF.mot'), \
    osim.TimeSeriesTable('./expData/2D2Darms/27_exo_IK_mk12_rv1_1.mot'), \
    metcost, \
    modelmass
    # testobj
    # osim.TimeSeriesTable('./guessFiles/exotendonTrackExotendon267/3exo_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \ # TODO fix this to go from the inputs, not the guess
    # osim.TimeSeriesTable('./guessFiles/exotendonTrackNatural267/3exo_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \      # TODO fix this to go from the inputs, not the guess

def py_overgroundGait2D_5basetrack(x, tag): #[effortWeight, effortExponent, activationWeight, headWeight, headExponent, implicitAuxWeight]):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % Author: Jon Stingel
    # # % 20230407
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/OpenSim 4.4/bin")
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

    import opensim as osim
    import re
    import numpy as np
    import pdb
    import helperOsimFunctions
    import bilevelTools

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

    # x = [1e2, 6e-4, 5e-1, 1e0, 8e-3]
    # x = [0e0, 1e-2, 1e-1, 1e0, 1e-3]

    # # baseline above, going to do two iterations, one with high activations and one with high metabolics
    # # higher activations
    # x = [0e0, 1e-2, 1e-1, 2e1, 2e-2]
    # x = [0e0, 3e-3, 2e-2, 8e0, 2e-2]

    # x = [0, 4e-3, 7, 6e0, 1e-2] # from old

    # x = [0, 1e-1, 6, 5e1, 9e-3] # from 2.67

    # x = [0, 1e-2, 7, 2e1, 9e-3] # this is starting point for new model

    # # x = [0, 1e-1, 7, 6e-0, 9e-3] # coming down from high effort towards tracking
    # # x = [0, 1e-1, 7, 6e-3, 9e-6] # coming from high tracking towards effort

    # x = [0, 1e-1, 7, 6e-1, 9e-4] # coming down from high effort towards tracking
    # # x = [0, 1e-1, 7, 6e-2, 9e-5] # coming from high tracking towards effort


    # x = [0, 1e0, 1, 1e0, 1e-4] # good bowl for act 

    # x = [0, 1e-1, 2, 1e-1, 8e-4] # this has been working okay


    # x = [0, 1e-1, 2, 1.0e2, 1.0e-2]
    # x = [0, 1e-1, 2, 5.48e1, 5.48e-3]
    # x = [0, 1e-1, 2, 3.01e1, 3.01e-3]
    # x = [0, 1e-1, 2, 1.65e1, 1.65e-3]
    # x = [0, 1e-1, 2, 9.05e0, 9.05e-4]
    # x = [0, 1e-1, 2, 4.96e0, 4.96e-4]
    # x = [0, 1e-1, 2, 2.72e0, 2.72e-4]
    # x = [0, 1e-1, 2, 1.49e0, 1.49e-4]
    # x = [0, 1e-1, 2, 8.19e-1, 8.19e-5]
    # x = [0, 1e-1, 2, 4.49e-1, 4.49e-5]
    # x = [0, 1e-1, 2, 2.46e-1, 2.46e-5]
    # x = [0, 1e-1, 2, 1.35e-1, 1.35e-5]
    # x = [0, 1e-1, 2, 7.41e-2, 7.41e-6]
    # x = [0, 1e-1, 2, 4.06e-2, 4.06e-6]
    # x = [0, 1e-1, 2, 2.23e-2, 2.23e-6]
    # x = [0, 1e-1, 2, 1.22e-2, 1.22e-6]


    x = [0, 1e-1, 2, 2.72e0, 2.72e-4]


    controlEffortWeight = x[0];  
    effortExponent = 2

    stateTrackingWeight = x[1];
    GRFTrackingWeight   = x[2] * x[1];

    activationWeight = x[3]
    activationWeightEach = 1e0

    metabolicsWeight = x[4]
    metabolicsExponent = 2

    # heelForceExponent = 2
    # heelForceWeight = 1e-2

    implicitWeight = 6e3

    convergeTolerance   = 1e-2;
    constraintTolerance = 1e-3;
    fractionExtraBoundSize = 0.45

    stepsize = .015; # might need smaller than 0.15
    maxiterations = 4000;
    initialTime = 0.0


    finalTime = 0.66 / 2;
    startendtime = 0.25
    endendtime = 0.78/2 # 7235

    guess = True;
    wantguess = False;

    guessfile = './goodresults/5ms_mk12/015_ActMet_tight_66/5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './goodresults/5ms_mk12/015_ActOnly_66/5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # guessfile = './goodresults/5ms_mk12/015_MetOnly_66/5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'



    ### need to remember to copy the results and update the folder for the guess!!!

    trackedfile = './5msActMet_bi_2D3D_muscle_GaitTracking_tracked_states.sto'

    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('5msActMet_bi_2D3D_muscle_GaitTracking');


    # % Set the OpenSim Model and give it a name
    # % TreadmillModel ='Running267_TM.osim'; %uigetfile('*.osim'); %This code will work for all three speed conditions, choose the model you want to run
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms.osim';
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms2D_5ms - Copy_stiff.osim'; ###########################
    # dgfprocess = osim.ModelProcessor(TreadmillModel)
    # dgfprocess.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    # # dgfprocess.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    # # dgfprocess.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    # # model = osim.Model(TreadmillModel);
    # model = dgfprocess.process()
    # model.printToXML('subject_2D_testing_PassiveCal_dgf.osim')
    # TreadmillModel = 'mk3_subject_testing_PassivCal_dgf_tendons_metabolics_vis.osim'
    # TreadmillModel = 'mk3_strong_PassivCal_dgf_tendons_metabolics_vis.osim'
    TreadmillModel = 'mk12_rv1_dgf_met.osim'
    model = osim.Model(TreadmillModel)

    # % Reference data for tracking problem
    # tableProcessor = osim.TableProcessor('./expData/Ham19_5ms/IK_results_5ms_1 - Copy.mot'); # IK_results_5ms_1 
    tableProcessor = osim.TableProcessor('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'); # IK_results_5ms_1 
    tableProcessor.append(osim.TabOpLowPassFilter(20));
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());


    '''
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
    metabolics.set_include_negative_mechanical_work(False)
    ##
    # other thing is to set fast and slow twitches for recruitment
    ##
    # loop and add all the muscles to the model
    for m in range(numMuscles):
        muscle = muscles.get(m)
        muscleName = muscle.getName()
        musclePath = muscle.getAbsolutePathString()
        ratio = helperOsimFunctions.getMuscleFiberRatios(muscleName, 'short2')
        metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 350000)
    # premetmodel.addComponent(metabolics)
    premetmodel.finalizeConnections()
    premetmodel.printToXML('mk3_subject_testing_PassivCal_dgf_tendons_metabolics.osim')
    modelProcessor = osim.ModelProcessor(premetmodel)
    # else:
    #     modelProcessor = osim.ModelProcessor(model)
    '''


    modelProcessor = osim.ModelProcessor(model)
    # make sure our tendons are compliant
    modelProcessor.append(osim.ModOpTendonComplianceDynamicsModeDGF('implicit'));



    # modelProcessor = osim.ModelProcessor(TreadmillModel);
    track.setModel(modelProcessor);
    track.setStatesReference(tableProcessor);
    track.set_states_global_tracking_weight(stateTrackingWeight);
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_apply_tracked_states_to_guess(True);
    track.set_initial_time(0.0);
    # track.set_mesh_interval(stepsize)
    track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
    study = track.initialize();
    problem = study.updProblem();

    # % Goals
    # % =====
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
    # stateTrackingGoal.setDivideByDuration(True)

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1000.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/lumbar/lumbar/value', 10000.0);
    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 1e1)
    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 1e1)
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 1e1)
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 1e1)

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 100.0);

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 100);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed', 100);

    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 1e2)
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 1e2)



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
        if str.startswith(currentStateName , '/jointset') and 'beta' not in currentStateName:# and 'list' not in currentStateName:
            print('\niiii')
            print(currentStateName)

            # take joints out of the state squared goal, only want activations
            activationGoal.setWeightForState(currentStateName, 0)

            if currentStateName.endswith('_r/value') or currentStateName.endswith('_r/speed'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('1 - rights and lefts - pair')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if currentStateName.endswith('_l/value') or currentStateName.endswith('_l/speed'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
                print('2 - lefts and rights - pair')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            if (currentStateName.endswith('_bending/value') or currentStateName.endswith('_bending/speed') or
                currentStateName.endswith('_list/value') or currentStateName.endswith('_list/speed') or 
                currentStateName.endswith('_tz/value') or currentStateName.endswith('_tz/speed')):
                if currentStateName.endswith('value'):
                    symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                    print('3 - pelvis bending, list, tz - pair value')
                    print(currentStateName)
                if currentStateName.endswith('speed'):
                    symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                    print('3 - pelvis list, lumbar bending, tz - negated pair speed')
                    print(currentStateName)
            if currentStateName.endswith('_rotation/value') or currentStateName.endswith('_rotation/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('4 - pelvis and lumbar rotation - negated pair')
                print(currentStateName)
            if currentStateName.endswith('_tx/speed'): # overground so not value
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('5 - pelvis tx speed - pair speed')
                print(currentStateName)
            if (currentStateName.endswith('_tilt/value') or currentStateName.endswith('_tilt/speed') or
                currentStateName.endswith('_extension/value') or currentStateName.endswith('_extension/speed')):
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('6 - pelvis tilt, lumbar extension - pair value and speed')
                print(currentStateName)
            if currentStateName.endswith('_ty/value') or currentStateName.endswith('_ty/speed'):
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('7 - pelvis ty - pair value and speed')
                print(currentStateName)
        if 'beta' in currentStateName:
            print('\niiii')
            print(currentStateName)
            activationGoal.setWeightForState(currentStateName, 0)

    print('\n\naaaaaaaaaaaaaaaaaaaahahaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # % Symmetric muscle activations
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):
            print('\naaaa')
            print(currentStateName)
            # activations squared for this actuator
            activationGoal.setWeightForState(currentStateName, activationWeightEach)

            if currentStateName.endswith('_r/activation'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('a1 - rights and lefts - pair')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if currentStateName.endswith('_l/activation'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('a2 - lefts and rights - pair')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            # bending , rotation , tz, list - these are gonna be different 
            if 'Bend' in currentStateName or 'Rot' in currentStateName:
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('a3 - lumbar bending, lumbar rotation actuators - negated pair')
                print(currentStateName)
            if 'Ext' in currentStateName:
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('a3 - lumbar extension actuator -  pair')
                print(currentStateName)



    # % Get a reference to the MocoControlGoal that is added to every MocoTrack
    # % problem by default and change the weight
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
    effort.setWeight(controlEffortWeight);
    effort.setDivideByDisplacement(False);
    effort.setExponent(effortExponent)
    if controlEffortWeight == 0:
        effort.setEnabled(False)
    else:
        effort.setEnabled(True)


    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(5);
    # speedGoal.setWeight(10)
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    # initial tendon velocity bounds goal
        # tendonStrings = osim.StdVectorString()
        # tendonStrings.append('/forceset/bfsh_r|tendon_velocity')
        # tendonStrings.append('/forceset/gasmed_r|tendon_velocity')
        # tendonStrings.append('/forceset/soleus_r|tendon_velocity')
        # tendonStrings.append('/forceset/tibant_r|tendon_velocity')
        # tendonStrings.append('/forceset/vasint_r|tendon_velocity')
        # tendonStrings.append('/forceset/recfem_r|tendon_velocity')
        # tendonStrings.append('/forceset/psoas_r|tendon_velocity')
        # tendonStrings.append('/forceset/semimem_r|tendon_velocity')
        # tendonStrings.append('/forceset/glmax2_r|tendon_velocity')
        # tendonStrings.append('/forceset/bfsh_l|tendon_velocity')
        # tendonStrings.append('/forceset/gasmed_l|tendon_velocity')
        # tendonStrings.append('/forceset/soleus_l|tendon_velocity')
        # tendonStrings.append('/forceset/tibant_l|tendon_velocity')
        # tendonStrings.append('/forceset/vasint_l|tendon_velocity')
        # tendonStrings.append('/forceset/recfem_l|tendon_velocity')
        # tendonStrings.append('/forceset/psoas_l|tendon_velocity')
        # tendonStrings.append('/forceset/semimem_l|tendon_velocity')
        # tendonStrings.append('/forceset/glmax2_l|tendon_velocity')
    # pdb.set_trace()

    bounds = osim.MocoBounds(-0.8, 0.8)
    boundsVec = osim.StdVectorMocoBounds()
    boundsVec.append(bounds)

    tenGoal_bfsh_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_r', 10)
    tenGoal_bfsh_r.setMode('endpoint_constraint')
    tenGoal_bfsh_r.setOutputPath('/forceset/bfsh_r|tendon_velocity')
    tenGoal_bfsh_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_r)
    tenGoal_gasmed_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_r', 10)
    tenGoal_gasmed_r.setMode('endpoint_constraint')
    tenGoal_gasmed_r.setOutputPath('/forceset/gasmed_r|tendon_velocity')
    tenGoal_gasmed_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_r)
    tenGoal_soleus_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_r', 10)
    tenGoal_soleus_r.setMode('endpoint_constraint')
    tenGoal_soleus_r.setOutputPath('/forceset/soleus_r|tendon_velocity')
    tenGoal_soleus_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_r)
    tenGoal_tibant_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_r', 10)
    tenGoal_tibant_r.setMode('endpoint_constraint')
    tenGoal_tibant_r.setOutputPath('/forceset/tibant_r|tendon_velocity')
    tenGoal_tibant_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_r)
    tenGoal_vasint_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_r', 10)
    tenGoal_vasint_r.setMode('endpoint_constraint')
    tenGoal_vasint_r.setOutputPath('/forceset/vasint_r|tendon_velocity')
    tenGoal_vasint_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_r)
    tenGoal_recfem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_r', 10)
    tenGoal_recfem_r.setMode('endpoint_constraint')
    tenGoal_recfem_r.setOutputPath('/forceset/recfem_r|tendon_velocity')
    tenGoal_recfem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_r)
    tenGoal_psoas_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_r', 10)
    tenGoal_psoas_r.setMode('endpoint_constraint')
    tenGoal_psoas_r.setOutputPath('/forceset/psoas_r|tendon_velocity')
    tenGoal_psoas_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_r)
    tenGoal_semimem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_r', 10)
    tenGoal_semimem_r.setMode('endpoint_constraint')
    tenGoal_semimem_r.setOutputPath('/forceset/semimem_r|tendon_velocity')
    tenGoal_semimem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_r)
    tenGoal_glmax2_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_r', 10)
    tenGoal_glmax2_r.setMode('endpoint_constraint')
    tenGoal_glmax2_r.setOutputPath('/forceset/glmax2_r|tendon_velocity')
    tenGoal_glmax2_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_r)
    tenGoal_bfsh_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_l', 10)
    tenGoal_bfsh_l.setMode('endpoint_constraint')
    tenGoal_bfsh_l.setOutputPath('/forceset/bfsh_l|tendon_velocity')
    tenGoal_bfsh_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_l)
    tenGoal_gasmed_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_l', 10)
    tenGoal_gasmed_l.setMode('endpoint_constraint')
    tenGoal_gasmed_l.setOutputPath('/forceset/gasmed_l|tendon_velocity')
    tenGoal_gasmed_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_l)
    tenGoal_soleus_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_l', 10)
    tenGoal_soleus_l.setMode('endpoint_constraint')
    tenGoal_soleus_l.setOutputPath('/forceset/soleus_l|tendon_velocity')
    tenGoal_soleus_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_l)
    tenGoal_tibant_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_l', 10)
    tenGoal_tibant_l.setMode('endpoint_constraint')
    tenGoal_tibant_l.setOutputPath('/forceset/tibant_l|tendon_velocity')
    tenGoal_tibant_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_l)
    tenGoal_vasint_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_l', 10)
    tenGoal_vasint_l.setMode('endpoint_constraint')
    tenGoal_vasint_l.setOutputPath('/forceset/vasint_l|tendon_velocity')
    tenGoal_vasint_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_l)
    tenGoal_recfem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_l', 10)
    tenGoal_recfem_l.setMode('endpoint_constraint')
    tenGoal_recfem_l.setOutputPath('/forceset/recfem_l|tendon_velocity')
    tenGoal_recfem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_l)
    tenGoal_psoas_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_l', 10)
    tenGoal_psoas_l.setMode('endpoint_constraint')
    tenGoal_psoas_l.setOutputPath('/forceset/psoas_l|tendon_velocity')
    tenGoal_psoas_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_l)
    tenGoal_semimem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_l', 10)
    tenGoal_semimem_l.setMode('endpoint_constraint')
    tenGoal_semimem_l.setOutputPath('/forceset/semimem_l|tendon_velocity')
    tenGoal_semimem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_l)
    tenGoal_glmax2_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_l', 10)
    tenGoal_glmax2_l.setMode('endpoint_constraint')
    tenGoal_glmax2_l.setOutputPath('/forceset/glmax2_l|tendon_velocity')
    tenGoal_glmax2_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_l)


    # if metabolics in the problem
    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight)
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        metabolicsGoal.setExponent(metabolicsExponent)
        problem.addGoal(metabolicsGoal)


    if 'heelForceWeight' in locals():
        # trying a contact force goal to not slam feet
        heelForceGoal_l = osim.MocoOutputGoal('heelforce_l');
        heelForceGoal_l.setOutputPath('contactHeel_l|sphere_force');
        heelForceGoal_l.setExponent(heelForceExponent); # type: ignore
        heelForceGoal_l.setWeight(heelForceWeight); # type: ignore
        problem.addGoal(heelForceGoal_l);

        heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
        heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
        heelForceGoal_r.setExponent(heelForceExponent); # type: ignore
        heelForceGoal_r.setWeight(heelForceWeight); # type: ignore
        problem.addGoal(heelForceGoal_r);


    # % Optionally, add a contact tracking goal.
    if GRFTrackingWeight != 0:
        # % Track the right and left vertical and fore-aft ground reaction forces.
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight);
        contactTracking.setExternalLoadsFile('grf_walk_nat_5ms.xml'); # grf_walk - Copy
        
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
        
        # contactTracking.setProjection('plane');
        # contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));

        # contactTracking.setDivideByDuration(True)
        problem.addGoal(contactTracking);



    # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, finalTime, trackedfile) # trackedfile

    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);

    problem.setStateInfo('/jointset/groundPelvis/pelvis_list/value', [-90*np.pi/180, 90*np.pi/180], [0])
    problem.setStateInfo('/jointset/back/lumbar_bending/value', [-90*np.pi/180, 90*np.pi/180], [0])

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
    # problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-40*np.pi/180, 30*np.pi/180]);
    # problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-40*np.pi/180, 30*np.pi/180]);
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
    # pdb.set_trace()
    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write('5msActMet_bi_bi_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard
    # gaitTrackingSolution = osim.MocoTrajectory('5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')
    # testobj = 5
    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    # addPatterns = osim.StdVectorString();
    # addPatterns.append('.*pelvis_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution);
    fullStride.write('./5msActMet_bi_2D3D_OG_muscles_Tracking_solution_FullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);


    ## run some analysis
    analyzeStrings_vel = osim.StdVectorString()
    analyzeStrings_vel.append('.*normalized_fiber_velocity')
    table_vel = study.analyze(fullStride, analyzeStrings_vel)
    osim.STOFileAdapter.write(table_vel, './analysesTools/mk3/quickAnalysis_5msActMet_bi_Normalized_Fiber_Velocity.sto')

    analyzeStrings_len = osim.StdVectorString()
    analyzeStrings_len.append('.*normalized_fiber_length')
    table_len = study.analyze(fullStride, analyzeStrings_len)
    osim.STOFileAdapter.write(table_len, './analysesTools/mk3/quickAnalysis_5msActMet_bi_Normalized_Fiber_Length.sto')

    analyzeStrings_probe = osim.StdVectorString()
    analyzeStrings_probe.append('/metabolic_cost.*')
    table_probe = study.analyze(fullStride, analyzeStrings_probe)
    osim.STOFileAdapter.write(table_probe, './analysesTools/mk3/quickAnalysis_5msActMet_bi_metabolics.sto')

    analyzeStrings_tenvel = osim.StdVectorString()
    analyzeStrings_tenvel.append('.*tendon_velocity')
    table_tenvel = study.analyze(fullStride, analyzeStrings_tenvel)
    osim.STOFileAdapter.write(table_tenvel, './analysesTools/mk3/quickAnalysis_5msActMet_bi_tendon_velocity.sto')

    analyzeStrings_mtu = osim.StdVectorString(); 
    analyzeStrings_mtu.append('.*length'); 
    table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_5msActMet_bi_mtu.sto')

    # analyzeStrings_mtu = osim.StdVectorString(); 
    # analyzeStrings_mtu.append('.*length'); 
    # table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    # osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_5msActMet_mtu.sto')



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
                                 './5msActMet_bi_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write('./5msActMet_bi_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitTrackingSolution to fullstrideGRF:  5msActMet_bi_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n\n')


    ## output some analyses
    helperOsimFunctions.simMetCost(table_probe, TreadmillModel)


    # tag = '5msActMet_bi'
    # ## evaluate the errors using the bilevel function
    # totalerr = bilevelTools.objective_sweep_nat(externalForcesTableFlat, \
    #                                             fullStride, \
    #                                             osim.TimeSeriesTable('./expData/Ham19_5ms/GRF_mk12_rv1_5ms_1.mot'), \
    #                                             osim.TimeSeriesTable('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'), \
    #                                             testobj, \
    #                                             x, tag)



    # # print('\nWARNING: these analyses have the wrong GRF, so loads and JRA will be wrong... probably...\n\n')
    # # solutionFile = '5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # # fullSolutionFile = '5msActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto'

    # # # TreadmillModel
    # # # statesFile = 'testingmetcoststates.sto'
    # # # controlsFile = 'testingmetcostControls.sto'
    # # grfFile = 'grf_walk_nat_1.xml'
    # # probeFile = './analysesTools/quickAnalysis_5msActMet_ProbeReporter_probes.sto'

    # # # helperOsimFunctions.quickAnalyze(solutionFile, TreadmillModel, '', '5ms')
    # # helperOsimFunctions.quickAnalyze(fullSolutionFile, TreadmillModel, '', '5msActMet')
    # # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # # generate a report
    # output = './analysesReports/5msActMet_report.pdf'
    # report = osim.report.Report(model,
    #                             './5msActMet_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto',
    #                             bilateral=True,
    #                             output=output)
    # reportFilePath = report.generate()


    # pdb.set_trace()
    # study.visualize(gaitTrackingSolution)

    # helperOsimFunctions.syncDrives(localDir, destDir)


    # pdb.set_trace()
    # helperOsimFunctions.syncDrives(localDir, destDir)
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \  # TODO check on this
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('../expData/Ham19_5ms/GRF_mk12_rv1_5ms_1.mot'), \
    osim.TimeSeriesTable('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'), \
    testobj

    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \ # TODO fix this to go from the inputs, not the guess
    # osim.TimeSeriesTable('./guessFiles/naturalTrack267/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \    # TODO fix this to go from the inputs, not the guess

def py_overgroundGait2D_5exotrack(x, tag):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % Author: Jon Stingel
    # # % 20240226
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/OpenSim 4.4/bin")
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

    import opensim as osim
    import re
    import numpy as np
    import pdb
    import helperOsimFunctions
    import bilevelTools

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

    # x = [1e2, 6e-4, 5e-1, 1e0, 8e-3]
    # x = [0e0, 1e-2, 1e-1, 1e0, 1e-3]

    # # baseline above, going to do two iterations, one with high activations and one with high metabolics
    # # higher activations
    # x = [0e0, 1e-2, 1e-1, 2e1, 2e-2]
    # x = [0e0, 3e-3, 2e-2, 8e0, 2e-2]

    # x = [0, 4e-3, 7, 6e0, 1e-2] # from old

    # x = [0, 1e-1, 6, 5e1, 9e-3] # from 2.67

    # x = [0, 1e-2, 7, 2e1, 9e-3]


    # # x = [0, 1e-1, 7, 6e-1, 9e-4] # coming down from high effort towards tracking
    # # x = [0, 1e-1, 7, 6e-2, 9e-5] # coming from high tracking towards effort


    # x = [0, 1e-1, 1, 1e0, 1e-4] # still not getting a bowl

    # x = [0, 1e-1, 2, 1e-1, 8e-4] # this has been working okay


    # x = [0, 1e-1, 2, 1.0e2, 1.0e-2]
    # x = [0, 1e-1, 2, 5.48e1, 5.48e-3]
    # x = [0, 1e-1, 2, 3.01e1, 3.01e-3]
    # x = [0, 1e-1, 2, 1.65e1, 1.65e-3]
    # x = [0, 1e-1, 2, 9.05e0, 9.05e-4]
    # x = [0, 1e-1, 2, 4.96e0, 4.96e-4]
    # x = [0, 1e-1, 2, 2.72e0, 2.72e-4]
    # x = [0, 1e-1, 2, 1.49e0, 1.49e-4]
    # x = [0, 1e-1, 2, 8.19e-1, 8.19e-5]
    # x = [0, 1e-1, 2, 4.49e-1, 4.49e-5]
    # x = [0, 1e-1, 2, 2.46e-1, 2.46e-5]
    # x = [0, 1e-1, 2, 1.35e-1, 1.35e-5]
    # x = [0, 1e-1, 2, 7.41e-2, 7.41e-6]
    # x = [0, 1e-1, 2, 4.06e-2, 4.06e-6]
    # x = [0, 1e-1, 2, 2.23e-2, 2.23e-6]
    # x = [0, 1e-1, 2, 1.22e-2, 1.22e-6]


    x = [0, 1e-1, 2, 2.72e0, 2.72e-4]


    controlEffortWeight = x[0];  
    effortExponent = 2

    stateTrackingWeight = x[1];
    GRFTrackingWeight   = x[2] * x[1];

    activationWeight = x[3]
    activationWeightEach = 1e0

    metabolicsWeight = x[4]
    metabolicsExponent = 2

    # heelForceExponent = 2
    # heelForceWeight = 1e-2

    implicitWeight = 6e3

    convergeTolerance   = 1e-2;
    constraintTolerance = 1e-3;
    fractionExtraBoundSize = 0.45

    stepsize = .015; # might need smaller than 0.15
    maxiterations = 4000;
    initialTime = 0.0


    finalTime = 0.66 / 2;
    startendtime = 0.25
    endendtime = 0.78/2 # .7235

    guess = True;
    wantguess = False;

    guessfile = './goodresults/5ms_mk12/015_ActMet_tight_66/5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    ### need to remember to copy the results and update the folder for the guess!!!

    trackedfile = './5msexoActMet_bi_2D3D_muscle_GaitTracking_tracked_states.sto'

    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('5msexoActMet_bi_2D3D_muscle_GaitTracking');


    # % Set the OpenSim Model and give it a name
    # % TreadmillModel ='Running267_TM.osim'; %uigetfile('*.osim'); %This code will work for all three speed conditions, choose the model you want to run
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms.osim';
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms2D_5ms - Copy_stiff.osim'; ###########################
    # dgfprocess = osim.ModelProcessor(TreadmillModel)
    # dgfprocess.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    # # dgfprocess.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    # # dgfprocess.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    # # model = osim.Model(TreadmillModel);
    # model = dgfprocess.process()
    # model.printToXML('subject_2D_testing_PassiveCal_dgf.osim')
    TreadmillModel = 'mk12_rv1_dgf_met_exo.osim'
    model = osim.Model(TreadmillModel)

    # % Reference data for tracking problem
    tableProcessor = osim.TableProcessor('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'); # IK_results_5ms_1 
    tableProcessor.append(osim.TabOpLowPassFilter(20));
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());


    '''
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
    metabolics.set_include_negative_mechanical_work(False)
    ##
    # other thing is to set fast and slow twitches for recruitment
    ##
    # loop and add all the muscles to the model
    for m in range(numMuscles):
        muscle = muscles.get(m)
        muscleName = muscle.getName()
        musclePath = muscle.getAbsolutePathString()
        ratio = helperOsimFunctions.getMuscleFiberRatios(muscleName, 'short2')
        metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)), ratio, 350000)
    # premetmodel.addComponent(metabolics)
    premetmodel.finalizeConnections()
    premetmodel.printToXML('mk3_subject_testing_PassivCal_dgf_tendons_metabolics.osim')
    modelProcessor = osim.ModelProcessor(premetmodel)
    # else:
    #     modelProcessor = osim.ModelProcessor(model)
    '''


    modelProcessor = osim.ModelProcessor(model)
    # make sure our tendons are compliant
    modelProcessor.append(osim.ModOpTendonComplianceDynamicsModeDGF('implicit'));



    # modelProcessor = osim.ModelProcessor(TreadmillModel);
    track.setModel(modelProcessor);
    track.setStatesReference(tableProcessor);
    track.set_states_global_tracking_weight(stateTrackingWeight);
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_apply_tracked_states_to_guess(True);
    track.set_initial_time(0.0);
    # track.set_mesh_interval(stepsize)
    track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
    study = track.initialize();
    problem = study.updProblem();

    # % Goals
    # % =====
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
    # stateTrackingGoal.setDivideByDuration(True)

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1000.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/lumbar/lumbar/value', 10000.0);
    stateTrackingGoal.setWeightForState('/jointset/acromial_r/arm_flex_r/value', 1e4)
    stateTrackingGoal.setWeightForState('/jointset/elbow_r/elbow_flex_r/value', 1e4)
    stateTrackingGoal.setWeightForState('/jointset/acromial_l/arm_flex_l/value', 1e4)
    stateTrackingGoal.setWeightForState('/jointset/elbow_l/elbow_flex_l/value', 1e4)

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tilt/value', 100.0);

    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/value', 100);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_list/speed', 100);

    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/value', 1e2)
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_bending/speed', 1e2)



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
        if str.startswith(currentStateName , '/jointset') and 'beta' not in currentStateName:# and 'list' not in currentStateName:
            print('\niiii')
            print(currentStateName)

            # take joints out of the state squared goal, only want activations
            activationGoal.setWeightForState(currentStateName, 0)

            if currentStateName.endswith('_r/value') or currentStateName.endswith('_r/speed'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('1 - rights and lefts - pair')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if currentStateName.endswith('_l/value') or currentStateName.endswith('_l/speed'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName,re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
                print('2 - lefts and rights - pair')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            if (currentStateName.endswith('_bending/value') or currentStateName.endswith('_bending/speed') or
                currentStateName.endswith('_list/value') or currentStateName.endswith('_list/speed') or 
                currentStateName.endswith('_tz/value') or currentStateName.endswith('_tz/speed')):
                if currentStateName.endswith('value'):
                    symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                    print('3 - pelvis bending, list, tz - pair value')
                    print(currentStateName)
                if currentStateName.endswith('speed'):
                    symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                    print('3 - pelvis list, lumbar bending, tz - negated pair speed')
                    print(currentStateName)
            if currentStateName.endswith('_rotation/value') or currentStateName.endswith('_rotation/speed'):
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('4 - pelvis and lumbar rotation - negated pair')
                print(currentStateName)
            if currentStateName.endswith('_tx/speed'): # overground so not value
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('5 - pelvis tx speed - pair speed')
                print(currentStateName)
            if (currentStateName.endswith('_tilt/value') or currentStateName.endswith('_tilt/speed') or
                currentStateName.endswith('_extension/value') or currentStateName.endswith('_extension/speed')):
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('6 - pelvis tilt, lumbar extension - pair value and speed')
                print(currentStateName)
            if currentStateName.endswith('_ty/value') or currentStateName.endswith('_ty/speed'):
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('7 - pelvis ty - pair value and speed')
                print(currentStateName)
        if 'beta' in currentStateName:
            print('\niiii')
            print(currentStateName)
            activationGoal.setWeightForState(currentStateName, 0)

    print('\n\naaaaaaaaaaaaaaaaaaaahahaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # % Symmetric muscle activations
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):
            print('\naaaa')
            print(currentStateName)
            # activations squared for this actuator
            activationGoal.setWeightForState(currentStateName, activationWeightEach)

            if currentStateName.endswith('_r/activation'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('a1 - rights and lefts - pair')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if currentStateName.endswith('_l/activation'):
                pair = osim.MocoPeriodicityGoalPair(currentStateName, re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('a2 - lefts and rights - pair')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            # bending , rotation , tz, list - these are gonna be different 
            if 'Bend' in currentStateName or 'Rot' in currentStateName:
                symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('a3 - lumbar bending, lumbar rotation actuators - negated pair')
                print(currentStateName)
            if 'Ext' in currentStateName:
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('a3 - lumbar extension actuator -  pair')
                print(currentStateName)



    # % Get a reference to the MocoControlGoal that is added to every MocoTrack
    # % problem by default and change the weight
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
    effort.setWeight(controlEffortWeight);
    effort.setDivideByDisplacement(False);
    effort.setExponent(effortExponent)
    if controlEffortWeight == 0:
        effort.setEnabled(False)
    else:
        effort.setEnabled(True)


    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(5);
    # speedGoal.setWeight(10)
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    # initial tendon velocity bounds goal
        # tendonStrings = osim.StdVectorString()
        # tendonStrings.append('/forceset/bfsh_r|tendon_velocity')
        # tendonStrings.append('/forceset/gasmed_r|tendon_velocity')
        # tendonStrings.append('/forceset/soleus_r|tendon_velocity')
        # tendonStrings.append('/forceset/tibant_r|tendon_velocity')
        # tendonStrings.append('/forceset/vasint_r|tendon_velocity')
        # tendonStrings.append('/forceset/recfem_r|tendon_velocity')
        # tendonStrings.append('/forceset/psoas_r|tendon_velocity')
        # tendonStrings.append('/forceset/semimem_r|tendon_velocity')
        # tendonStrings.append('/forceset/glmax2_r|tendon_velocity')
        # tendonStrings.append('/forceset/bfsh_l|tendon_velocity')
        # tendonStrings.append('/forceset/gasmed_l|tendon_velocity')
        # tendonStrings.append('/forceset/soleus_l|tendon_velocity')
        # tendonStrings.append('/forceset/tibant_l|tendon_velocity')
        # tendonStrings.append('/forceset/vasint_l|tendon_velocity')
        # tendonStrings.append('/forceset/recfem_l|tendon_velocity')
        # tendonStrings.append('/forceset/psoas_l|tendon_velocity')
        # tendonStrings.append('/forceset/semimem_l|tendon_velocity')
        # tendonStrings.append('/forceset/glmax2_l|tendon_velocity')
    # pdb.set_trace()

    bounds = osim.MocoBounds(-0.8, 0.8)
    boundsVec = osim.StdVectorMocoBounds()
    boundsVec.append(bounds)

    tenGoal_bfsh_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_r', 10)
    tenGoal_bfsh_r.setMode('endpoint_constraint')
    tenGoal_bfsh_r.setOutputPath('/forceset/bfsh_r|tendon_velocity')
    tenGoal_bfsh_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_r)
    tenGoal_gasmed_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_r', 10)
    tenGoal_gasmed_r.setMode('endpoint_constraint')
    tenGoal_gasmed_r.setOutputPath('/forceset/gasmed_r|tendon_velocity')
    tenGoal_gasmed_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_r)
    tenGoal_soleus_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_r', 10)
    tenGoal_soleus_r.setMode('endpoint_constraint')
    tenGoal_soleus_r.setOutputPath('/forceset/soleus_r|tendon_velocity')
    tenGoal_soleus_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_r)
    tenGoal_tibant_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_r', 10)
    tenGoal_tibant_r.setMode('endpoint_constraint')
    tenGoal_tibant_r.setOutputPath('/forceset/tibant_r|tendon_velocity')
    tenGoal_tibant_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_r)
    tenGoal_vasint_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_r', 10)
    tenGoal_vasint_r.setMode('endpoint_constraint')
    tenGoal_vasint_r.setOutputPath('/forceset/vasint_r|tendon_velocity')
    tenGoal_vasint_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_r)
    tenGoal_recfem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_r', 10)
    tenGoal_recfem_r.setMode('endpoint_constraint')
    tenGoal_recfem_r.setOutputPath('/forceset/recfem_r|tendon_velocity')
    tenGoal_recfem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_r)
    tenGoal_psoas_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_r', 10)
    tenGoal_psoas_r.setMode('endpoint_constraint')
    tenGoal_psoas_r.setOutputPath('/forceset/psoas_r|tendon_velocity')
    tenGoal_psoas_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_r)
    tenGoal_semimem_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_r', 10)
    tenGoal_semimem_r.setMode('endpoint_constraint')
    tenGoal_semimem_r.setOutputPath('/forceset/semimem_r|tendon_velocity')
    tenGoal_semimem_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_r)
    tenGoal_glmax2_r = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_r', 10)
    tenGoal_glmax2_r.setMode('endpoint_constraint')
    tenGoal_glmax2_r.setOutputPath('/forceset/glmax2_r|tendon_velocity')
    tenGoal_glmax2_r.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_r)
    tenGoal_bfsh_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_bfsh_l', 10)
    tenGoal_bfsh_l.setMode('endpoint_constraint')
    tenGoal_bfsh_l.setOutputPath('/forceset/bfsh_l|tendon_velocity')
    tenGoal_bfsh_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_bfsh_l)
    tenGoal_gasmed_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_gasmed_l', 10)
    tenGoal_gasmed_l.setMode('endpoint_constraint')
    tenGoal_gasmed_l.setOutputPath('/forceset/gasmed_l|tendon_velocity')
    tenGoal_gasmed_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_gasmed_l)
    tenGoal_soleus_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_soleus_l', 10)
    tenGoal_soleus_l.setMode('endpoint_constraint')
    tenGoal_soleus_l.setOutputPath('/forceset/soleus_l|tendon_velocity')
    tenGoal_soleus_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_soleus_l)
    tenGoal_tibant_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_tibant_l', 10)
    tenGoal_tibant_l.setMode('endpoint_constraint')
    tenGoal_tibant_l.setOutputPath('/forceset/tibant_l|tendon_velocity')
    tenGoal_tibant_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_tibant_l)
    tenGoal_vasint_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_vasint_l', 10)
    tenGoal_vasint_l.setMode('endpoint_constraint')
    tenGoal_vasint_l.setOutputPath('/forceset/vasint_l|tendon_velocity')
    tenGoal_vasint_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_vasint_l)
    tenGoal_recfem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_recfem_l', 10)
    tenGoal_recfem_l.setMode('endpoint_constraint')
    tenGoal_recfem_l.setOutputPath('/forceset/recfem_l|tendon_velocity')
    tenGoal_recfem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_recfem_l)
    tenGoal_psoas_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_psoas_l', 10)
    tenGoal_psoas_l.setMode('endpoint_constraint')
    tenGoal_psoas_l.setOutputPath('/forceset/psoas_l|tendon_velocity')
    tenGoal_psoas_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_psoas_l)
    tenGoal_semimem_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_semimem_l', 10)
    tenGoal_semimem_l.setMode('endpoint_constraint')
    tenGoal_semimem_l.setOutputPath('/forceset/semimem_l|tendon_velocity')
    tenGoal_semimem_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_semimem_l)
    tenGoal_glmax2_l = osim.MocoInitialOutputGoal('tendonVelocityGoal_glmax2_l', 10)
    tenGoal_glmax2_l.setMode('endpoint_constraint')
    tenGoal_glmax2_l.setOutputPath('/forceset/glmax2_l|tendon_velocity')
    tenGoal_glmax2_l.setEndpointConstraintBounds(boundsVec)
    problem.addGoal(tenGoal_glmax2_l)



    # if metabolics in the problem
    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight)
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        metabolicsGoal.setExponent(metabolicsExponent)
        problem.addGoal(metabolicsGoal)


    if 'heelForceWeight' in locals():
        # trying a contact force goal to not slam feet
        heelForceGoal_l = osim.MocoOutputGoal('heelforce_l');
        heelForceGoal_l.setOutputPath('contactHeel_l|sphere_force');
        heelForceGoal_l.setExponent(heelForceExponent); # type: ignore
        heelForceGoal_l.setWeight(heelForceWeight); # type: ignore
        problem.addGoal(heelForceGoal_l);

        heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
        heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
        heelForceGoal_r.setExponent(heelForceExponent); # type: ignore
        heelForceGoal_r.setWeight(heelForceWeight); # type: ignore
        problem.addGoal(heelForceGoal_r);


    # % Optionally, add a contact tracking goal.
    if GRFTrackingWeight != 0:
        # % Track the right and left vertical and fore-aft ground reaction forces.
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight);
        contactTracking.setExternalLoadsFile('grf_walk_nat_5ms.xml'); # grf_walk - Copy
        
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
        
        # contactTracking.setProjection('plane');
        # contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));

        # contactTracking.setDivideByDuration(True)
        problem.addGoal(contactTracking);



    # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, finalTime, trackedfile) # trackedfile

    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);

    problem.setStateInfo('/jointset/groundPelvis/pelvis_list/value', [-90*np.pi/180, 90*np.pi/180], [0])
    problem.setStateInfo('/jointset/back/lumbar_bending/value', [-90*np.pi/180, 90*np.pi/180], [0])

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
    # problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-40*np.pi/180, 30*np.pi/180]);
    # problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-40*np.pi/180, 30*np.pi/180]);
    # problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);

    problem.setStateInfo('/jointset/acromial_l/arm_flex_l/value', [-70*np.pi/180, 35*np.pi/180])
    problem.setStateInfo('/jointset/acromial_r/arm_flex_r/value', [-70*np.pi/180, 35*np.pi/180])

    problem.setStateInfo('/jointset/elbow_l/elbow_flex_l/value', [45*np.pi/180, 150*np.pi/180])
    problem.setStateInfo('/jointset/elbow_r/elbow_flex_r/value', [45*np.pi/180, 150*np.pi/180])

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
    # pdb.set_trace()
    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write('5msexoActMet_bi_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # pdb.set_trace()
    # gaitTrackingSolution = osim.MocoTrajectory('5msexoActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')
    # testobj = 5
    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    # addPatterns = osim.StdVectorString();
    # addPatterns.append('.*pelvis_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution);
    fullStride.write('./5msexoActMet_bi_2D3D_OG_muscles_Tracking_solution_FullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);


    ## run some analysis
    analyzeStrings_vel = osim.StdVectorString()
    analyzeStrings_vel.append('.*normalized_fiber_velocity')
    table_vel = study.analyze(fullStride, analyzeStrings_vel)
    osim.STOFileAdapter.write(table_vel, './analysesTools/mk3/quickAnalysis_5msexoActMet_bi_Normalized_Fiber_Velocity.sto')

    analyzeStrings_len = osim.StdVectorString()
    analyzeStrings_len.append('.*normalized_fiber_length')
    table_len = study.analyze(fullStride, analyzeStrings_len)
    osim.STOFileAdapter.write(table_len, './analysesTools/mk3/quickAnalysis_5msexoActMet_bi_Normalized_Fiber_Length.sto')

    analyzeStrings_probe = osim.StdVectorString()
    analyzeStrings_probe.append('/metabolic_cost.*')
    table_probe = study.analyze(fullStride, analyzeStrings_probe)
    osim.STOFileAdapter.write(table_probe, './analysesTools/mk3/quickAnalysis_5msexoActMet_bi_metabolics.sto')

    analyzeStrings_tenvel = osim.StdVectorString()
    analyzeStrings_tenvel.append('.*tendon_velocity')
    table_tenvel = study.analyze(fullStride, analyzeStrings_tenvel)
    osim.STOFileAdapter.write(table_tenvel, './analysesTools/mk3/quickAnalysis_5msexoActMet_bi_tendon_velocity.sto')

    analyzeStrings_mtu = osim.StdVectorString(); 
    analyzeStrings_mtu.append('.*length'); 
    table_mtu = study.analyze(fullStride, analyzeStrings_mtu); 
    osim.STOFileAdapter.write(table_mtu, './analysesTools/mk3/quickAnalysis_5msexoActMet_bi_mtu.sto')


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
                                 './5msexoActMet_bi_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write('./5msexoActMet_bi_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitTrackingSolution to fullstrideGRF:  5msexoActMet_bi_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n\n')


    helperOsimFunctions.simMetCost(table_probe, TreadmillModel)


    # tag = '5msexoActMet_bi'
    # ## evaluate the errors using the bilevel function
    # totalerr = bilevelTools.objective_sweep_exo(externalForcesTableFlat, \
    #                                             fullStride, \
    #                                             osim.TimeSeriesTable('./expData/Ham19_5ms/GRF_mk12_rv1_5ms_1.mot'), \
    #                                             osim.TimeSeriesTable('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'), \
    #                                             testobj, \
    #                                             x, tag)



    # print('\nWARNING: these analyses have the wrong GRF, so loads and JRA will be wrong... probably...\n\n')
    # solutionFile = '5msActMet_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # fullSolutionFile = '5msActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto'

    # # TreadmillModel
    # # statesFile = 'testingmetcoststates.sto'
    # # controlsFile = 'testingmetcostControls.sto'
    # grfFile = 'grf_walk_nat_1.xml'
    # probeFile = './analysesTools/quickAnalysis_5msActMet_ProbeReporter_probes.sto'

    # # helperOsimFunctions.quickAnalyze(solutionFile, TreadmillModel, '', '5ms')
    # helperOsimFunctions.quickAnalyze(fullSolutionFile, TreadmillModel, '', '5msActMet')
    # helperOsimFunctions.quickMetCost(probeFile, TreadmillModel)

    # generate a report
    output = './analysesReports/5msexoActMet_bi_report.pdf'
    report = osim.report.Report(model,
                                './5msexoActMet_bi_2D3D_OG_muscles_Tracking_solution_fullStride_wGRF.sto',
                                bilateral=True,
                                output=output)
    reportFilePath = report.generate()


    # pdb.set_trace()
    # study.visualize(gaitTrackingSolution)

    # helperOsimFunctions.syncDrives(localDir, destDir)
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \  # TODO check on this
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./expData/Ham19_5ms/GRF_mk12_rv1_5ms_1.mot'), \
    osim.TimeSeriesTable('./expData/Ham19_5ms/IK_mk12_rv1_5ms_1.mot'), \
    testobj
    # osim.TimeSeriesTable('./guessFiles/exotendonTrackExotendon267/3exo_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \ # TODO fix this to go from the inputs, not the guess
    # osim.TimeSeriesTable('./guessFiles/exotendonTrackNatural267/3exo_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \      # TODO fix this to go from the inputs, not the guess

def objective_bilevel_CMATrackTight(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    # pdb.set_trace()

    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        tag = 'CMATrack_'

        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveValnat, modelmass = py_overgroundGait2D_3basetrack(x, tag)
        
        # load in the matlab standard devs for the coordinates
        # standevsData = loadmat('coordinates2StandardDeviations.mat')
        # standevs = standevsData['standevs']
        # standevsdf = pd.DataFrame(standevs)
        # grfdevsData = loadmat('standardDevs_ExternalForces.mat')
        # grfdevs = grfdevsData['standevs']

        grfDevs_nat = pd.read_csv('std_externalForces_nat.csv')
        grfDevs_both = pd.read_csv('std_externalForces_both.csv')
        grfDevs_exo = pd.read_csv('std_externalForces_exo.csv')
        coordDevs_nat = pd.read_csv('std_coords_nat.csv')
        coordDevs_both = pd.read_csv('std_coords_both.csv')
        coordDevs_exo = pd.read_csv('std_coords_exo.csv')

        stdCompares = {}
        fig, ax = plt.subplots(1,2, figsize=(8,3)) # , dpi=300
        ax = ax.flatten()
        count = 0

        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_y = gaitIDTable.getDependentColumn(tempname[0:-1] + 'y').to_numpy();
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # vgf_std = grfdevs['calcn_r_Right_GRF_Fy_nat'][0][0][0][0]
        vgf_std = grfDevs_nat['calcn_r_Right_GRF_Fy']
        # TODO figure out how to do normalization for the GRF values or un normalize the stddev data. 
        tempreturngrfy = helperOsimFunctions.coordSTDCompare(predvec2_y/(modelmass*9.81), IDvec2_y/(modelmass*9.81), inDegrees, vgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfy[0:3]
        fig = tempreturngrfy[3]
        ax = tempreturngrfy[4]
        ax[count].set_ylabel('GRFy', fontsize=12)
        count += 1


        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_x = gaitIDTable.getDependentColumn(tempname[0:-1] + 'x').to_numpy();
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # hgf_std = grfdevs['calcn_r_Right_GRF_Fx_nat'][0][0][0][0]
        hgf_std = grfDevs_nat['calcn_r_Right_GRF_Fx']
        tempreturngrfx = helperOsimFunctions.coordSTDCompare(predvec2_x/(modelmass*9.81), IDvec2_x/(modelmass*9.81), inDegrees, hgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfx[0:3]
        fig = tempreturngrfx[3]
        ax = tempreturngrfx[4]
        ax[count].set_ylabel('GRFx', fontsize=12)
        count += 1


        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        # get the std from gait data
        # pelvis_ty_std = standevs['pelvisTy_nat'][0][0][0][0]
        pelvis_ty_std = coordDevs_nat['pelvis_ty']
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        tempreturnTy = helperOsimFunctions.coordSTDCompare(predvec2_ty_value, IKvec2_ty_value, inDegrees, pelvis_ty_std) #, fig, ax, count)
        stdCompares['pelvis_ty'] = tempreturnTy[0:3]
        # fig = tempreturnTy[3]
        # ax = tempreturnTy[4]
        # ax[count].set_ylabel('pelvisTy', fontsize=12)
        count += 1

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 

        # pelvis list
        predvec_list_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy());
        IKvec_list_value = kinemIKTable.getDependentColumn('pelvis_list').to_numpy(); # /jointset/groundPelvis/pelvis_list/value
        predvec2_list_value = predvec_list_value.flatten()
        IKvec2_list_value = IKvec_list_value.flatten()
        listerr = helperOsimFunctions.dtw_rmse(predvec2_list_value, IKvec2_list_value, inDegrees)
        # pelvislist_std = standevs['pelvisList_nat'][0][0][0][0]
        pelvislist_std = coordDevs_nat['pelvis_list']
        tempreturnlist = helperOsimFunctions.coordSTDCompare(predvec2_list_value, IKvec2_list_value, inDegrees, pelvislist_std) #, fig, ax, count)
        stdCompares['pelvis_list'] = tempreturnlist[0:3]
        # fig = tempreturnlist[3]
        # ax = tempreturnlist[4]
        # ax[count].set_ylabel('pelvis list', fontsize=12)
        count += 1
        # pelvis rotation
        predvec_rotation_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy());
        IKvec_rotation_value = kinemIKTable.getDependentColumn('pelvis_rotation').to_numpy(); # /jointset/groundPelvis/pelvis_rotation/value
        predvec2_rotation_value = predvec_rotation_value.flatten()
        IKvec2_rotation_value = IKvec_rotation_value.flatten()
        rotationerr = helperOsimFunctions.dtw_rmse(predvec2_rotation_value, IKvec2_rotation_value, inDegrees)
        # pelvisrotation_std = standevs['pelvisRotation_nat'][0][0][0][0]
        pelvisrotation_std = coordDevs_nat['pelvis_rotation']
        tempreturnrotation = helperOsimFunctions.coordSTDCompare(predvec2_rotation_value, IKvec2_rotation_value, inDegrees, pelvisrotation_std) #, fig, ax, count)
        stdCompares['pelvis_rotation'] = tempreturnrotation[0:3]
        # fig = tempreturnrotation[3]
        # ax = tempreturnrotation[4]
        # ax[count].set_ylabel('pelvis rotation', fontsize=12)
        count += 1
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)
        # pelvisTilt_std = standevs['pelvisTilt_nat'][0][0][0][0]
        pelvisTilt_std = coordDevs_nat['pelvis_tilt']
        tempreturntilt = helperOsimFunctions.coordSTDCompare(predvec2_tilt_value, IKvec2_tilt_value, inDegrees, pelvisTilt_std) #, fig, ax, count)
        stdCompares['pelvis_tilt'] = tempreturntilt[0:3]
        # fig = tempreturntilt[3]
        # ax = tempreturntilt[4]
        # ax[count].set_ylabel('pelvis tilt', fontsize=12)
        count += 1

        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # hip_flex_std = standevs['hipFlexionR_nat'][0][0][0][0]
        hip_flex_std = coordDevs_nat['hip_flexion_r']
        tempreturnhipflex = helperOsimFunctions.coordSTDCompare(predvec2_hip, IKvec2_hip, inDegrees, hip_flex_std) #, fig, ax, count)
        stdCompares['hip_flexion_r'] = tempreturnhipflex[0:3]
        # fig = tempreturnhipflex[3]
        # ax = tempreturnhipflex[4]
        # ax[count].set_ylabel('hip flexion', fontsize=12)
        count += 1
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # knee_flex_std = standevs['kneeAngleR_nat'][0][0][0][0]
        knee_flex_std = coordDevs_nat['knee_angle_r']
        tempreturnkneeflex = helperOsimFunctions.coordSTDCompare(predvec2_knee, IKvec2_knee, inDegrees, knee_flex_std) #, fig, ax, count)
        stdCompares['knee_angle_r'] = tempreturnkneeflex[0:3]
        # fig = tempreturnkneeflex[3]
        # ax = tempreturnkneeflex[4]
        # ax[count].set_ylabel('knee flexion', fontsize=12)
        count += 1
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # ankle_flex_std = standevs['ankleAngleR_nat'][0][0][0][0]
        ankle_flex_std = coordDevs_nat['ankle_angle_r']
        tempreturnankle = helperOsimFunctions.coordSTDCompare(predvec2_ankle, IKvec2_ankle, inDegrees, ankle_flex_std) #, fig, ax, count)
        stdCompares['ankle_angle_r'] = tempreturnankle[0:3]
        # fig = tempreturnankle[3]
        # ax = tempreturnankle[4]
        # ax[count].set_ylabel('ankle angle', fontsize=12)
        count += 1
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))
        # mtp_flex_std = standevs['mtpAngleR_nat'][0][0][0][0]
        mtp_flex_std = coordDevs_nat['mtp_angle_r']
        tempreturnmtp = helperOsimFunctions.coordSTDCompare(predvec2_mtp, IKvec2_mtp, inDegrees, mtp_flex_std) #, fig, ax, count)
        stdCompares['mtp_angle_r'] = tempreturnmtp[0:3]
        # fig = tempreturnmtp[3]
        # ax = tempreturnmtp[4]
        # ax[count].set_ylabel('mtp angle', fontsize=12)
        count += 1

        # lumbar extension
        predvec_lumbarExt = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy());
        try:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar_extension').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarExt = predvec_lumbarExt.flatten()
        IKvec2_lumbarExt = IKvec_lumbarExt.flatten()
        lumbarExterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees) # 
        # lumbarExt_std = standevs['lumbarExtension_nat'][0][0][0][0]
        lumbarExt_std = coordDevs_nat['lumbar_extension']
        tempreturnlumbarExt = helperOsimFunctions.coordSTDCompare(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees, lumbarExt_std) #, fig, ax, count)
        stdCompares['lumbar_extension'] = tempreturnlumbarExt[0:3]
        # fig = tempreturnlumbarExt[3]
        # ax = tempreturnlumbarExt[4]
        # ax[count].set_ylabel('lumbar extension', fontsize=12)
        count += 1
        # lumbar bending
        predvec_lumbarBend = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy());
        try:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar_bending').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarBend = predvec_lumbarBend.flatten()
        IKvec2_lumbarBend = IKvec_lumbarBend.flatten()
        lumbarBenderr = helperOsimFunctions.dtw_rmse(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees) # 
        # lumbarBend_std = standevs['lumbarBending_nat'][0][0][0][0]
        lumbarBend_std = coordDevs_nat['lumbar_bending']
        tempreturnlumbarBend = helperOsimFunctions.coordSTDCompare(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees, lumbarBend_std) #, fig, ax, count)
        stdCompares['lumbar_bending'] = tempreturnlumbarBend[0:3]
        # fig = tempreturnlumbarBend[3]
        # ax = tempreturnlumbarBend[4]
        # ax[count].set_ylabel('lumbar bending', fontsize=12)
        count += 1
        # lumbar rotation
        predvec_lumbarRot = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy());
        try:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar_rotation').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarRot = predvec_lumbarRot.flatten()
        IKvec2_lumbarRot = IKvec_lumbarRot.flatten()
        lumbarRoterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees) # 
        # lumbarRot_std = standevs['lumbarRotation_nat'][0][0][0][0]
        lumbarRot_std = coordDevs_nat['lumbar_rotation']
        tempreturnlumbarRot = helperOsimFunctions.coordSTDCompare(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees, lumbarRot_std) #, fig, ax, count)
        stdCompares['lumbar_rotation'] = tempreturnlumbarRot[0:3]
        # fig = tempreturnlumbarRot[3]
        # ax = tempreturnlumbarRot[4]
        # ax[count].set_ylabel('lumbar rotation', fontsize=12)
        count += 1

        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        
        # check out the plot
        # plt.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./analysesReports/' + tag + '_kinematics_withinRange.png', dpi=300)



        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr | lumbarExterr lumbarBenderr lumbarRoterr
        kinerr =  (50*vgrferr + 50*hgrferr + 
                hiperr + kneeerr + ankleerr + acromerr + elbowerr + 
                lumbarExterr + lumbarBenderr + lumbarRoterr + 
                listerr + rotationerr + tilterr) # + mtperr + err 
        kinerrNat = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: natural %s \n' % tag)
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('hgrferr: %f' % hgrferr)
        print('vgrferr: %f' % vgrferr)
        print('hiperr: %f' % hiperr)
        print('kneeerr: %f' % kneeerr)
        print('ankleerr: %f' % ankleerr)
        print('mtperr: %f' % mtperr)
        print('acromerr: %f' % acromerr)
        print('elbowerr: %f' % elbowerr)

        # print('\nlumbarerr: %f' % lumbarerr)
        print('lumbarExterr: %f' % lumbarExterr)
        print('lumbarBenderr: %f' % lumbarBenderr)
        print('lumbarRoterr: %f' % lumbarRoterr)

        print('pelvis ty err: %f' % tyvalueerr)
        print('pelvis list err: %f' % listerr)
        print('pelvis rotation err: %f' % rotationerr)
        print('pelvis tilt err: %f' % tilterr)

        print('predictiontime: %f' % predictiontime)
        print('IKTime: %f' % gaitIDtime)
        print('err: %f' % err)
        # print('\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('met cost: %f' % objectiveValnat)
        print('tot_err: %f \n\n' % kinerrNat)
        # pdb.set_trace()
        for each in stdCompares.keys():
            print(each + ': ' + str(stdCompares[each][0:2]))
            # print(stdCompares[each])
        # print(stdCompares)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nlumbarExterr: %f' % lumbarExterr)
        outlog.write('\nlumbarBenderr: %f' % lumbarBenderr)
        outlog.write('\nlumbarRoterr: %f' % lumbarRoterr)

        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nlisterr: %f' % listerr)
        outlog.write('\npelvrotationerr: %f' % rotationerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\met cost: %f' % objectiveValnat)
        outlog.write('\ntot_err: %f' % kinerrNat)
        for each in stdCompares.keys():
            outlog.write('\n')
            outlog.write(each + ': ' + str(stdCompares[each][0:2]))
            # outlog.write('\n')
            # outlog.write(str(stdCompares[each]))
        outlog.close()
        # print('\nafter file stuff')

        # tot_err = kinerr

        


        tag = 'CMATrack_'

        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveValexo, modelmass = py_overgroundGait2D_3exotrack(x, tag)


        # load in the std data for basic exo comparison
        # standevsData = loadmat('coordinates2StandardDeviations.mat')
        # standevs = standevsData['standevs']
        # standevsdf = pd.DataFrame(standevs)
        # and std dev data for the GRF
        # grfdevsData = loadmat('standardDevs_ExternalForces.mat')
        # grfdevs = grfdevsData['standevs']

        stdCompares = {}
        fig, ax = plt.subplots(1,2, figsize=(8,3)) # , dpi=300
        ax = ax.flatten()
        count = 0

        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()

        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_y = gaitIDTable.getDependentColumn(tempname[0:-1] + 'y').to_numpy();
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # vgf_std = grfdevs['calcn_r_Right_GRF_Fy_exo'][0][0][0][0]
        vgf_std = grfDevs_exo['calcn_r_Right_GRF_Fy']
        tempreturngrfy = helperOsimFunctions.coordSTDCompare(predvec2_y/(modelmass*9.81), IDvec2_y/(modelmass*9.81), inDegrees, vgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfy[0:3]
        fig = tempreturngrfy[3]
        ax = tempreturngrfy[4]
        ax[count].set_ylabel('GRFy', fontsize=12)
        count += 1


        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_x = gaitIDTable.getDependentColumn(tempname[0:-1] + 'x').to_numpy();
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # hgf_std = grfdevs['calcn_r_Right_GRF_Fx_exo'][0][0][0][0]
        hgf_std = grfDevs_exo['calcn_r_Right_GRF_Fx']
        tempreturngrfx = helperOsimFunctions.coordSTDCompare(predvec2_x/(modelmass*9.81), IDvec2_x/(modelmass*9.81), inDegrees, hgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfx[0:3]
        fig = tempreturngrfx[3]
        ax = tempreturngrfx[4]
        ax[count].set_ylabel('GRFx', fontsize=12)
        count += 1


        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        # get the std from gait data
        # pelvis_ty_std = standevs['pelvisTy_exo'][0][0][0][0]
        pelvis_ty_std = coordDevs_exo['pelvis_ty']
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        tempreturnTy = helperOsimFunctions.coordSTDCompare(predvec2_ty_value, IKvec2_ty_value, inDegrees, pelvis_ty_std) #, fig, ax, count)
        stdCompares['pelvis_ty'] = tempreturnTy[0:3]
        # fig = tempreturnTy[3]
        # ax = tempreturnTy[4]
        # ax[count].set_ylabel('pelvisTy', fontsize=12)
        count += 1

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 

        # pelvis list
        predvec_list_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy());
        IKvec_list_value = kinemIKTable.getDependentColumn('pelvis_list').to_numpy(); # /jointset/groundPelvis/pelvis_list/value
        predvec2_list_value = predvec_list_value.flatten()
        IKvec2_list_value = IKvec_list_value.flatten()
        listerr = helperOsimFunctions.dtw_rmse(predvec2_list_value, IKvec2_list_value, inDegrees)
        # pelvislist_std = standevs['pelvisList_exo'][0][0][0][0]
        pelvislist_std = coordDevs_exo['pelvis_list']
        tempreturnlist = helperOsimFunctions.coordSTDCompare(predvec2_list_value, IKvec2_list_value, inDegrees, pelvislist_std) #, fig, ax, count)
        stdCompares['pelvis_list'] = tempreturnlist[0:3]
        # fig = tempreturnlist[3]
        # ax = tempreturnlist[4]
        # ax[count].set_ylabel('pelvis list', fontsize=12)
        count += 1
        # pelvis rotation
        predvec_rotation_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy());
        IKvec_rotation_value = kinemIKTable.getDependentColumn('pelvis_rotation').to_numpy(); # /jointset/groundPelvis/pelvis_rotation/value
        predvec2_rotation_value = predvec_rotation_value.flatten()
        IKvec2_rotation_value = IKvec_rotation_value.flatten()
        rotationerr = helperOsimFunctions.dtw_rmse(predvec2_rotation_value, IKvec2_rotation_value, inDegrees)
        # pelvisrotation_std = standevs['pelvisRotation_exo'][0][0][0][0]
        pelvisrotation_std = coordDevs_exo['pelvis_rotation']
        tempreturnrotation = helperOsimFunctions.coordSTDCompare(predvec2_rotation_value, IKvec2_rotation_value, inDegrees, pelvisrotation_std) #, fig, ax, count)
        stdCompares['pelvis_rotation'] = tempreturnrotation[0:3]
        # fig = tempreturnrotation[3]
        # ax = tempreturnrotation[4]
        # ax[count].set_ylabel('pelvis rotation', fontsize=12)
        count += 1
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)
        # pelvisTilt_std = standevs['pelvisTilt_exo'][0][0][0][0]
        pelvisTilt_std = coordDevs_exo['pelvis_tilt']
        tempreturntilt = helperOsimFunctions.coordSTDCompare(predvec2_tilt_value, IKvec2_tilt_value, inDegrees, pelvisTilt_std) #, fig, ax, count)
        stdCompares['pelvis_tilt'] = tempreturntilt[0:3]
        # fig = tempreturntilt[3]
        # ax = tempreturntilt[4]
        # ax[count].set_ylabel('pelvis tilt', fontsize=12)
        count += 1
        
        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # hip_flex_std = standevs['hipFlexionR_exo'][0][0][0][0]
        hip_flex_std = coordDevs_exo['hip_flexion_r']
        tempreturnhipflex = helperOsimFunctions.coordSTDCompare(predvec2_hip, IKvec2_hip, inDegrees, hip_flex_std) #, fig, ax, count)
        stdCompares['hip_flexion_r'] = tempreturnhipflex[0:3]
        # fig = tempreturnhipflex[3]
        # ax = tempreturnhipflex[4]
        # ax[count].set_ylabel('hip flexion', fontsize=12)
        count += 1
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # knee_flex_std = standevs['kneeAngleR_exo'][0][0][0][0]
        knee_flex_std = coordDevs_exo['knee_angle_r']
        tempreturnkneeflex = helperOsimFunctions.coordSTDCompare(predvec2_knee, IKvec2_knee, inDegrees, knee_flex_std) #, fig, ax, count)
        stdCompares['knee_angle_r'] = tempreturnkneeflex[0:3]
        # fig = tempreturnkneeflex[3]
        # ax = tempreturnkneeflex[4]
        # ax[count].set_ylabel('knee flexion', fontsize=12)
        count += 1
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # ankle_flex_std = standevs['ankleAngleR_exo'][0][0][0][0]
        ankle_flex_std = coordDevs_exo['ankle_angle_r']
        tempreturnankle = helperOsimFunctions.coordSTDCompare(predvec2_ankle, IKvec2_ankle, inDegrees, ankle_flex_std) #, fig, ax, count)
        stdCompares['ankle_angle_r'] = tempreturnankle[0:3]
        # fig = tempreturnankle[3]
        # ax = tempreturnankle[4]
        # ax[count].set_ylabel('ankle angle', fontsize=12)
        count += 1
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))
        # mtp_flex_std = standevs['mtpAngleR_exo'][0][0][0][0]
        mtp_flex_std = coordDevs_exo['mtp_angle_r']
        tempreturnmtp = helperOsimFunctions.coordSTDCompare(predvec2_mtp, IKvec2_mtp, inDegrees, mtp_flex_std) #, fig, ax, count)
        stdCompares['mtp_angle_r'] = tempreturnmtp[0:3]
        # fig = tempreturnmtp[3]
        # ax = tempreturnmtp[4]
        # ax[count].set_ylabel('mtp angle', fontsize=12)
        count += 1

        # lumbar extension
        predvec_lumbarExt = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy());
        try:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar_extension').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarExt = predvec_lumbarExt.flatten()
        IKvec2_lumbarExt = IKvec_lumbarExt.flatten()
        lumbarExterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees) # 
        # lumbarExt_std = standevs['lumbarExtension_exo'][0][0][0][0]
        lumbarExt_std = coordDevs_exo['lumbar_extension']
        tempreturnlumbarExt = helperOsimFunctions.coordSTDCompare(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees, lumbarExt_std) #, fig, ax, count)
        stdCompares['lumbar_extension'] = tempreturnlumbarExt[0:3]
        # fig = tempreturnlumbarExt[3]
        # ax = tempreturnlumbarExt[4]
        # ax[count].set_ylabel('lumbar extension', fontsize=12)
        count += 1
        # lumbar bending
        predvec_lumbarBend = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy());
        try:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar_bending').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarBend = predvec_lumbarBend.flatten()
        IKvec2_lumbarBend = IKvec_lumbarBend.flatten()
        lumbarBenderr = helperOsimFunctions.dtw_rmse(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees) # 
        # lumbarBend_std = standevs['lumbarBending_exo'][0][0][0][0]
        lumbarBend_std = coordDevs_exo['lumbar_bending']
        tempreturnlumbarBend = helperOsimFunctions.coordSTDCompare(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees, lumbarBend_std) #, fig, ax, count)
        stdCompares['lumbar_bending'] = tempreturnlumbarBend[0:3]
        # fig = tempreturnlumbarBend[3]
        # ax = tempreturnlumbarBend[4]
        # ax[count].set_ylabel('lumbar bending', fontsize=12)
        count += 1
        # lumbar rotation
        predvec_lumbarRot = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy());
        try:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar_rotation').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar').to_numpy()
        predvec2_lumbarRot = predvec_lumbarRot.flatten()
        IKvec2_lumbarRot = IKvec_lumbarRot.flatten()
        lumbarRoterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees) # 
        # lumbarRot_std = standevs['lumbarRotation_exo'][0][0][0][0]
        lumbarRot_std = coordDevs_exo['lumbar_rotation']
        tempreturnlumbarRot = helperOsimFunctions.coordSTDCompare(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees, lumbarRot_std) #, fig, ax, count)
        stdCompares['lumbar_rotation'] = tempreturnlumbarRot[0:3]
        # fig = tempreturnlumbarRot[3]
        # ax = tempreturnlumbarRot[4]
        # ax[count].set_ylabel('lumbar rotation', fontsize=12)
        count += 1

        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        
        # check out the plot
        # plt.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./analysesReports/' + tag + '_kinematics_withinRange.png', dpi=300)

        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr | lumbarExterr lumbarBenderr lumbarRoterr
        kinerr =  (50*vgrferr + 50*hgrferr + 
                hiperr + kneeerr + ankleerr + acromerr + elbowerr + 
                lumbarExterr + lumbarBenderr + lumbarRoterr + 
                listerr + rotationerr + tilterr) # + mtperr + err 
        kinerrExo = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: exotendon %s \n' % tag)
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('hgrferr: %f' % hgrferr)
        print('vgrferr: %f' % vgrferr)
        print('hiperr: %f' % hiperr)
        print('kneeerr: %f' % kneeerr)
        print('ankleerr: %f' % ankleerr)
        print('mtperr: %f' % mtperr)
        print('acromerr: %f' % acromerr)
        print('elbowerr: %f' % elbowerr)

        # print('\nlumbarerr: %f' % lumbarerr)
        print('lumbarExterr: %f' % lumbarExterr)
        print('lumbarBenderr: %f' % lumbarBenderr)
        print('lumbarRoterr: %f' % lumbarRoterr)

        print('pelvis ty err: %f' % tyvalueerr)
        print('pelvis list err: %f' % listerr)
        print('pelvis rotation err: %f' % rotationerr)
        print('pelvis tilt err: %f' % tilterr)

        print('predictiontime: %f' % predictiontime)
        print('IKTime: %f' % gaitIDtime)
        print('err: %f' % err)
        # print\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('met cost: %f' % objectiveValexo)
        print('tot_err: %f \n\n' % kinerrExo)
        # pdb.set_trace()
        for each in stdCompares.keys():
            print(each + ': ' + str(stdCompares[each][0:2]))
            # print(stdCompares[each])
        # print(stdCompares)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nlumbarExterr: %f' % lumbarExterr)
        outlog.write('\nlumbarBenderr: %f' % lumbarBenderr)
        outlog.write('\nlumbarRoterr: %f' % lumbarRoterr)

        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nlisterr: %f' % listerr)
        outlog.write('\npelvrotationerr: %f' % rotationerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\met cost: %f' % objectiveValexo)
        outlog.write('\ntot_err: %f' % kinerrExo)
        for each in stdCompares.keys():
            outlog.write('\n')
            outlog.write(each + ': ' + str(stdCompares[each][0:2]))
            # outlog.write('\n')
            # outlog.write(str(stdCompares[each]))
        outlog.close()
        # print('\nafter file stuff')
        if (objectiveValnat - objectiveValexo) < 0:
            tot_err = 5e8
        else:
            tot_err = kinerrNat + kinerrExo + ((10/(objectiveValnat - objectiveValexo))**2)

    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err



def objective_bilevel_CMATrackTightnat(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # pdb.set_trace()
        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3basetrack(x, 'CMATrack_')
        # pdb.set_trace()
        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('R_ground_force_vy').to_numpy(); # ground_force_r_vy # rF_y
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = 10*helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('R_ground_force_vx').to_numpy(); # ground_force_r_vx # rF_x
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = 10*helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 


        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = 3*helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = 3*helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = 2*helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # lumbar flexion
        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar_extension').to_numpy(); # /jointset/lumbar/lumbar/value # lumbar
        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = 2*helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = 2*helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr
        kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + mtperr + tilterr + err 

        kinerrNat = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: natural\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nmtperr: %f' % mtperr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\nacromerr: %f' % acromerr)
        print('\nelbowerr: %f' % elbowerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\npelvis tilt err: %f' % tilterr)
        print('\npredictiontime: %f' % predictiontime)
        print('\nIKTime: %f' % gaitIDtime)
        print('\nerr: %f' % err)
        # print('\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('\nobjective: %f' % objectiveVal)
        print('\ntot_err: %f' % kinerrNat)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\nobjective: %f' % objectiveVal)
        outlog.write('\ntot_err: %f' % kinerrNat)
        outlog.close()
        # print('\nafter file stuff')

        tot_err = kinerr


        # # now the exotendon version
        # # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3exotrack(x, 'CMATrack_')
        # # pdb.set_trace()
        # # # pdb.set_trace()
        # # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # # print('\ngot the returns')
        # # pdb.set_trace()
        # kinemPredTable = kinemPred.exportToStatesTable()


        # # compute end times for the predicion and the experimental reference
        # predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        # IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        # predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        # gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        # IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # # vertical GRF
        # predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        # IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy(); # ground_force_r_vy
        # predvec2_y = predvec_y.flatten()
        # IDvec2_y = IDvec_y.flatten()
        # vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # # horizontal GRF
        # predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy(); # ground_force_r_vx
        # predvec2_x = predvec_x.flatten()
        # IDvec2_x = IDvec_x.flatten()
        # hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # # get predicted hip, knee, ankle, lumbar, arms
        # print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # # hip flexion
        # predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        # IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        # predvec2_hip = predvec_hip.flatten()
        # IKvec2_hip = IKvec_hip.flatten()
        # hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # # knee flexion 
        # predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        # IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        # predvec2_knee = predvec_knee.flatten()
        # IKvec2_knee = IKvec_knee.flatten()
        # kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # # ankle flexion angle
        # predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        # IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        # predvec2_ankle = predvec_ankle.flatten()
        # IKvec2_ankle = IKvec_ankle.flatten()
        # ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # # mtp flexion angle
        # predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        # IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        # predvec2_mtp = predvec_mtp.flatten()
        # IKvec2_mtp = IKvec_mtp.flatten()
        # mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # # lumbar flexion
        # predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        # IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar').to_numpy(); # /jointset/lumbar/lumbar/value
        # predvec2_lumbar = predvec_lumbar.flatten()
        # IKvec2_lumbar = IKvec_lumbar.flatten()
        # lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # # acromial - shoulder flex 
        # predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        # IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        # predvec2_acrom = predvec_acrom.flatten()
        # IKvec2_acrom = IKvec_acrom.flatten()
        # acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # # elbow flexion
        # predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        # IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        # predvec2_elbow = predvec_elbow.flatten()
        # IKvec2_elbow = IKvec_elbow.flatten()
        # elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # # get the pelvis height 
        # predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        # IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        # predvec2_ty_value = predvec_ty_value.flatten()
        # IKvec2_ty_value = IKvec_ty_value.flatten()
        # tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        
        # # pelvis tilt
        # predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        # IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        # predvec2_tilt_value = predvec_tilt_value.flatten()
        # IKvec2_tilt_value = IKvec_tilt_value.flatten()
        # tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        # #################################
        # # then want cost of total duration for the gait cycle
        # predictiontime_norm = predictiontime / gaitIDtime
        # gaitIDtime_norm = 1.0


        # err = 1e5*abs(predictiontime - gaitIDtime);
        # # conditional to heavily weight times that are very different
        # # if err >= 0.01:
        # #     err = err*1e6
        # # else: 
        # #     err = err*4e4

        # # not including
        # # tyvalueerr

        # # TODO check the grf terms: 
        # kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + lumbarerr + mtperr + tilterr + err 

        # kinerrExo = kinerr
        # # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # # objexp = abs(np.log10(objectiveVal)) - 1
        # # objErr = 100**(abs(objexp))


        # outlog = open(outlogfile, 'a')
        # outlog.write('\n\nInputs: exo\n')
        # outlog.write(str(x))
        # print('\n\ninputs\n')
        # print(x)
        # print('\n')

        # # pdb.set_trace()

        # # print errors to see relative magnitudes
        # print('\nhgrferr: %f' % hgrferr)
        # print('\nvgrferr: %f' % vgrferr)
        # print('\nhiperr: %f' % hiperr)
        # print('\nkneeerr: %f' % kneeerr)
        # print('\nankleerr: %f' % ankleerr)
        # print('\nmtperr: %f' % mtperr)
        # print('\nlumbarerr: %f' % lumbarerr)
        # print('\nacromerr: %f' % acromerr)
        # print('\nelbowerr: %f' % elbowerr)
        # print('\npelvis ty err: %f' % tyvalueerr)
        # print('\npelvis tilt err: %f' % tilterr)
        # print('\npredictiontime: %f' % predictiontime)
        # print('\nIKTime: %f' % gaitIDtime)
        # print('\nerr: %f' % err)
        # # print('\nl1cost: %f' % l1cost)
        # # print('\nobjectivecost: %f' % objErr)
        # # print('\nobjective: %f' % objectiveVal)
        # print('\ntot_err: %f' % kinerrExo)


        # outlog.write('\nhgrferr: %f' % hgrferr)
        # outlog.write('\nvgrferr: %f' % vgrferr)
        # outlog.write('\nhiperr: %f' % hiperr)
        # outlog.write('\nkneeerr: %f' % kneeerr)
        # outlog.write('\nankleerr: %f' % ankleerr)
        # outlog.write('\nmtperr: %f' % mtperr)
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        # outlog.write('\nacromerr: %f' % acromerr)
        # outlog.write('\nelbowerr: %f' % elbowerr)
        # outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        # outlog.write('\npelvistilterr: %f' % tilterr)
        # outlog.write('\npredictiontime: %f' % predictiontime)
        # outlog.write('\nIKTime: %f' % gaitIDtime)
        # outlog.write('\nerr: %f' % err)
        # # outlog.write('\nl1cost: %f' % l1cost)
        # # outlog.write('\nobjectiveErr %f' % objErr)
        # # outlog.write('\nobjective: %f' % objectiveVal)
        # outlog.write('\ntot_err: %f' % kinerrExo)
        # outlog.close()
        # # print('\nafter file stuff')
        # tot_err = kinerrNat + kinerrExo



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

def objective_bilevel_CMATrackTightexo(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3basetrack(x, 'CMATrack_')
        # # pdb.set_trace()
        # # # pdb.set_trace()
        # # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # # print('\ngot the returns')
        # # pdb.set_trace()
        # kinemPredTable = kinemPred.exportToStatesTable()


        # # compute end times for the predicion and the experimental reference
        # predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        # IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        # predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        # gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        # IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # # vertical GRF
        # predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        # IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy(); # ground_force_r_vy
        # predvec2_y = predvec_y.flatten()
        # IDvec2_y = IDvec_y.flatten()
        # vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # # horizontal GRF
        # predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy(); # ground_force_r_vx
        # predvec2_x = predvec_x.flatten()
        # IDvec2_x = IDvec_x.flatten()
        # hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # # get predicted hip, knee, ankle, lumbar, arms
        # print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # # hip flexion
        # predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        # IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        # predvec2_hip = predvec_hip.flatten()
        # IKvec2_hip = IKvec_hip.flatten()
        # hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # # knee flexion 
        # predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        # IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        # predvec2_knee = predvec_knee.flatten()
        # IKvec2_knee = IKvec_knee.flatten()
        # kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # # ankle flexion angle
        # predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        # IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        # predvec2_ankle = predvec_ankle.flatten()
        # IKvec2_ankle = IKvec_ankle.flatten()
        # ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # # mtp flexion angle
        # predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        # IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        # predvec2_mtp = predvec_mtp.flatten()
        # IKvec2_mtp = IKvec_mtp.flatten()
        # mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # # lumbar flexion
        # predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        # IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar').to_numpy(); # /jointset/lumbar/lumbar/value
        # predvec2_lumbar = predvec_lumbar.flatten()
        # IKvec2_lumbar = IKvec_lumbar.flatten()
        # lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # # acromial - shoulder flex 
        # predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        # IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        # predvec2_acrom = predvec_acrom.flatten()
        # IKvec2_acrom = IKvec_acrom.flatten()
        # acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # # elbow flexion
        # predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        # IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        # predvec2_elbow = predvec_elbow.flatten()
        # IKvec2_elbow = IKvec_elbow.flatten()
        # elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # # get the pelvis height 
        # predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        # IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        # predvec2_ty_value = predvec_ty_value.flatten()
        # IKvec2_ty_value = IKvec_ty_value.flatten()
        # tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        
        # # pelvis tilt
        # predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        # IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        # predvec2_tilt_value = predvec_tilt_value.flatten()
        # IKvec2_tilt_value = IKvec_tilt_value.flatten()
        # tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        # #################################
        # # then want cost of total duration for the gait cycle
        # predictiontime_norm = predictiontime / gaitIDtime
        # gaitIDtime_norm = 1.0


        # err = 1e5*abs(predictiontime - gaitIDtime);
        # # conditional to heavily weight times that are very different
        # # if err >= 0.01:
        # #     err = err*1e6
        # # else: 
        # #     err = err*4e4

        # # not including
        # # tyvalueerr

        # # TODO check the grf terms: 
        # kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + lumbarerr + mtperr + tilterr + err 

        # kinerrNat = kinerr
        # # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # # objexp = abs(np.log10(objectiveVal)) - 1
        # # objErr = 100**(abs(objexp))


        # outlog = open(outlogfile, 'a')
        # outlog.write('\n\nInputs: natural\n')
        # outlog.write(str(x))
        # print('\n\ninputs\n')
        # print(x)
        # print('\n')

        # # pdb.set_trace()

        # # print errors to see relative magnitudes
        # print('\nhgrferr: %f' % hgrferr)
        # print('\nvgrferr: %f' % vgrferr)
        # print('\nhiperr: %f' % hiperr)
        # print('\nkneeerr: %f' % kneeerr)
        # print('\nankleerr: %f' % ankleerr)
        # print('\nmtperr: %f' % mtperr)
        # print('\nlumbarerr: %f' % lumbarerr)
        # print('\nacromerr: %f' % acromerr)
        # print('\nelbowerr: %f' % elbowerr)
        # print('\npelvis ty err: %f' % tyvalueerr)
        # print('\npelvis tilt err: %f' % tilterr)
        # print('\npredictiontime: %f' % predictiontime)
        # print('\nIKTime: %f' % gaitIDtime)
        # print('\nerr: %f' % err)
        # # print('\nl1cost: %f' % l1cost)
        # # print('\nobjectivecost: %f' % objErr)
        # # print('\nobjective: %f' % objectiveVal)
        # print('\ntot_err: %f' % kinerrNat)


        # outlog.write('\nhgrferr: %f' % hgrferr)
        # outlog.write('\nvgrferr: %f' % vgrferr)
        # outlog.write('\nhiperr: %f' % hiperr)
        # outlog.write('\nkneeerr: %f' % kneeerr)
        # outlog.write('\nankleerr: %f' % ankleerr)
        # outlog.write('\nmtperr: %f' % mtperr)
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        # outlog.write('\nacromerr: %f' % acromerr)
        # outlog.write('\nelbowerr: %f' % elbowerr)
        # outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        # outlog.write('\npelvistilterr: %f' % tilterr)
        # outlog.write('\npredictiontime: %f' % predictiontime)
        # outlog.write('\nIKTime: %f' % gaitIDtime)
        # outlog.write('\nerr: %f' % err)
        # # outlog.write('\nl1cost: %f' % l1cost)
        # # outlog.write('\nobjectiveErr %f' % objErr)
        # # outlog.write('\nobjective: %f' % objectiveVal)
        # outlog.write('\ntot_err: %f' % kinerrNat)
        # outlog.close()
        # # print('\nafter file stuff')



        # now the exotendon version
        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3exotrack(x, 'CMATrack_')
        # pdb.set_trace()
        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy(); # ground_force_r_vy
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = 10*helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy(); # ground_force_r_vx
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = 10*helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 


        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = 3*helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = 3*helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = 3*helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # lumbar flexion
        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar').to_numpy(); # /jointset/lumbar/lumbar/value
        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = 2*helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = 2*helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr
        kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + mtperr + tilterr + err 

        kinerrExo = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: exo\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nmtperr: %f' % mtperr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\nacromerr: %f' % acromerr)
        print('\nelbowerr: %f' % elbowerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\npelvis tilt err: %f' % tilterr)
        print('\npredictiontime: %f' % predictiontime)
        print('\nIKTime: %f' % gaitIDtime)
        print('\nerr: %f' % err)
        # print('\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('\nobjective: %f' % objectiveVal)
        print('\ntot_err: %f' % kinerrExo)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\nobjective: %f' % objectiveVal)
        outlog.write('\ntot_err: %f' % kinerrExo)
        outlog.close()
        # print('\nafter file stuff')
        
        # tot_err = kinerrNat + kinerrExo
        tot_err = kinerr


    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

###########################################################################
# tracking problems for the manual sweeps - to still output errors. 
def objective_sweep_nat(gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal, x, tag, modelmass, resultspath):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfile = 'NatsweepTrack_logfile.txt'

        # outlogfilename = open('NatsweepTrack_logfile.txt', 'r')
        # outlogfile = outlogfilename.read()
        # outlogfilename.close()


        # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3basetrack(x, 'CMATrack_')
        # pdb.set_trace()
        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # load in the matlab standard devs for the coordinates
        # standevsData = loadmat('coordinates2StandardDeviations.mat')
        # standevs = standevsData['standevs']
        # standevsdf = pd.DataFrame(standevs)
        # grfdevsData = loadmat('standardDevs_ExternalForces.mat')
        # grfdevs = grfdevsData['standevs']

        grfDevs_nat = pd.read_csv('std_externalForces_nat.csv')
        grfDevs_both = pd.read_csv('std_externalForces_both.csv')
        # grfDevs_exo = pd.read_csv('std_externalForces_exo.csv')
        coordDevs_nat = pd.read_csv('std_coords_nat.csv')
        coordDevs_both = pd.read_csv('std_coords_both.csv')
        # coordDevs_exo = pd.read_csv('std_coords_exo.csv')

        stdCompares = {}
        fig, ax = plt.subplots(1,2, figsize=(8,3)) # , dpi=300
        ax = ax.flatten()
        count = 0

        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_y = gaitIDTable.getDependentColumn(tempname[0:-1] + 'y').to_numpy();
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # vgf_std = grfdevs['calcn_r_Right_GRF_Fy_nat'][0][0][0][0]
        vgf_std = grfDevs_nat['calcn_r_Right_GRF_Fy']
        # TODO figure out how to do normalization for the GRF values or un normalize the stddev data. 
        tempreturngrfy = helperOsimFunctions.coordSTDCompare(predvec2_y/(modelmass*9.81), IDvec2_y/(modelmass*9.81), inDegrees, vgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfy[0:3]
        fig = tempreturngrfy[3]
        ax = tempreturngrfy[4]
        ax[count].set_ylabel('GRFy', fontsize=12)
        count += 1


        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_x = gaitIDTable.getDependentColumn(tempname[0:-1] + 'x').to_numpy();
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # hgf_std = grfdevs['calcn_r_Right_GRF_Fx_nat'][0][0][0][0]
        hgf_std = grfDevs_nat['calcn_r_Right_GRF_Fx']
        tempreturngrfx = helperOsimFunctions.coordSTDCompare(predvec2_x/(modelmass*9.81), IDvec2_x/(modelmass*9.81), inDegrees, hgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_x'] = tempreturngrfx[0:3]
        fig = tempreturngrfx[3]
        ax = tempreturngrfx[4]
        ax[count].set_ylabel('GRFx', fontsize=12)
        count += 1


        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        try:
            IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy()
        except:
            IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy() 
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        # get the std from gait data
        # pelvis_ty_std = standevs['pelvisTy_nat'][0][0][0][0]
        pelvis_ty_std = coordDevs_nat['pelvis_ty']
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        tempreturnTy = helperOsimFunctions.coordSTDCompare(predvec2_ty_value, IKvec2_ty_value, inDegrees, pelvis_ty_std) #, fig, ax, count)
        stdCompares['pelvis_ty'] = tempreturnTy[0:3]
        # fig = tempreturnTy[3]
        # ax = tempreturnTy[4]
        # ax[count].set_ylabel('pelvisTy', fontsize=12)
        count += 1

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 

        # pelvis list
        predvec_list_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy());
        try: 
            IKvec_list_value = kinemIKTable.getDependentColumn('pelvis_list').to_numpy()
        except:
            IKvec_list_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
        predvec2_list_value = predvec_list_value.flatten()
        IKvec2_list_value = IKvec_list_value.flatten()
        listerr = helperOsimFunctions.dtw_rmse(predvec2_list_value, IKvec2_list_value, inDegrees)
        # pelvislist_std = standevs['pelvisList_nat'][0][0][0][0]
        pelvislist_std = coordDevs_nat['pelvis_list']
        tempreturnlist = helperOsimFunctions.coordSTDCompare(predvec2_list_value, IKvec2_list_value, inDegrees, pelvislist_std) #, fig, ax, count)
        stdCompares['pelvis_list'] = tempreturnlist[0:3]
        # fig = tempreturnlist[3]
        # ax = tempreturnlist[4]
        # ax[count].set_ylabel('pelvis list', fontsize=12)
        count += 1
        # pelvis rotation
        predvec_rotation_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy());
        try:
            IKvec_rotation_value = kinemIKTable.getDependentColumn('pelvis_rotation').to_numpy(); # /jointset/groundPelvis/pelvis_rotation/value
        except: 
            IKvec_rotation_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
        predvec2_rotation_value = predvec_rotation_value.flatten()
        IKvec2_rotation_value = IKvec_rotation_value.flatten()
        rotationerr = helperOsimFunctions.dtw_rmse(predvec2_rotation_value, IKvec2_rotation_value, inDegrees)
        # pelvisrotation_std = standevs['pelvisRotation_nat'][0][0][0][0]
        pelvisrotation_std = coordDevs_nat['pelvis_rotation']
        tempreturnrotation = helperOsimFunctions.coordSTDCompare(predvec2_rotation_value, IKvec2_rotation_value, inDegrees, pelvisrotation_std) #, fig, ax, count)
        stdCompares['pelvis_rotation'] = tempreturnrotation[0:3]
        # fig = tempreturnrotation[3]
        # ax = tempreturnrotation[4]
        # ax[count].set_ylabel('pelvis rotation', fontsize=12)
        count += 1
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        try:
            IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        except: 
            IKvec_tilt_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)
        # pelvisTilt_std = standevs['pelvisTilt_nat'][0][0][0][0]
        pelvisTilt_std = coordDevs_nat['pelvis_tilt']
        tempreturntilt = helperOsimFunctions.coordSTDCompare(predvec2_tilt_value, IKvec2_tilt_value, inDegrees, pelvisTilt_std) #, fig, ax, count)
        stdCompares['pelvis_tilt'] = tempreturntilt[0:3]
        # fig = tempreturntilt[3]
        # ax = tempreturntilt[4]
        # ax[count].set_ylabel('pelvis tilt', fontsize=12)
        count += 1

        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        try:
            IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy()
        except:
            IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # hip_flex_std = standevs['hipFlexionR_nat'][0][0][0][0]
        hip_flex_std = coordDevs_nat['hip_flexion_r']
        tempreturnhipflex = helperOsimFunctions.coordSTDCompare(predvec2_hip, IKvec2_hip, inDegrees, hip_flex_std) #, fig, ax, count)
        stdCompares['hip_flexion_r'] = tempreturnhipflex[0:3]
        # fig = tempreturnhipflex[3]
        # ax = tempreturnhipflex[4]
        # ax[count].set_ylabel('hip flexion', fontsize=12)
        count += 1
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy());
        try:
            IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy()
        except:
            IKvec_knee = kinemIKTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # knee_flex_std = standevs['kneeAngleR_nat'][0][0][0][0]
        knee_flex_std = coordDevs_nat['knee_angle_r']
        tempreturnkneeflex = helperOsimFunctions.coordSTDCompare(predvec2_knee, IKvec2_knee, inDegrees, knee_flex_std) #, fig, ax, count)
        stdCompares['knee_angle_r'] = tempreturnkneeflex[0:3]
        # fig = tempreturnkneeflex[3]
        # ax = tempreturnkneeflex[4]
        # ax[count].set_ylabel('knee flexion', fontsize=12)
        count += 1
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        try:
            IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy()
        except:
            IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # ankle_flex_std = standevs['ankleAngleR_nat'][0][0][0][0]
        ankle_flex_std = coordDevs_nat['ankle_angle_r']
        tempreturnankle = helperOsimFunctions.coordSTDCompare(predvec2_ankle, IKvec2_ankle, inDegrees, ankle_flex_std) #, fig, ax, count)
        stdCompares['ankle_angle_r'] = tempreturnankle[0:3]
        # fig = tempreturnankle[3]
        # ax = tempreturnankle[4]
        # ax[count].set_ylabel('ankle angle', fontsize=12)
        count += 1
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        try:
            IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        except:
            IKvec_mtp = kinemIKTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy()
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))
        # mtp_flex_std = standevs['mtpAngleR_nat'][0][0][0][0]
        mtp_flex_std = coordDevs_nat['mtp_angle_r']
        tempreturnmtp = helperOsimFunctions.coordSTDCompare(predvec2_mtp, IKvec2_mtp, inDegrees, mtp_flex_std) #, fig, ax, count)
        stdCompares['mtp_angle_r'] = tempreturnmtp[0:3]
        # fig = tempreturnmtp[3]
        # ax = tempreturnmtp[4]
        # ax[count].set_ylabel('mtp angle', fontsize=12)
        count += 1

        # lumbar extension
        predvec_lumbarExt = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy());
        try:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar_extension').to_numpy()
        except:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()
        predvec2_lumbarExt = predvec_lumbarExt.flatten()
        IKvec2_lumbarExt = IKvec_lumbarExt.flatten()
        lumbarExterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees) # 
        # lumbarExt_std = standevs['lumbarExtension_nat'][0][0][0][0]
        lumbarExt_std = coordDevs_nat['lumbar_extension']
        tempreturnlumbarExt = helperOsimFunctions.coordSTDCompare(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees, lumbarExt_std) #, fig, ax, count)
        stdCompares['lumbar_extension'] = tempreturnlumbarExt[0:3]
        # fig = tempreturnlumbarExt[3]
        # ax = tempreturnlumbarExt[4]
        # ax[count].set_ylabel('lumbar extension', fontsize=12)
        count += 1
        # lumbar bending
        predvec_lumbarBend = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy());
        try:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar_bending').to_numpy()
        except:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
        predvec2_lumbarBend = predvec_lumbarBend.flatten()
        IKvec2_lumbarBend = IKvec_lumbarBend.flatten()
        lumbarBenderr = helperOsimFunctions.dtw_rmse(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees) # 
        # lumbarBend_std = standevs['lumbarBending_nat'][0][0][0][0]
        lumbarBend_std = coordDevs_nat['lumbar_bending']
        tempreturnlumbarBend = helperOsimFunctions.coordSTDCompare(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees, lumbarBend_std) #, fig, ax, count)
        stdCompares['lumbar_bending'] = tempreturnlumbarBend[0:3]
        # fig = tempreturnlumbarBend[3]
        # ax = tempreturnlumbarBend[4]
        # ax[count].set_ylabel('lumbar bending', fontsize=12)
        count += 1
        # lumbar rotation
        predvec_lumbarRot = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy());
        try:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar_rotation').to_numpy()
        except:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
        predvec2_lumbarRot = predvec_lumbarRot.flatten()
        IKvec2_lumbarRot = IKvec_lumbarRot.flatten()
        lumbarRoterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees) # 
        # lumbarRot_std = standevs['lumbarRotation_nat'][0][0][0][0]
        lumbarRot_std = coordDevs_nat['lumbar_rotation']
        tempreturnlumbarRot = helperOsimFunctions.coordSTDCompare(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees, lumbarRot_std) #, fig, ax, count)
        stdCompares['lumbar_rotation'] = tempreturnlumbarRot[0:3]
        # fig = tempreturnlumbarRot[3]
        # ax = tempreturnlumbarRot[4]
        # ax[count].set_ylabel('lumbar rotation', fontsize=12)
        count += 1

        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        try:
            IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        except:
            IKvec_acrom = kinemIKTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy()    
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        try:
            IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy()
        except:
            IKvec_elbow = kinemIKTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy()    
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        
        # check out the plot
        # plt.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.tight_layout()
        # plt.show()
        plt.savefig(resultspath + tag + '_kinematics_withinRange.png', dpi=300)



        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr | lumbarExterr lumbarBenderr lumbarRoterr
        kinerr =  (vgrferr + hgrferr + 
                hiperr + kneeerr + ankleerr + acromerr + elbowerr + 
                lumbarExterr + lumbarBenderr + lumbarRoterr + 
                listerr + rotationerr + tilterr) # + mtperr + err 
        kinerrNat = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: natural %s \n' % tag)
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('hgrferr: %f' % hgrferr)
        print('vgrferr: %f' % vgrferr)
        print('hiperr: %f' % hiperr)
        print('kneeerr: %f' % kneeerr)
        print('ankleerr: %f' % ankleerr)
        print('mtperr: %f' % mtperr)
        print('acromerr: %f' % acromerr)
        print('elbowerr: %f' % elbowerr)

        # print('\nlumbarerr: %f' % lumbarerr)
        print('lumbarExterr: %f' % lumbarExterr)
        print('lumbarBenderr: %f' % lumbarBenderr)
        print('lumbarRoterr: %f' % lumbarRoterr)

        print('pelvis ty err: %f' % tyvalueerr)
        print('pelvis list err: %f' % listerr)
        print('pelvis rotation err: %f' % rotationerr)
        print('pelvis tilt err: %f' % tilterr)

        print('predictiontime: %f' % predictiontime)
        print('IKTime: %f' % gaitIDtime)
        print('err: %f' % err)
        # print('\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('objective: %f' % objectiveVal)
        print('tot_err: %f \n\n' % kinerrNat)
        # pdb.set_trace()
        for each in stdCompares.keys():
            print(each + ': ' + str(stdCompares[each][0:2]))
            # print(stdCompares[each])
        # print(stdCompares)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nlumbarExterr: %f' % lumbarExterr)
        outlog.write('\nlumbarBenderr: %f' % lumbarBenderr)
        outlog.write('\nlumbarRoterr: %f' % lumbarRoterr)

        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nlisterr: %f' % listerr)
        outlog.write('\npelvrotationerr: %f' % rotationerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\nobjective: %f' % objectiveVal)
        outlog.write('\ntot_err: %f' % kinerrNat)
        for each in stdCompares.keys():
            outlog.write('\n')
            outlog.write(each + ': ' + str(stdCompares[each][0:2]))
            # outlog.write('\n')
            # outlog.write(str(stdCompares[each]))
        outlog.close()
        # print('\nafter file stuff')

        tot_err = kinerr


        # # now the exotendon version
        # # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3exotrack(x, 'CMATrack_')
        # # pdb.set_trace()
        # # # pdb.set_trace()
        # # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # # print('\ngot the returns')
        # # pdb.set_trace()
        # kinemPredTable = kinemPred.exportToStatesTable()


        # # compute end times for the predicion and the experimental reference
        # predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        # IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        # predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        # gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        # IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # # vertical GRF
        # predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        # IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy(); # ground_force_r_vy
        # predvec2_y = predvec_y.flatten()
        # IDvec2_y = IDvec_y.flatten()
        # vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # # horizontal GRF
        # predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy(); # ground_force_r_vx
        # predvec2_x = predvec_x.flatten()
        # IDvec2_x = IDvec_x.flatten()
        # hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # # get predicted hip, knee, ankle, lumbar, arms
        # print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # # hip flexion
        # predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        # IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        # predvec2_hip = predvec_hip.flatten()
        # IKvec2_hip = IKvec_hip.flatten()
        # hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # # knee flexion 
        # predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        # IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        # predvec2_knee = predvec_knee.flatten()
        # IKvec2_knee = IKvec_knee.flatten()
        # kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # # ankle flexion angle
        # predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        # IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        # predvec2_ankle = predvec_ankle.flatten()
        # IKvec2_ankle = IKvec_ankle.flatten()
        # ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # # mtp flexion angle
        # predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        # IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        # predvec2_mtp = predvec_mtp.flatten()
        # IKvec2_mtp = IKvec_mtp.flatten()
        # mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # # lumbar flexion
        # predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        # IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar').to_numpy(); # /jointset/lumbar/lumbar/value
        # predvec2_lumbar = predvec_lumbar.flatten()
        # IKvec2_lumbar = IKvec_lumbar.flatten()
        # lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # # acromial - shoulder flex 
        # predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        # IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        # predvec2_acrom = predvec_acrom.flatten()
        # IKvec2_acrom = IKvec_acrom.flatten()
        # acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # # elbow flexion
        # predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        # IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        # predvec2_elbow = predvec_elbow.flatten()
        # IKvec2_elbow = IKvec_elbow.flatten()
        # elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # # get the pelvis height 
        # predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        # IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        # predvec2_ty_value = predvec_ty_value.flatten()
        # IKvec2_ty_value = IKvec_ty_value.flatten()
        # tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        
        # # pelvis tilt
        # predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        # IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        # predvec2_tilt_value = predvec_tilt_value.flatten()
        # IKvec2_tilt_value = IKvec_tilt_value.flatten()
        # tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        # #################################
        # # then want cost of total duration for the gait cycle
        # predictiontime_norm = predictiontime / gaitIDtime
        # gaitIDtime_norm = 1.0


        # err = 1e5*abs(predictiontime - gaitIDtime);
        # # conditional to heavily weight times that are very different
        # # if err >= 0.01:
        # #     err = err*1e6
        # # else: 
        # #     err = err*4e4

        # # not including
        # # tyvalueerr

        # # TODO check the grf terms: 
        # kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + lumbarerr + mtperr + tilterr + err 

        # kinerrExo = kinerr
        # # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # # objexp = abs(np.log10(objectiveVal)) - 1
        # # objErr = 100**(abs(objexp))


        # outlog = open(outlogfile, 'a')
        # outlog.write('\n\nInputs: exo\n')
        # outlog.write(str(x))
        # print('\n\ninputs\n')
        # print(x)
        # print('\n')

        # # pdb.set_trace()

        # # print errors to see relative magnitudes
        # print('\nhgrferr: %f' % hgrferr)
        # print('\nvgrferr: %f' % vgrferr)
        # print('\nhiperr: %f' % hiperr)
        # print('\nkneeerr: %f' % kneeerr)
        # print('\nankleerr: %f' % ankleerr)
        # print('\nmtperr: %f' % mtperr)
        # print('\nlumbarerr: %f' % lumbarerr)
        # print('\nacromerr: %f' % acromerr)
        # print('\nelbowerr: %f' % elbowerr)
        # print('\npelvis ty err: %f' % tyvalueerr)
        # print('\npelvis tilt err: %f' % tilterr)
        # print('\npredictiontime: %f' % predictiontime)
        # print('\nIKTime: %f' % gaitIDtime)
        # print('\nerr: %f' % err)
        # # print('\nl1cost: %f' % l1cost)
        # # print('\nobjectivecost: %f' % objErr)
        # # print('\nobjective: %f' % objectiveVal)
        # print('\ntot_err: %f' % kinerrExo)


        # outlog.write('\nhgrferr: %f' % hgrferr)
        # outlog.write('\nvgrferr: %f' % vgrferr)
        # outlog.write('\nhiperr: %f' % hiperr)
        # outlog.write('\nkneeerr: %f' % kneeerr)
        # outlog.write('\nankleerr: %f' % ankleerr)
        # outlog.write('\nmtperr: %f' % mtperr)
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        # outlog.write('\nacromerr: %f' % acromerr)
        # outlog.write('\nelbowerr: %f' % elbowerr)
        # outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        # outlog.write('\npelvistilterr: %f' % tilterr)
        # outlog.write('\npredictiontime: %f' % predictiontime)
        # outlog.write('\nIKTime: %f' % gaitIDtime)
        # outlog.write('\nerr: %f' % err)
        # # outlog.write('\nl1cost: %f' % l1cost)
        # # outlog.write('\nobjectiveErr %f' % objErr)
        # # outlog.write('\nobjective: %f' % objectiveVal)
        # outlog.write('\ntot_err: %f' % kinerrExo)
        # outlog.close()
        # # print('\nafter file stuff')
        # tot_err = kinerrNat + kinerrExo



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfile = 'NatsweepTrack_logfile.txt'
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

# TODO above ^^^^
def objective_sweep_exo(gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal, x, tag, modelmass, resultspath):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfile = 'ExosweepTrack_logfile.txt'

        '''
        # # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3basetrack(x, 'CMATrack_')
        # # pdb.set_trace()
        # # # pdb.set_trace()
        # # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # # print('\ngot the returns')
        # # pdb.set_trace()
        # kinemPredTable = kinemPred.exportToStatesTable()


        # # compute end times for the predicion and the experimental reference
        # predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        # IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        # predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        # gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        # IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # # vertical GRF
        # predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        # IDvec_y = gaitIDTable.getDependentColumn('rF_y').to_numpy(); # ground_force_r_vy
        # predvec2_y = predvec_y.flatten()
        # IDvec2_y = IDvec_y.flatten()
        # vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # # horizontal GRF
        # predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # IDvec_x = gaitIDTable.getDependentColumn('rF_x').to_numpy(); # ground_force_r_vx
        # predvec2_x = predvec_x.flatten()
        # IDvec2_x = IDvec_x.flatten()
        # hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # # get predicted hip, knee, ankle, lumbar, arms
        # print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # # hip flexion
        # predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        # IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        # predvec2_hip = predvec_hip.flatten()
        # IKvec2_hip = IKvec_hip.flatten()
        # hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # # knee flexion 
        # predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        # IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        # predvec2_knee = predvec_knee.flatten()
        # IKvec2_knee = IKvec_knee.flatten()
        # kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # # ankle flexion angle
        # predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        # IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        # predvec2_ankle = predvec_ankle.flatten()
        # IKvec2_ankle = IKvec_ankle.flatten()
        # ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # # mtp flexion angle
        # predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        # IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        # predvec2_mtp = predvec_mtp.flatten()
        # IKvec2_mtp = IKvec_mtp.flatten()
        # mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # # lumbar flexion
        # predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        # IKvec_lumbar = kinemIKTable.getDependentColumn('lumbar').to_numpy(); # /jointset/lumbar/lumbar/value
        # predvec2_lumbar = predvec_lumbar.flatten()
        # IKvec2_lumbar = IKvec_lumbar.flatten()
        # lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # # acromial - shoulder flex 
        # predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        # IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        # predvec2_acrom = predvec_acrom.flatten()
        # IKvec2_acrom = IKvec_acrom.flatten()
        # acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # # elbow flexion
        # predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        # IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        # predvec2_elbow = predvec_elbow.flatten()
        # IKvec2_elbow = IKvec_elbow.flatten()
        # elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # # get the pelvis height 
        # predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        # IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        # predvec2_ty_value = predvec_ty_value.flatten()
        # IKvec2_ty_value = IKvec_ty_value.flatten()
        # tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        
        # # pelvis tilt
        # predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        # IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        # predvec2_tilt_value = predvec_tilt_value.flatten()
        # IKvec2_tilt_value = IKvec_tilt_value.flatten()
        # tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        # #################################
        # # then want cost of total duration for the gait cycle
        # predictiontime_norm = predictiontime / gaitIDtime
        # gaitIDtime_norm = 1.0


        # err = 1e5*abs(predictiontime - gaitIDtime);
        # # conditional to heavily weight times that are very different
        # # if err >= 0.01:
        # #     err = err*1e6
        # # else: 
        # #     err = err*4e4

        # # not including
        # # tyvalueerr

        # # TODO check the grf terms: 
        # kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + lumbarerr + mtperr + tilterr + err 

        # kinerrNat = kinerr
        # # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # # objexp = abs(np.log10(objectiveVal)) - 1
        # # objErr = 100**(abs(objexp))


        # outlog = open(outlogfile, 'a')
        # outlog.write('\n\nInputs: natural\n')
        # outlog.write(str(x))
        # print('\n\ninputs\n')
        # print(x)
        # print('\n')

        # # pdb.set_trace()

        # # print errors to see relative magnitudes
        # print('\nhgrferr: %f' % hgrferr)
        # print('\nvgrferr: %f' % vgrferr)
        # print('\nhiperr: %f' % hiperr)
        # print('\nkneeerr: %f' % kneeerr)
        # print('\nankleerr: %f' % ankleerr)
        # print('\nmtperr: %f' % mtperr)
        # print('\nlumbarerr: %f' % lumbarerr)
        # print('\nacromerr: %f' % acromerr)
        # print('\nelbowerr: %f' % elbowerr)
        # print('\npelvis ty err: %f' % tyvalueerr)
        # print('\npelvis tilt err: %f' % tilterr)
        # print('\npredictiontime: %f' % predictiontime)
        # print('\nIKTime: %f' % gaitIDtime)
        # print('\nerr: %f' % err)
        # # print('\nl1cost: %f' % l1cost)
        # # print('\nobjectivecost: %f' % objErr)
        # # print('\nobjective: %f' % objectiveVal)
        # print('\ntot_err: %f' % kinerrNat)


        # outlog.write('\nhgrferr: %f' % hgrferr)
        # outlog.write('\nvgrferr: %f' % vgrferr)
        # outlog.write('\nhiperr: %f' % hiperr)
        # outlog.write('\nkneeerr: %f' % kneeerr)
        # outlog.write('\nankleerr: %f' % ankleerr)
        # outlog.write('\nmtperr: %f' % mtperr)
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        # outlog.write('\nacromerr: %f' % acromerr)
        # outlog.write('\nelbowerr: %f' % elbowerr)
        # outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        # outlog.write('\npelvistilterr: %f' % tilterr)
        # outlog.write('\npredictiontime: %f' % predictiontime)
        # outlog.write('\nIKTime: %f' % gaitIDtime)
        # outlog.write('\nerr: %f' % err)
        # # outlog.write('\nl1cost: %f' % l1cost)
        # # outlog.write('\nobjectiveErr %f' % objErr)
        # # outlog.write('\nobjective: %f' % objectiveVal)
        # outlog.write('\ntot_err: %f' % kinerrNat)
        # outlog.close()
        # # print('\nafter file stuff')

        '''

        # now the exotendon version
        # this actually runs the optimization 
        # gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_3exotrack(x, 'CMATrack_')
        # pdb.set_trace()
        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')

        # load in the std data for basic exo comparison
        standevsData = loadmat('coordinates2StandardDeviations.mat')
        standevs = standevsData['standevs']
        # standevsdf = pd.DataFrame(standevs)
        # and std dev data for the GRF
        # grfdevsData = loadmat('standardDevs_ExternalForces.mat')
        # grfdevs = grfdevsData['standevs']

        # grfDevs_nat = pd.read_csv('std_externalForces_nat.csv')
        grfDevs_both = pd.read_csv('std_externalForces_both.csv')
        grfDevs_exo = pd.read_csv('std_externalForces_exo.csv')
        # coordDevs_nat = pd.read_csv('std_coords_nat.csv')
        coordDevs_both = pd.read_csv('std_coords_both.csv')
        coordDevs_exo = pd.read_csv('std_coords_exo.csv')

        stdCompares = {}
        fig, ax = plt.subplots(1,2, figsize=(8,3)) # , dpi=300
        ax = ax.flatten()
        count = 0

        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_y = gaitIDTable.getDependentColumn(tempname[0:-1] + 'y').to_numpy();
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # vgf_std = grfdevs['calcn_r_Right_GRF_Fy_exo'][0][0][0][0]
        vgf_std = grfDevs_exo['calcn_r_Right_GRF_Fy']
        tempreturngrfy = helperOsimFunctions.coordSTDCompare(predvec2_y/(modelmass*9.81), IDvec2_y/(modelmass*9.81), inDegrees, vgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_y'] = tempreturngrfy[0:3]
        fig = tempreturngrfy[3]
        ax = tempreturngrfy[4]
        ax[count].set_ylabel('GRFy', fontsize=12)
        count += 1


        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        temp = gaitIDTable.getColumnLabels()
        tempname = temp[0]
        IDvec_x = gaitIDTable.getDependentColumn(tempname[0:-1] + 'x').to_numpy();
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # hgf_std = grfdevs['calcn_r_Right_GRF_Fx_exo'][0][0][0][0]
        hgf_std = grfDevs_exo['calcn_r_Right_GRF_Fx']
        tempreturngrfx = helperOsimFunctions.coordSTDCompare(predvec2_x/(modelmass*9.81), IDvec2_x/(modelmass*9.81), inDegrees, hgf_std/(modelmass*9.81), fig, ax, count)
        stdCompares['grf_x'] = tempreturngrfx[0:3]
        fig = tempreturngrfx[3]
        ax = tempreturngrfx[4]
        ax[count].set_ylabel('GRFx', fontsize=12)
        count += 1


        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        try:
            IKvec_ty_value = kinemIKTable.getDependentColumn('pelvis_ty').to_numpy(); # /jointset/groundPelvis/pelvis_ty/value
        except:
            IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        # get the std from gait data
        # pelvis_ty_std = standevs['pelvisTy_exo'][0][0][0][0]
        pelvis_ty_std = coordDevs_exo['pelvis_ty']
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        tempreturnTy = helperOsimFunctions.coordSTDCompare(predvec2_ty_value, IKvec2_ty_value, inDegrees, pelvis_ty_std) #, fig, ax, count)
        stdCompares['pelvis_ty'] = tempreturnTy[0:3]
        # fig = tempreturnTy[3]
        # ax = tempreturnTy[4]
        # ax[count].set_ylabel('pelvisTy', fontsize=12)
        count += 1

        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 

        # pelvis list
        predvec_list_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy());
        try:
            IKvec_list_value = kinemIKTable.getDependentColumn('pelvis_list').to_numpy(); # /jointset/groundPelvis/pelvis_list/value
        except:
            IKvec_list_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
        predvec2_list_value = predvec_list_value.flatten()
        IKvec2_list_value = IKvec_list_value.flatten()
        listerr = helperOsimFunctions.dtw_rmse(predvec2_list_value, IKvec2_list_value, inDegrees)
        # pelvislist_std = standevs['pelvisList_exo'][0][0][0][0]
        pelvislist_std = coordDevs_exo['pelvis_list']
        tempreturnlist = helperOsimFunctions.coordSTDCompare(predvec2_list_value, IKvec2_list_value, inDegrees, pelvislist_std) #, fig, ax, count)
        stdCompares['pelvis_list'] = tempreturnlist[0:3]
        # fig = tempreturnlist[3]
        # ax = tempreturnlist[4]
        # ax[count].set_ylabel('pelvis list', fontsize=12)
        count += 1
        # pelvis rotation
        predvec_rotation_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy());
        try:
            IKvec_rotation_value = kinemIKTable.getDependentColumn('pelvis_rotation').to_numpy(); # /jointset/groundPelvis/pelvis_rotation/value
        except:
            IKvec_rotation_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()    
        predvec2_rotation_value = predvec_rotation_value.flatten()
        IKvec2_rotation_value = IKvec_rotation_value.flatten()
        rotationerr = helperOsimFunctions.dtw_rmse(predvec2_rotation_value, IKvec2_rotation_value, inDegrees)
        # pelvisrotation_std = standevs['pelvisRotation_exo'][0][0][0][0]
        pelvisrotation_std = coordDevs_exo['pelvis_rotation']
        tempreturnrotation = helperOsimFunctions.coordSTDCompare(predvec2_rotation_value, IKvec2_rotation_value, inDegrees, pelvisrotation_std) #, fig, ax, count)
        stdCompares['pelvis_rotation'] = tempreturnrotation[0:3]
        # fig = tempreturnrotation[3]
        # ax = tempreturnrotation[4]
        # ax[count].set_ylabel('pelvis rotation', fontsize=12)
        count += 1
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        try:
            IKvec_tilt_value = kinemIKTable.getDependentColumn('pelvis_tilt').to_numpy(); # /jointset/groundPelvis/pelvis_tilt/value
        except:
            IKvec_tilt_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)
        # pelvisTilt_std = standevs['pelvisTilt_exo'][0][0][0][0]
        pelvisTilt_std = coordDevs_exo['pelvis_tilt']
        tempreturntilt = helperOsimFunctions.coordSTDCompare(predvec2_tilt_value, IKvec2_tilt_value, inDegrees, pelvisTilt_std) #, fig, ax, count)
        stdCompares['pelvis_tilt'] = tempreturntilt[0:3]
        # fig = tempreturntilt[3]
        # ax = tempreturntilt[4]
        # ax[count].set_ylabel('pelvis tilt', fontsize=12)
        count += 1
        
        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        try:
            IKvec_hip = kinemIKTable.getDependentColumn('hip_flexion_r').to_numpy(); # /jointset/hip_r/hip_flexion_r/value
        except:
            IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # hip_flex_std = standevs['hipFlexionR_exo'][0][0][0][0]
        hip_flex_std = coordDevs_exo['hip_flexion_r']
        tempreturnhipflex = helperOsimFunctions.coordSTDCompare(predvec2_hip, IKvec2_hip, inDegrees, hip_flex_std) #, fig, ax, count)
        stdCompares['hip_flexion_r'] = tempreturnhipflex[0:3]
        # fig = tempreturnhipflex[3]
        # ax = tempreturnhipflex[4]
        # ax[count].set_ylabel('hip flexion', fontsize=12)
        count += 1
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy());
        try:
            IKvec_knee = kinemIKTable.getDependentColumn('knee_angle_r').to_numpy(); # /jointset/knee_r/knee_angle_r/value
        except:
            IKvec_knee = kinemIKTable.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # knee_flex_std = standevs['kneeAngleR_exo'][0][0][0][0]
        knee_flex_std = coordDevs_exo['knee_angle_r']
        tempreturnkneeflex = helperOsimFunctions.coordSTDCompare(predvec2_knee, IKvec2_knee, inDegrees, knee_flex_std) #, fig, ax, count)
        stdCompares['knee_angle_r'] = tempreturnkneeflex[0:3]
        # fig = tempreturnkneeflex[3]
        # ax = tempreturnkneeflex[4]
        # ax[count].set_ylabel('knee flexion', fontsize=12)
        count += 1
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        try:
            IKvec_ankle = kinemIKTable.getDependentColumn('ankle_angle_r').to_numpy(); # /jointset/ankle_r/ankle_angle_r/value
        except:
            IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()    
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # ankle_flex_std = standevs['ankleAngleR_exo'][0][0][0][0]
        ankle_flex_std = coordDevs_exo['ankle_angle_r']
        tempreturnankle = helperOsimFunctions.coordSTDCompare(predvec2_ankle, IKvec2_ankle, inDegrees, ankle_flex_std) #, fig, ax, count)
        stdCompares['ankle_angle_r'] = tempreturnankle[0:3]
        # fig = tempreturnankle[3]
        # ax = tempreturnankle[4]
        # ax[count].set_ylabel('ankle angle', fontsize=12)
        count += 1
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        try:
            IKvec_mtp = kinemIKTable.getDependentColumn('mtp_angle_r').to_numpy(); # /jointset/mtp_r/mtp_angle_r/value
        except:
            IKvec_mtp = kinemIKTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy()    
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))
        # mtp_flex_std = standevs['mtpAngleR_exo'][0][0][0][0]
        mtp_flex_std = coordDevs_exo['mtp_angle_r']
        tempreturnmtp = helperOsimFunctions.coordSTDCompare(predvec2_mtp, IKvec2_mtp, inDegrees, mtp_flex_std) #, fig, ax, count)
        stdCompares['mtp_angle_r'] = tempreturnmtp[0:3]
        # fig = tempreturnmtp[3]
        # ax = tempreturnmtp[4]
        # ax[count].set_ylabel('mtp angle', fontsize=12)
        count += 1

        # lumbar extension
        predvec_lumbarExt = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy());
        try:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('lumbar_extension').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarExt = kinemIKTable.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()
        predvec2_lumbarExt = predvec_lumbarExt.flatten()
        IKvec2_lumbarExt = IKvec_lumbarExt.flatten()
        lumbarExterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees) # 
        # lumbarExt_std = standevs['lumbarExtension_exo'][0][0][0][0]
        lumbarExt_std = coordDevs_exo['lumbar_extension']
        tempreturnlumbarExt = helperOsimFunctions.coordSTDCompare(predvec2_lumbarExt, IKvec2_lumbarExt, inDegrees, lumbarExt_std) #, fig, ax, count)
        stdCompares['lumbar_extension'] = tempreturnlumbarExt[0:3]
        # fig = tempreturnlumbarExt[3]
        # ax = tempreturnlumbarExt[4]
        # ax[count].set_ylabel('lumbar extension', fontsize=12)
        count += 1
        # lumbar bending
        predvec_lumbarBend = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy());
        try:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('lumbar_bending').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarBend = kinemIKTable.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
        predvec2_lumbarBend = predvec_lumbarBend.flatten()
        IKvec2_lumbarBend = IKvec_lumbarBend.flatten()
        lumbarBenderr = helperOsimFunctions.dtw_rmse(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees) # 
        # lumbarBend_std = standevs['lumbarBending_exo'][0][0][0][0]
        lumbarBend_std = coordDevs_exo['lumbar_bending']
        tempreturnlumbarBend = helperOsimFunctions.coordSTDCompare(predvec2_lumbarBend, IKvec2_lumbarBend, inDegrees, lumbarBend_std) #, fig, ax, count)
        stdCompares['lumbar_bending'] = tempreturnlumbarBend[0:3]
        # fig = tempreturnlumbarBend[3]
        # ax = tempreturnlumbarBend[4]
        # ax[count].set_ylabel('lumbar bending', fontsize=12)
        count += 1
        # lumbar rotation
        predvec_lumbarRot = (kinemPredTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy());
        try:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('lumbar_rotation').to_numpy(); # /jointset/lumbar/lumbar/value
        except:
            IKvec_lumbarRot = kinemIKTable.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
        predvec2_lumbarRot = predvec_lumbarRot.flatten()
        IKvec2_lumbarRot = IKvec_lumbarRot.flatten()
        lumbarRoterr = helperOsimFunctions.dtw_rmse(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees) # 
        # lumbarRot_std = standevs['lumbarRotation_exo'][0][0][0][0]
        lumbarRot_std = coordDevs_exo['lumbar_rotation']
        tempreturnlumbarRot = helperOsimFunctions.coordSTDCompare(predvec2_lumbarRot, IKvec2_lumbarRot, inDegrees, lumbarRot_std) #, fig, ax, count)
        stdCompares['lumbar_rotation'] = tempreturnlumbarRot[0:3]
        # fig = tempreturnlumbarRot[3]
        # ax = tempreturnlumbarRot[4]
        # ax[count].set_ylabel('lumbar rotation', fontsize=12)
        count += 1

        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        try:
            IKvec_acrom = kinemIKTable.getDependentColumn('arm_flex_r').to_numpy(); # /jointset/acromial_r/arm_flex_r/value
        except:
            IKvec_acrom = kinemIKTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy()
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        try:
            IKvec_elbow = kinemIKTable.getDependentColumn('elbow_flex_r').to_numpy(); # /jointset/elbow_r/elbow_flex_r/value
        except:
            IKvec_elbow = kinemIKTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy()
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        
        # check out the plot
        # plt.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.tight_layout()
        # plt.show()
        plt.savefig(resultspath + tag + '_kinematics_withinRange.png', dpi=300)

        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: lumbarerr | lumbarExterr lumbarBenderr lumbarRoterr
        kinerr =  (vgrferr + hgrferr + 
                hiperr + kneeerr + ankleerr + acromerr + elbowerr + 
                lumbarExterr + lumbarBenderr + lumbarRoterr + 
                listerr + rotationerr + tilterr) # + mtperr + err 
        kinerrNat = kinerr
        # # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        # objexp = abs(np.log10(objectiveVal)) - 1
        # objErr = 100**(abs(objexp))


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs: natural %s \n' % tag)
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('hgrferr: %f' % hgrferr)
        print('vgrferr: %f' % vgrferr)
        print('hiperr: %f' % hiperr)
        print('kneeerr: %f' % kneeerr)
        print('ankleerr: %f' % ankleerr)
        print('mtperr: %f' % mtperr)
        print('acromerr: %f' % acromerr)
        print('elbowerr: %f' % elbowerr)

        # print('\nlumbarerr: %f' % lumbarerr)
        print('lumbarExterr: %f' % lumbarExterr)
        print('lumbarBenderr: %f' % lumbarBenderr)
        print('lumbarRoterr: %f' % lumbarRoterr)

        print('pelvis ty err: %f' % tyvalueerr)
        print('pelvis list err: %f' % listerr)
        print('pelvis rotation err: %f' % rotationerr)
        print('pelvis tilt err: %f' % tilterr)

        print('predictiontime: %f' % predictiontime)
        print('IKTime: %f' % gaitIDtime)
        print('err: %f' % err)
        # print\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('objective: %f' % objectiveVal)
        print('tot_err: %f \n\n' % kinerrNat)
        # pdb.set_trace()
        for each in stdCompares.keys():
            print(each + ': ' + str(stdCompares[each][0:2]))
            # print(stdCompares[each])
        # print(stdCompares)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        
        # outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nlumbarExterr: %f' % lumbarExterr)
        outlog.write('\nlumbarBenderr: %f' % lumbarBenderr)
        outlog.write('\nlumbarRoterr: %f' % lumbarRoterr)

        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nlisterr: %f' % listerr)
        outlog.write('\npelvrotationerr: %f' % rotationerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nIKTime: %f' % gaitIDtime)
        outlog.write('\nerr: %f' % err)
        # outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\nobjective: %f' % objectiveVal)
        outlog.write('\ntot_err: %f' % kinerrNat)
        for each in stdCompares.keys():
            outlog.write('\n')
            outlog.write(each + ': ' + str(stdCompares[each][0:2]))
            # outlog.write('\n')
            # outlog.write(str(stdCompares[each]))
        outlog.close()
        # print('\nafter file stuff')

        tot_err = kinerr


    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfile = 'ExosweepTrack_logfile.txt'
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

###########################################################################
# needs updated:    working fully predictive, no tracking
###########################################################################
def objective_bilevel_CMA(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results
        

        # get the outlog file
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable = py_overgroundGait2D_bi(x, 'CMA_')

        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')




        # print('\ngot the returns')
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];

        IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # get vectors of the GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('ground_force_r_vy').to_numpy();
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # flatten size
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()

        # get predicted hip, knee, ankle, lumbar
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy();

        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy();

        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy();

        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy();


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy();



        # flatten size
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()

        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()

        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()

        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()


        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()

        ####################################
        # integrate through time for differences

        # # get the DTW RMSE 
        # distance, path = fastdtw(predvec2, IDvec2) #, dist=euclidean)
        # path_x, path_y = zip(*path)
        # aligned_x = np.array([predvec2[p] for p in path_x])
        # aligned_y = np.array([IDvec2[p] for p in path_y])
        # rmse = np.sqrt(np.mean((aligned_x - aligned_y) ** 2))

        # use DTW to get RMSE across time, and divide by experimental mean - dividing by stuff in the cost function now
        # hgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # vgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # hiperr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # kneeerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # ankleerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # tyvalueerr = 20 * helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)


        # print('\n before the dtw')

        # trying again with normalized by inf norm
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x)) # type: ignore
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y)) # type: ignore
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip)) # type: ignore
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee)) # type: ignore
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle)) # type: ignore
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # # type: ignore
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees) # type: ignore

        # print('\nafter dtw')
        # TODO - revisit this as the error function

        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e3*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # TODO add all terms
        tot_err = vgrferr + hgrferr + hiperr + kneeerr + ankleerr + lumbarerr + tyvalueerr + err 

        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\nerr: %f' % err)
        print('\ntot_err: %f' % tot_err)

        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\nerr: %f' % err)
        outlog.write('\ntot_err: %f' % tot_err)
        outlog.close()
        # print('\nafter file stuff')



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

def py_overgroundGait2D_bi(x, tag): #[effortWeight, effortExponent, activationWeight, headWeight, headExponent, implicitAuxWeight]):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % Author: Jon Stingel
    # # % 20230407
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

    import opensim as osim
    import re
    import numpy as np
    import pdb
    import helperOsimFunctions


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
    controlEffortWeight = 1e-5;
    stateTrackingWeight = 1e-2;
    GRFTrackingWeight   = 1e-1;

    constraintTolerance = 1e-3;
    convergeTolerance   = 1e-2;

    stepsize = .03;
    maxiterations = 2000;
    fractionExtraBoundSize = 0.3;
    initialTime = 0.0;
    finalTime = 0.6835/2;

    guess = False;
    wantguess = False;
    guessfile = './tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'


    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('bi_muscle_GaitTracking');


    # % Set the OpenSim Model and give it a name
    # % TreadmillModel ='Running267_TM.osim'; %uigetfile('*.osim'); %This code will work for all three speed conditions, choose the model you want to run
    # TreadmillModel = '2DMuscles_OG_basictoes_spheredown_smallsphere.osim';
    TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere.osim'
    model = osim.Model(TreadmillModel);


    # % Reference data for tracking problem
    tableProcessor = osim.TableProcessor('./expData/2Darms/nat_1_IK.mot');
    tableProcessor.append(osim.TabOpLowPassFilter(6));
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());

    modelProcessor = osim.ModelProcessor(model);
    track.setModel(modelProcessor);
    track.setStatesReference(tableProcessor);
    track.set_states_global_tracking_weight(stateTrackingWeight);
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_apply_tracked_states_to_guess(True);
    track.set_initial_time(0.0);
    track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
    study = track.initialize();
    problem = study.updProblem();


    # % Goals
    # % =====

    # % # % # % set specific weights for the individual weight set
    # % coordinateweights = MocoWeightSet();
    # % coordinateweights.cloneAndAppend(MocoWeight("lumbar", 10000000))
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tx", 1000000));
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_ty", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tz", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_list", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_rotation", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tilt", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_l", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_l", 0));
    # % track.set_states_weight_set(coordinateweights);


    # % # % Set different tracking weights for states (weights for states not 
    # % # % explicitly set here have a default value of 1.0). The values below
    # % # % were obtained by trial and error.
    # stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
    # stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1000.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 100.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 100.0);
    # stateTrackingGoal.setWeightForState('/jointset/lumbar/lumbar/value', 10000.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 10.0);



    # % Symmetry (to permit simulating only one step)
    symmetryGoal = osim.MocoPeriodicityGoal('symmetryGoal');
    problem.addGoal(symmetryGoal);
    model = modelProcessor.process();
    model.initSystem();

    # % Symmetric coordinate values (except for pelvis_tx) and speeds
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.startswith(currentStateName , '/jointset'):
            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                               re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                                re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
            
            if ('_r' not in currentStateName and 
                '_l' not in currentStateName and 
                'Treadmill_tx/value' not in currentStateName and 
                'Treadmill_tx/speed' not in currentStateName and 
                'Compensate_tx/value' not in currentStateName and 
                'Compensate_tx/speed' not in currentStateName and 
                '/activation' not in currentStateName):
                # % ~contains(currentStateName,'pelvis_tx/value') and 
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));


    # % Symmetric muscle activations
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):
            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);


    # % Symmetric coordinate actuator controls
    symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/lumbarAct'));

    # % Get a reference to the MocoControlGoal that is added to every MocoTrack
    # % problem by default and change the weight
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
    effort.setWeight(controlEffortWeight);

    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(0.0);
    speedGoal.setMode('endpoint_constraint');
    # % problem.addGoal(speedGoal);


    # % Optionally, add a contact tracking goal.
    if GRFTrackingWeight != 0:
        # % Track the right and left vertical and fore-aft ground reaction forces.
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight);
        contactTracking.setExternalLoadsFile('grf_walk - Copy.xml');
        
        forceNamesRightFoot = osim.StdVectorString();
        forceNamesRightFoot.append('/contactHeel_r');
        # forceNamesRightFoot.append('/forceset/contactLateralRearfoot_r');
        forceNamesRightFoot.append('/contactLateralMidfoot_r');
        forceNamesRightFoot.append('/contactLateralToe_r');
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
        forceNamesLeftFoot.append('/contactLateralToe_l');
        forceNamesLeftFoot.append('/contactMedialToe_l');
        forceNamesLeftFoot.append('/contactMedialMidfoot_l');
        # contactTracking.addContactGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft = osim.MocoContactTrackingGoalGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft.append_alternative_frame_paths('/bodyset/toes_l')
        contactTracking.addContactGroup(contactTrackingSplitLeft);


        contactTracking.setProjection('plane');
        contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));
        problem.addGoal(contactTracking);



    # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, finalTime, guessfile)
    # problem.setTimeBounds(0, [0.25, 0.35]);

    '''
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [1.2, 1.3]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);
    problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/knee_l/knee_angle_l/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);
    '''

    # % Configure the solver
    # % ====================
    solver = study.initCasADiSolver();
    solver.set_optim_finite_difference_scheme('forward')
    solver.set_parameters_require_initsystem(False);
    duration = track.get_final_time() - track.get_initial_time();
    num_mesh = round(duration/stepsize);
    solver.set_num_mesh_intervals(num_mesh);
    solver.set_verbosity(2);
    solver.set_optim_solver('ipopt');
    solver.set_optim_convergence_tolerance(convergeTolerance);
    solver.set_optim_constraint_tolerance(constraintTolerance);
    solver.set_optim_max_iterations(maxiterations);
    solver.set_scale_variables_using_bounds(True);
    solver.set_minimize_implicit_auxiliary_derivatives(True);
    solver.set_implicit_auxiliary_derivatives_weight(1e-6);

    if guess:
        solver.setGuess(osim.MocoTrajectory(guessfile));


    '''
    # %{%
    # % Solve the problem
    # % =================
    gaitTrackingSolution = study.solve();
    gaitTrackingSolution.write('3_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard



    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    addPatterns = osim.StdVectorString();
    addPatterns.append('.*pelvis_tx/value');
    addPatterns.append('.*Treadmill_tx/value');
    addPatterns.append('.*Compensate_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution, addPatterns);
    fullStride.write('3_muscles_Tracking_solution_FullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);


    # % Extract ground reaction forces
    # % ==============================
    contact_r = osim.StdVectorString();
    contact_l = osim.StdVectorString();
    contact_r.append('contactHeel_r');
    contact_r.append('contactFront_r');
    contact_l.append('contactHeel_l');
    contact_l.append('contactFront_l');

    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, 
                                     fullStride,contact_r,contact_l);
    osim.STOFileAdapter.write(externalForcesTableFlat, 
                                 '3_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write('3_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitTrackingSolution to fullstrideGRF:  3_muscles_Tracking_solution_fullStride_wGRF.sto\n\n')


    helperOsimFunctions.syncDrives(localDir, destDir)
    pdb.set_trace()



    keyboard
    if wantguess
        gaitTrackingSolution.write('3_muscles_tracking_guess.sto');
        wantguess = False;
    end
    syncDrives(localDir, destDir);

    # % Uncomment next line to terminate after solving only the tracking problem
    # % return;


    keyboard
    %}
    '''

    gaitTrackingSolutionFile = './tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    gaitTrackingSolution = osim.MocoTrajectory(gaitTrackingSolutionFile);



    # % ------------------------------------------------------------------------
    # % Set up a gait prediction problem where the goal is to minimize effort
    # % (squared controls) divided by distance traveled while enforcing symmetry of
    # % the walking cycle and a prescribed average gait speed through endpoint
    # % constraints. The solution of the coordinate tracking problem is
    # % used as an initial guess for the prediction.
    # import org.opensim.modeling.*;

    # % Define the optimal control problem
    # % ==================================
    study = osim.MocoStudy();
    study.setName('bi_muscle_gaitPrediction');


    print('\n\nNeed to get something that doesnt cut it short, and actually flexes the legs a little. \n')

    # excitation effort
    effortWeight = 1e1 #x[0]; # 1e2;
    effortExponent = 3; # 6; # 3 was working 

    # activations squared 
    activationWeighttemp =1e-6 # x[2]; # 1e-6;  ## filler bc not using at the moment
    activationWeight = 10**(-activationWeighttemp)
    activationWeightEach = 1e0;

    # implicit ??
    implicitAuxWeight = 1e-6;

    # GRF magnitudes - set these to the same weight
    heelForceWeighttemp = x[0] # 1e-20;
    heelForceWeight = 10**(-heelForceWeighttemp)
    heelForceExponent = 3;
    toeForceWeighttemp = x[1] # heelForceWeight # 1e-20;
    toeForceWeight = 10**(-toeForceWeighttemp)
    toeForceExponent = 3;

    # heel acc 
    # heelAccWeighttemp = x[0] # 1e-2;
    # heelAccWeight = 10**(-heelAccWeighttemp)
    # heelAccExponent = 2;

    # head acceleration
    headWeighttemp = x[2]; # 1e-2; # 1e-2 for now - seems okay on the ty motion
    headWeight = 10**(-headWeighttemp)
    headExponent = 2; # 2;
    

    # seems like this approach works, need to find the right termss
    # joint accelerations. ???


    # other params
    convergenceTolerance = 1e-2;
    constraintTolerance = 1e-3;

    stepsize = 0.02;
    maxiterations = 2500;
    fractionExtraBoundSize = 0.2;
    initialTime = 0.0;
    finalTime = 0.6835 / 2;

    problem = study.updProblem();
    modelProcessor = osim.ModelProcessor(model);
    problem.setModelProcessor(modelProcessor);


    # % Goals
    # % =====

    # activations squared goal
    activationGoal = osim.MocoSumSquaredStateGoal('activationGoal');
    activationGoal.setWeight(activationWeight)
    activationGoal.setDivideByDisplacement(True)
    # problem.addGoal(activationGoal);

    # % Symmetry (to permit simulating only one step)
    symmetryGoal = osim.MocoPeriodicityGoal('symmetryGoal');
    problem.addGoal(symmetryGoal);

    # init model
    model = modelProcessor.process();
    model.initSystem();

    # % Symmetric coordinate values (except for pelvis_tx) and speeds
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.startswith(currentStateName , '/jointset'):
            print(currentStateName)
            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                               re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                                re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
            
            if ('_r' not in currentStateName and 
                '_l' not in currentStateName and 
                'Treadmill_tx/value' not in currentStateName and 
                'Treadmill_tx/speed' not in currentStateName and 
                'Compensate_tx/value' not in currentStateName and 
                'Compensate_tx/speed' not in currentStateName and
                'pelvis_tx/value' not in currentStateName and
                'pelvis_tx/speed' not in currentStateName and
                '/activation' not in currentStateName):
                # % ~contains(currentStateName,'pelvis_tx/value') and 
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));


    # % Symmetric muscle activations & activation squared goal
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):

            # activations squared add to goal
            activationGoal.setWeightForState(currentStateName, activationWeightEach)

            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);


    # % Symmetric coordinate actuator controls
    symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/lumbarAct'));


    # % Prescribed average gait speed
    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(2.67);
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);

    # % Effort over distance
    effortGoal = osim.MocoControlGoal('effort', effortWeight); # %10 stock
    effortGoal.setDivideByDisplacement(True)  # not for treadmill 
    effortGoal.setExponent(effortExponent);
    problem.addGoal(effortGoal);

    
    if 'jointWeight' in locals():
        ankleLoadGoal_r = osim.MocoJointReactionGoal('anklejointload_r')
        ankleLoadGoal_r.setJointPath('/jointset/ankle_r')
        ankleLoadGoal_r.setLoadsFrame('child')
        ankleLoadGoal_r.setExpressedInFramePath('/bodyset/calcn_r')
        jointReaction_r = osim.StdVectorString();
        jointReaction_r.append('force-y');
        ankleLoadGoal_r.setReactionMeasures(jointReaction_r)
        ankleLoadGoal_r.setWeight('force-y', ankleLoadWeight) # type: ignore

        ankleLoadGoal_l = osim.MocoJointReactionGoal('anklejointload_l')
        ankleLoadGoal_l.setJointPath('/jointset/ankle_l')
        ankleLoadGoal_l.setLoadsFrame('child')
        ankleLoadGoal_l.setExpressedInFramePath('/bodyset/calcn_l')
        jointReaction_l = osim.StdVectorString();
        jointReaction_l.append('force-y');
        ankleLoadGoal_l.setReactionMeasures(jointReaction_l)
        ankleLoadGoal_l.setWeight('force-y', ankleLoadWeight) # type: ignore

        kneeLoadGoal_r = osim.MocoJointReactionGoal('kneejointload_r')
        kneeLoadGoal_r.setJointPath('/jointset/knee_r')
        kneeLoadGoal_r.setLoadsFrame('child')
        kneeLoadGoal_r.setExpressedInFramePath('/bodyset/tibia_r')
        jointReaction_r = osim.StdVectorString();
        jointReaction_r.append('force-y');
        kneeLoadGoal_r.setReactionMeasures(jointReaction_r)
        kneeLoadGoal_r.setWeight('force-y', kneeLoadWeight) # type: ignore

        kneeLoadGoal_l = osim.MocoJointReactionGoal('kneejointload_l')
        kneeLoadGoal_l.setJointPath('/jointset/knee_l')
        kneeLoadGoal_l.setLoadsFrame('child')
        kneeLoadGoal_l.setExpressedInFramePath('/bodyset/tibia_l')
        jointReaction_l = osim.StdVectorString();
        jointReaction_l.append('force-y');
        kneeLoadGoal_l.setReactionMeasures(jointReaction_l)
        kneeLoadGoal_l.setWeight('force-y', kneeLoadWeight) # type: ignore

        problem.addGoal(ankleLoadGoal_r)
        problem.addGoal(ankleLoadGoal_l)
        problem.addGoal(kneeLoadGoal_r)
        problem.addGoal(kneeLoadGoal_l)


    # % head accelerations
    if 'headWeight' in locals():        
        headGoal = osim.MocoOutputGoal('headacc');
        headGoal.setOutputPath('bodyset/torso/head|acceleration');
        headGoal.setExponent(headExponent);
        headGoal.setWeight(headWeight);
        problem.addGoal(headGoal);

    if 'heelForceWeight' in locals():
        # trying a contact force goal to not slam feet - taking out the left foot, I think at really high weights, the numbers shrink and make the cost function weird.
        # heelForceGoal_l = osim.MocoOutputGoal('heelforce_l');
        # heelForceGoal_l.setOutputPath('contactHeel_l|sphere_force');
        # heelForceGoal_l.setExponent(heelForceExponent);
        # heelForceGoal_l.setWeight(heelForceWeight);

        heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
        heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
        heelForceGoal_r.setExponent(heelForceExponent);
        heelForceGoal_r.setWeight(heelForceWeight);

        # lateralmidfootforcegoal_l = osim.MocoOutputGoal('lateralmidfoot_l');
        # lateralmidfootforcegoal_l.setOutputPath('contactLateralMidfoot_l|sphere_force');
        # lateralmidfootforcegoal_l.setExponent(toeForceExponent);
        # lateralmidfootforcegoal_l.setWeight(toeForceWeight);

        lateralmidfootforcegoal_r = osim.MocoOutputGoal('lateralmidfoot_r');
        lateralmidfootforcegoal_r.setOutputPath('contactLateralMidfoot_r|sphere_force');
        lateralmidfootforcegoal_r.setExponent(toeForceExponent);
        lateralmidfootforcegoal_r.setWeight(toeForceWeight);

        # medialmidfootforcegoal_l = osim.MocoOutputGoal('medialmidfoot_l');
        # medialmidfootforcegoal_l.setOutputPath('contactMedialMidfoot_l|sphere_force');
        # medialmidfootforcegoal_l.setExponent(toeForceExponent);
        # medialmidfootforcegoal_l.setWeight(toeForceWeight);

        medialmidfootforcegoal_r = osim.MocoOutputGoal('medialmidfoot_r');
        medialmidfootforcegoal_r.setOutputPath('contactMedialMidfoot_r|sphere_force');
        medialmidfootforcegoal_r.setExponent(toeForceExponent);
        medialmidfootforcegoal_r.setWeight(toeForceWeight);

        # medialtoeforcegoal_l = osim.MocoOutputGoal('medialtoe_l');
        # medialtoeforcegoal_l.setOutputPath('contactMedialToe_l|sphere_force');
        # medialtoeforcegoal_l.setExponent(toeForceExponent);
        # medialtoeforcegoal_l.setWeight(toeForceWeight);

        medialtoeforcegoal_r = osim.MocoOutputGoal('medialtoe_r');
        medialtoeforcegoal_r.setOutputPath('contactMedialToe_r|sphere_force');
        medialtoeforcegoal_r.setExponent(toeForceExponent);
        medialtoeforcegoal_r.setWeight(toeForceWeight);

        # lateraltoeforcegoal_l = osim.MocoOutputGoal('lateraltoe_l');
        # lateraltoeforcegoal_l.setOutputPath('contactLateralToe_l|sphere_force');
        # lateraltoeforcegoal_l.setExponent(toeForceExponent);
        # lateraltoeforcegoal_l.setWeight(toeForceWeight);

        lateraltoeforcegoal_r = osim.MocoOutputGoal('lateraltoe_r');
        lateraltoeforcegoal_r.setOutputPath('contactLateralToe_r|sphere_force');
        lateraltoeforcegoal_r.setExponent(toeForceExponent);
        lateraltoeforcegoal_r.setWeight(toeForceWeight);

        # problem.addGoal(heelForceGoal_l);
        problem.addGoal(heelForceGoal_r);
        # problem.addGoal(lateralmidfootforcegoal_l);
        problem.addGoal(lateralmidfootforcegoal_r);
        # problem.addGoal(medialmidfootforcegoal_l);
        problem.addGoal(medialmidfootforcegoal_r);
        # problem.addGoal(medialtoeforcegoal_l);
        problem.addGoal(medialtoeforcegoal_r);
        # problem.addGoal(lateraltoeforcegoal_l);
        problem.addGoal(lateraltoeforcegoal_r);


    if 'heelAccWeight' in locals():
        # now trying the heel/ foot accelerations cost
        # try heel acclerations
        heelGoalr = osim.MocoOutputGoal('heelracc');
        heelGoalr.setOutputPath('bodyset/calcn_r/heelr|acceleration');
        heelGoalr.setExponent(heelAccExponent); # type: ignore
        heelGoalr.setWeight(heelAccWeight); # type: ignore

        heelGoall = osim.MocoOutputGoal('heellacc');
        heelGoall.setOutputPath('bodyset/calcn_l/heell|acceleration');
        heelGoall.setExponent(heelAccExponent); # type: ignore
        heelGoall.setWeight(heelAccWeight); # type: ignore

    if 'toeAccWeight' in locals():
        # now toe frame accelerations
        toeGoalr = osim.MocoOutputGoal('toeracc');
        toeGoalr.setOutputPath('bodyset/toes_r/toer|acceleration');
        toeGoalr.setExponent(toeAccExponent); # type: ignore
        toeGoalr.setWeight(toeAccWeight); # type: ignore

        toeGoall = osim.MocoOutputGoal('toelacc');
        toeGoall.setOutputPath('bodyset/toes_l/toel|acceleration');
        toeGoall.setExponent(toeAccExponent); # type: ignore
        toeGoall.setWeight(toeAccWeight); # type: ignore

        problem.addGoal(heelGoalr)
        problem.addGoal(heelGoall)
        problem.addGoal(toeGoalr)
        problem.addGoal(toeGoall)

    # % Bounds
    # % ======
    helperOsimFunctions.constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, finalTime, gaitTrackingSolutionFile);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/speed', [0, 5]);
    problem.setTimeBounds(0, [0.30, 0.35]);

    '''
    problem.setTimeBounds(0, [0.33, 0.35]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [1.2, 1.3]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);
    problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/knee_l/knee_angle_l/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);
    '''

    # % Configure the solver
    # % ====================
    solver = study.initCasADiSolver();
    solver.set_optim_finite_difference_scheme('forward')
    solver.set_parameters_require_initsystem(False);
    duration = track.get_final_time() - track.get_initial_time();
    num_mesh = round(duration/stepsize);
    solver.set_num_mesh_intervals(num_mesh);
    solver.set_verbosity(2);
    solver.set_optim_solver('ipopt');
    solver.set_optim_convergence_tolerance(convergeTolerance);
    solver.set_optim_constraint_tolerance(constraintTolerance);
    solver.set_optim_max_iterations(maxiterations);
    solver.set_scale_variables_using_bounds(True);
    solver.set_minimize_implicit_auxiliary_derivatives(True);
    solver.set_implicit_auxiliary_derivatives_weight(implicitAuxWeight);

    solver.setGuess(gaitTrackingSolution); # % Use tracking solution as initial guess


    # % Solve problem
    # % =============
    gaitPredictionSolution = study.solve();
    gaitPredictionSolution.write(tag + 'bi_OG_muscles_Predicted_solution__Halfgaitcycle.sto');

    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    addPatterns = osim.StdVectorString();
    addPatterns.append('.*pelvis_tx/value');
    # addPatterns.append('.*Treadmill_tx/value');
    # addPatterns.append('.*Compensate_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitPredictionSolution, addPatterns);
    fullStride.write(tag + 'bi_OG_muscles_Predicted_solution_fullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);

    # % Extract ground reaction forces
    # % ==============================
    contact_r = osim.StdVectorString();
    contact_l = osim.StdVectorString();
    contact_r.append('/contactHeel_r');
    # contact_r.append('/forceset/contactLateralRearfoot_r');
    contact_r.append('/contactLateralMidfoot_r');
    contact_r.append('/contactLateralToe_r');
    contact_r.append('/contactMedialToe_r');
    contact_r.append('/contactMedialMidfoot_r');

    contact_l.append('/contactHeel_l');
    # contact_l.append('/forceset/contactLateralRearfoot_l');
    contact_l.append('/contactLateralMidfoot_l');
    contact_l.append('/contactLateralToe_l');
    contact_l.append('/contactMedialToe_l');
    contact_l.append('/contactMedialMidfoot_l');

    # % Create a conventional ground reaction forces file by summing the contact
    # % forces of contact spheres on each foot.
    # % For details, view the Doxygen documentation for
    # % createExternalLoadsTableForGait().
    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, 
                                 fullStride, contact_r, contact_l);
    osim.STOFileAdapter.write(externalForcesTableFlat, 
                                 tag + 'bi_OG_muscles_Predicted_solutionGRF_fullStride.sto');

    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write(tag + 'bi_OG_muscles_predicted_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitPredictionSolution to fullstrideGRF:  bi_OG_muscles_predicted_solution_fullStride_wGRF.sto\n\n')


    # helperOsimFunctions.syncDrives(localDir, destDir)
    # pdb.set_trace()

    # return osim.TimeSeriesTable(tag + 'bi_OG_muscles_Predicted_solutionGRF_fullStride.sto'), \
    # osim.TimeSeriesTable('./expData/nat_1_GRF.mot'), \
    # osim.TimeSeriesTable(tag + 'bi_OG_muscles_Predicted_solution_fullStride.sto'), \
    # osim.TimeSeriesTable('./expData/nat_1_2D_IK.mot')
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \
    osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')

###########################################################################
# basic inverse optimal control prob  x2
###########################################################################
def objective_bilevel_CMATrack(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_bitrack(x, 'CMATrack_')
        # pdb.set_trace()
        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')


        # print('\ngot the returns')
        # pdb.set_trace()
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];
        IKtime = np.asarray(kinemIKTable.getIndependentColumn())

        inDegrees = False

        # vertical GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('ground_force_r_vy').to_numpy();
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))

        # horizontal GRF
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('ground_force_r_vx').to_numpy();
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        

        # get predicted hip, knee, ankle, lumbar, arms
        print('\n\nCAUTION: Need to consider adding in the contralateral side here as well - might help regularize it')

        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy();
        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)
        


        # check if the IK table is degrees or rad
        ifDegreesstr = kinemIKTable.getTableMetaDataAsString('inDegrees')
        if ifDegreesstr == 'yes': 
            inDegrees = True
        else:
            inDegrees = False 


        # hip flexion
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy();
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        
        # knee flexion 
        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy();
        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        
        # ankle flexion angle
        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy();
        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        
        # mtp flexion angle
        predvec_mtp = (kinemPredTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy());
        IKvec_mtp = kinemIKTable.getDependentColumn('/jointset/mtp_r/mtp_angle_r/value').to_numpy();
        predvec2_mtp = predvec_mtp.flatten()
        IKvec2_mtp = IKvec_mtp.flatten()
        mtperr = helperOsimFunctions.dtw_rmse(predvec2_mtp, IKvec2_mtp, inDegrees) # / abs(np.mean(IKvec2_mtp))

        # lumbar flexion
        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy();
        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) # 
        
        # acromial - shoulder flex 
        predvec_acrom = (kinemPredTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy());
        IKvec_acrom = kinemIKTable.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy();
        predvec2_acrom = predvec_acrom.flatten()
        IKvec2_acrom = IKvec_acrom.flatten()
        acromerr = helperOsimFunctions.dtw_rmse(predvec2_acrom, IKvec2_acrom, inDegrees)
        
        # elbow flexion
        predvec_elbow = (kinemPredTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy());
        IKvec_elbow = kinemIKTable.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy();
        predvec2_elbow = predvec_elbow.flatten()
        IKvec2_elbow = IKvec_elbow.flatten()
        elbowerr = helperOsimFunctions.dtw_rmse(predvec2_elbow, IKvec2_elbow, inDegrees)
        
        # pelvis tilt
        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy();
        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees)


        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # not including
        # tyvalueerr

        # TODO check the grf terms: 
        kinerr =  vgrferr + hgrferr + hiperr + kneeerr + ankleerr + acromerr + elbowerr + lumbarerr + mtperr + tilterr + err 

        # get the weights L1 regularization cost
        l1cost = 1e6 * np.sum(np.abs(10**(-x[0]))) # just 0 and 1 idx since they are the tracking weights - others can be whatever, since we want them to fill the void

        # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        objexp = abs(np.log10(objectiveVal)) - 1
        objErr = 100**(abs(objexp))
        

        # do some weights error to try and get the tracking weights lower. 
        tot_err = kinerr + l1cost # + objErr


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nmtperr: %f' % mtperr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\nacromerr: %f' % acromerr)
        print('\nelbowerr: %f' % elbowerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\npelvis tilt err: %f' % tilterr)
        print('\npredictiontime: %f' % predictiontime)
        print('\nerr: %f' % err)
        print('\nl1cost: %f' % l1cost)
        # print('\nobjectivecost: %f' % objErr)
        print('\nobjective: %f' % objectiveVal)
        print('\ntot_err: %f' % tot_err)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nmtperr: %f' % mtperr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\nacromerr: %f' % acromerr)
        outlog.write('\nelbowerr: %f' % elbowerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\npelvistilterr: %f' % tilterr)
        outlog.write('\npredictiontime: %f' % predictiontime)
        outlog.write('\nerr: %f' % err)
        outlog.write('\nl1cost: %f' % l1cost)
        # outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\nobjective: %f' % objectiveVal)
        outlog.write('\ntot_err: %f' % tot_err)
        outlog.close()
        # print('\nafter file stuff')



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

def py_overgroundGait2D_bitrack(x, tag): #[effortWeight, effortExponent, activationWeight, headWeight, headExponent, implicitAuxWeight]):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % this function will be a tracking problem used in CMA to reduce tracking terms
    # # % while also figuring out what other terms need to be included to maintain good
    # # % kinematics that are representative of experimental (known)
    # # % Author: Jon Stingel
    # # % 20230720
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

    import opensim as osim
    import re
    import numpy as np
    import pdb
    import helperOsimFunctions


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
    # % controlEffortWeight = 1e-8;
    # % stateTrackingWeight = 5e-1;
    # % GRFTrackingWeight   = 1e-1;

    # % # % these are the good weights
    # % controlEffortWeight = 1e-1;
    # % stateTrackingWeight = 1e-1;
    # % GRFTrackingWeight   = 1e-1;

    # stateTrackingWeight = 1e-1;
    # GRFTrackingWeight   = 1e0;
    
    # tracking weights
    wanttracking = True
    if wanttracking:
        trackingWeightTemp = x[0]
        stateTrackingWeight = 10**(-trackingWeightTemp) 
        GRFTrackingWeight = 10**(-trackingWeightTemp+1)
    else:
        stateTrackingWeight = 0
        GRFTrackingWeight = 0


    # excitations weight
    controlEffortWeight = 10**(-6) # 1e-3; # 1e-1 # x[1]
    effortExponent = 2


    # activations weights
    activationWeight = 10**(0) # 10**(-x[1]) # 1e0
    activationWeightEach = 1e0

    # metabolic cost term
    metabolicsWeight = 10**(-x[1])


    implicitAuxWeight = 1e-6;
    # now to add all the other weights from the other problems that we are going to include

    # GRF magnitudes - set these to the same weight
    heelForceWeighttemp = x[2] # 1e-20;
    heelForceWeight = 10**(-heelForceWeighttemp)
    heelForceExponent = 3;
    toeForceWeighttemp = x[2] # heelForceWeight # 1e-20;
    toeForceWeight = 10**(-toeForceWeighttemp)
    toeForceExponent = 3;

    # heel acc 
    # heelAccWeighttemp = x[0] # 1e-2;
    # heelAccWeight = 10**(-heelAccWeighttemp)
    # heelAccExponent = 2;

    # head acceleration
    headWeighttemp = x[3]; # 1e-2; # 1e-2 for now - seems okay on the ty motion
    headWeight = 10**(-headWeighttemp)
    headExponent = 2; # 2;
    
    # pelvis acceleration
    pelvisWeight = 10**(-x[4])
    pelvisExponent = 2


    # editing the contact spheres
    # newstifftemp = x[3]; #3067776
    # newstiff = 10**(newstifftemp)

    convergeTolerance   = 1e-2;
    constraintTolerance = 1e-3;

    stepsize = .03;
    maxiterations = 2500;
    finalTime = 0.6835 / 2;

    guess = True;
    wantguess = False;
    # guessfile = 'goodOverground_muscle_tracking_guess.sto';
    # guessfile = 'guess_2D3DsphereArms_half.sto';
    # guessfile = 'good_guess_2D3DsphereArms_half.sto';

    # good tracking guess no arms
    # guessfile = './tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'
    # good tracking guess 2D arms
    guessfile = './tracking2D2DArms_activation/3_OG_muscles_Tracking_solution__Halfgaitcycle.sto' ##############################

    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('CMATrack_muscle_GaitTracking');


    # % Set the OpenSim Model and give it a name
    # 2D no arms
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere.osim'
    # 2D arms
    TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms2D.osim'     ###################################
    # set the model
    model = osim.Model(TreadmillModel);


    # % Reference data for tracking problem
    tableProcessor = osim.TableProcessor('./tracking2D2Darms_activation/3_OG_muscles_Tracking_solution__Halfgaitcycle.sto') # 2D2Darms/nat_1_IK.mot'); ###########################
    # tableProcessor.append(osim.TabOpLowPassFilter(6));
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());


        # change the contact sphere stiffnesses
    if 'newstiff' in locals():
        heelr_c = model.getComponent('contactHeel_r')
        heelr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(heelr_c)
        heelr_f.set_stiffness(newstiff) # type: ignore
        heell_c = model.getComponent('contactHeel_l')
        heell_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(heell_c)
        heell_f.set_stiffness(newstiff) # type: ignore

        latmidr_c = model.getComponent('contactLateralMidfoot_r')
        latmidr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(latmidr_c)
        latmidr_f.set_stiffness(newstiff) # type: ignore
        latmidl_c = model.getComponent('contactLateralMidfoot_l')
        latmidl_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(latmidl_c)
        latmidl_f.set_stiffness(newstiff) # type: ignore

        lattoer_c = model.getComponent('contactLateralToe_r')
        lattoer_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(lattoer_c)
        lattoer_f.set_stiffness(newstiff) # type: ignore
        lattoel_c = model.getComponent('contactLateralToe_l')
        lattoel_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(lattoel_c)
        lattoel_f.set_stiffness(newstiff) # type: ignore

        medtoer_c = model.getComponent('contactMedialToe_r')
        medtoer_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medtoer_c)
        medtoer_f.set_stiffness(newstiff) # type: ignore
        medtoel_c = model.getComponent('contactMedialToe_l')
        medtoel_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medtoel_c)
        medtoel_f.set_stiffness(newstiff) # type: ignore

        medmidr_c = model.getComponent('contactMedialMidfoot_r')
        medmidr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medmidr_c)
        medmidr_f.set_stiffness(newstiff) # type: ignore
        medmidl_c = model.getComponent('contactMedialMidfoot_l')
        medmidl_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medmidl_c)
        medmidl_f.set_stiffness(newstiff) # type: ignore

        model.initSystem()
        model.printToXML('2D_arms_stiffnessTweak.osim')
        model.initSystem()



    # if doing metabolics in the problem tweak the model
    if 'metabolicsWeight' in locals():
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
        ##
        # other thing is to set fast and slow twitches for recruitment
        ##
        # loop and add all the muscles to the model
        for m in range(numMuscles):
            muscle = muscles.get(m)
            muscleName = muscle.getName()
            musclePath = muscle.getAbsolutePathString()
            metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)))
        # premetmodel.addComponent(metabolics)
        premetmodel.finalizeConnections()
        premetmodel.printToXML('2D_arms_metabolicsModel.osim')
        modelProcessor = osim.ModelProcessor(premetmodel)

    else:
        modelProcessor = osim.ModelProcessor(model)




    # modelProcessor = osim.ModelProcessor(TreadmillModel);
    track.setModel(modelProcessor);
    track.setStatesReference(tableProcessor);
    track.set_states_global_tracking_weight(stateTrackingWeight);
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_apply_tracked_states_to_guess(True);
    track.set_initial_time(0.0);
    # track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
    study = track.initialize();
    problem = study.updProblem();

    # % Goals
    # % =====

    # % # % # % set specific weights for the individual weight set
    # % coordinateweights = MocoWeightSet();
    # % coordinateweights.cloneAndAppend(MocoWeight("lumbar", 10000000))
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tx", 1000000));
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_ty", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tz", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_list", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_rotation", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tilt", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_l", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_l", 0));
    # % track.set_states_weight_set(coordinateweights);


    # % # % Set different tracking weights for states (weights for states not 
    # % # % explicitly set here have a default value of 1.0). The values below
    # % # % were obtained by trial and error.
    # stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
    # stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0.0);
    # # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1000.0);
    # stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);
    # stateTrackingGoal.setWeightForState('/jointset/lumbar/lumbar/value', 10000.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 10.0);



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
        if str.startswith(currentStateName , '/jointset'):
            print('\niiii')
            print(currentStateName)

            # take joints out of the state squared goal, only want activations
            activationGoal.setWeightForState(currentStateName, 0)


            if '_r' in currentStateName and '_rot' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                               re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('first - rights')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if '_l' in currentStateName and '_rot' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                                re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
                print('second - lefts')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            if '_r' in currentStateName and '_rot' in currentStateName and '_l' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                    re.sub('_r/', '_l/', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('third - r with rot')
                print(currentStateName)
                print(re.sub('_r/', '_l/', currentStateName))
            if '_l' in currentStateName and '_rot' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                    re.sub('_l/', '_r/', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('forth - l with rot')
                print(currentStateName)
                print(re.sub('_l/', '_r/', currentStateName))
            if ('_r' not in currentStateName and 
                '_l' not in currentStateName and 
                'Treadmill_tx/value' not in currentStateName and 
                'Treadmill_tx/speed' not in currentStateName and 
                'Compensate_tx/value' not in currentStateName and 
                'Compensate_tx/speed' not in currentStateName and
                'pelvis_tx/value' not in currentStateName and
                '/activation' not in currentStateName):
                # 'pelvis_tx/speed' not in currentStateName and
                # % ~contains(currentStateName,'pelvis_tx/value') and 
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('bottom - leftovers that are not paired. ')
                print(currentStateName)


    # % Symmetric muscle activations
    # muscle activations cost as well
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):
            
            # activations squared for this actuator
            activationGoal.setWeightForState(currentStateName, activationWeightEach)

            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);


    # can try adding symmetry for the arms actuators for speed.



    # if using activation coordinate actuators, include symmetry for the lumbar
    symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair('/lumbarAct/activation'))
    
    # % Symmetric coordinate actuator controls
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/lumbarAct'));
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_flex_Actu_r', '/shoulder_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_add_Actu_r', '/shoulder_add_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_rot_Actu_r', '/shoulder_rot_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/elbow_flex_Actu_r', '/elbow_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/pro_sup_Actu_r', '/pro_sup_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_flex_Actu_r', '/wrist_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_dev_Actu_r', '/wrist_dev_Actu_l'))

    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_flex_Actu_l', '/shoulder_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_add_Actu_l', '/shoulder_add_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_rot_Actu_l', '/shoulder_rot_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/elbow_flex_Actu_l', '/elbow_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/pro_sup_Actu_l', '/pro_sup_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_flex_Actu_l', '/wrist_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_dev_Actu_l', '/wrist_dev_Actu_r'))



    # % Get a reference to the MocoControlGoal that is added to every MocoTrack
    # % problem by default and change the weight
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
    effort.setWeight(controlEffortWeight);
    effort.setDivideByDisplacement(True);
    effort.setExponent(effortExponent)

    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(2.67);
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    # if metabolics in the problem
    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight)
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        metabolicsGoal.setExponent(2)
        problem.addGoal(metabolicsGoal)



    # % head accelerations
    if 'headWeight' in locals():        
        headGoal = osim.MocoOutputGoal('headacc');
        headGoal.setOutputPath('bodyset/torso/head|acceleration');
        headGoal.setExponent(headExponent);
        headGoal.setWeight(headWeight);
        problem.addGoal(headGoal);

    # % pelvis accelerations
    if 'pelvisWeight' in locals():        
        pelvisGoal = osim.MocoOutputGoal('pelvisacc');
        pelvisGoal.setOutputPath('bodyset/pelvis/midPelv|acceleration');
        pelvisGoal.setExponent(pelvisExponent);
        pelvisGoal.setWeight(pelvisWeight);
        problem.addGoal(pelvisGoal);



    if 'heelForceWeight' in locals():
        # trying a contact force goal to not slam feet - taking out the left foot, I think at really high weights, the numbers shrink and make the cost function weird.
        heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
        heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
        heelForceGoal_r.setExponent(heelForceExponent);
        heelForceGoal_r.setWeight(heelForceWeight);

        lateralmidfootforcegoal_r = osim.MocoOutputGoal('lateralmidfoot_r');
        lateralmidfootforcegoal_r.setOutputPath('contactLateralMidfoot_r|sphere_force');
        lateralmidfootforcegoal_r.setExponent(toeForceExponent);
        lateralmidfootforcegoal_r.setWeight(toeForceWeight);

        medialmidfootforcegoal_r = osim.MocoOutputGoal('medialmidfoot_r');
        medialmidfootforcegoal_r.setOutputPath('contactMedialMidfoot_r|sphere_force');
        medialmidfootforcegoal_r.setExponent(toeForceExponent);
        medialmidfootforcegoal_r.setWeight(toeForceWeight);

        medialtoeforcegoal_r = osim.MocoOutputGoal('medialtoe_r');
        medialtoeforcegoal_r.setOutputPath('contactMedialToe_r|sphere_force');
        medialtoeforcegoal_r.setExponent(toeForceExponent);
        medialtoeforcegoal_r.setWeight(toeForceWeight);

        lateraltoeforcegoal_r = osim.MocoOutputGoal('lateraltoe_r');
        lateraltoeforcegoal_r.setOutputPath('contactLateralToe_r|sphere_force');
        lateraltoeforcegoal_r.setExponent(toeForceExponent);
        lateraltoeforcegoal_r.setWeight(toeForceWeight);

        problem.addGoal(heelForceGoal_r);
        problem.addGoal(lateralmidfootforcegoal_r);
        problem.addGoal(medialmidfootforcegoal_r);
        problem.addGoal(medialtoeforcegoal_r);
        problem.addGoal(lateraltoeforcegoal_r);


        # heelForceGoal_l = osim.MocoOutputGoal('heelforce_l');
        # heelForceGoal_l.setOutputPath('contactHeel_l|sphere_force');
        # heelForceGoal_l.setExponent(heelForceExponent);
        # heelForceGoal_l.setWeight(heelForceWeight);

        # lateralmidfootforcegoal_l = osim.MocoOutputGoal('lateralmidfoot_l');
        # lateralmidfootforcegoal_l.setOutputPath('contactLateralMidfoot_l|sphere_force');
        # lateralmidfootforcegoal_l.setExponent(toeForceExponent);
        # lateralmidfootforcegoal_l.setWeight(toeForceWeight);

        # medialmidfootforcegoal_l = osim.MocoOutputGoal('medialmidfoot_l');
        # medialmidfootforcegoal_l.setOutputPath('contactMedialMidfoot_l|sphere_force');
        # medialmidfootforcegoal_l.setExponent(toeForceExponent);
        # medialmidfootforcegoal_l.setWeight(toeForceWeight);

        # medialtoeforcegoal_l = osim.MocoOutputGoal('medialtoe_l');
        # medialtoeforcegoal_l.setOutputPath('contactMedialToe_l|sphere_force');
        # medialtoeforcegoal_l.setExponent(toeForceExponent);
        # medialtoeforcegoal_l.setWeight(toeForceWeight);

        # lateraltoeforcegoal_l = osim.MocoOutputGoal('lateraltoe_l');
        # lateraltoeforcegoal_l.setOutputPath('contactLateralToe_l|sphere_force');
        # lateraltoeforcegoal_l.setExponent(toeForceExponent);
        # lateraltoeforcegoal_l.setWeight(toeForceWeight);
        
        # problem.addGoal(heelForceGoal_l);
        # problem.addGoal(lateralmidfootforcegoal_l);
        # problem.addGoal(medialmidfootforcegoal_l);
        # problem.addGoal(medialtoeforcegoal_l);
        # problem.addGoal(lateraltoeforcegoal_l);


    if 'heelAccWeight' in locals():
        # now trying the heel/ foot accelerations cost
        # try heel acclerations
        heelGoalr = osim.MocoOutputGoal('heelracc');
        heelGoalr.setOutputPath('bodyset/calcn_r/heelr|acceleration');
        heelGoalr.setExponent(heelAccExponent); # type: ignore
        heelGoalr.setWeight(heelAccWeight); # type: ignore

        heelGoall = osim.MocoOutputGoal('heellacc');
        heelGoall.setOutputPath('bodyset/calcn_l/heell|acceleration');
        heelGoall.setExponent(heelAccExponent); # type: ignore
        heelGoall.setWeight(heelAccWeight); # type: ignore

        problem.addGoal(heelGoalr)
        problem.addGoal(heelGoall)


    if 'toeAccWeight' in locals():
        # now toe frame accelerations
        toeGoalr = osim.MocoOutputGoal('toeracc');
        toeGoalr.setOutputPath('bodyset/toes_r/toer|acceleration');
        toeGoalr.setExponent(toeAccExponent); # type: ignore
        toeGoalr.setWeight(toeAccWeight); # type: ignore

        toeGoall = osim.MocoOutputGoal('toelacc');
        toeGoall.setOutputPath('bodyset/toes_l/toel|acceleration');
        toeGoall.setExponent(toeAccExponent); # type: ignore
        toeGoall.setWeight(toeAccWeight); # type: ignore

        problem.addGoal(toeGoalr)
        problem.addGoal(toeGoall)



    # % Optionally, add a contact tracking goal.
    if GRFTrackingWeight != 0:
        # % Track the right and left vertical and fore-aft ground reaction forces.
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight);
        contactTracking.setExternalLoadsFile('grf_walk - Copy.xml');
        
        forceNamesRightFoot = osim.StdVectorString();
        forceNamesRightFoot.append('/contactHeel_r');
        # forceNamesRightFoot.append('/forceset/contactLateralRearfoot_r');
        forceNamesRightFoot.append('/contactLateralMidfoot_r');
        forceNamesRightFoot.append('/contactLateralToe_r');
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
        forceNamesLeftFoot.append('/contactLateralToe_l');
        forceNamesLeftFoot.append('/contactMedialToe_l');
        forceNamesLeftFoot.append('/contactMedialMidfoot_l');
        # contactTracking.addContactGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft = osim.MocoContactTrackingGoalGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft.append_alternative_frame_paths('/bodyset/toes_l')
        contactTracking.addContactGroup(contactTrackingSplitLeft);

        contactTracking.setProjection('plane');
        contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));
        problem.addGoal(contactTracking);


    # % Bounds
    # % ======
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);
    problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/knee_l/knee_angle_l/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);

    problem.setTimeBounds(0, [0.30, 0.35]);


    # % Configure the solver
    # % ====================
    solver = study.initCasADiSolver();
    solver.set_optim_finite_difference_scheme('forward')
    solver.set_parameters_require_initsystem(False);
    # duration = track.get_final_time() - track.get_initial_time();
    duration = finalTime - track.get_initial_time()
    num_mesh = round(duration/stepsize);
    solver.set_num_mesh_intervals(num_mesh);
    solver.set_verbosity(2);
    solver.set_optim_solver('ipopt');
    solver.set_optim_convergence_tolerance(convergeTolerance);
    solver.set_optim_constraint_tolerance(constraintTolerance);
    solver.set_optim_max_iterations(maxiterations);
    solver.set_scale_variables_using_bounds(True);
    solver.set_minimize_implicit_auxiliary_derivatives(True);
    solver.set_implicit_auxiliary_derivatives_weight(implicitAuxWeight);

    if guess:
        randomguess = solver.createGuess('bounds')
        newguess = helperOsimFunctions.fillGuess(randomguess, guessfile)
        solver.setGuess(newguess)



    # %{%
    # % Solve the problem
    # % =================
    gaitTrackingSolution = study.solve();
    # pdb.set_trace()
    # study.visualize(gaitTrackingSolution)

    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write(tag + 'CMATrack_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard
    # study.visualize(gaitTrackingSolution)
    # pdb.set_trace()
    # gaitTrackingTrajectory = osim.MocoTrajectory('CMATrack_OG_muscles_Tracking_solution__Halfgaitcycle.sto')

    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    addPatterns = osim.StdVectorString();
    addPatterns.append('.*pelvis_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution, addPatterns);
    fullStride.write(tag + 'CMATrack_OG_muscles_Tracking_solution_FullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);


    # % Extract ground reaction forces
    # % ==============================
    contact_r = osim.StdVectorString();
    contact_l = osim.StdVectorString();
    contact_r.append('/contactHeel_r');
    # contact_r.append('/contactLateralRearfoot_r');
    contact_r.append('/contactLateralMidfoot_r');
    contact_r.append('/contactLateralToe_r');
    contact_r.append('/contactMedialToe_r');
    contact_r.append('/contactMedialMidfoot_r');

    contact_l.append('/contactHeel_l');
    # contact_l.append('/contactLateralRearfoot_l');
    contact_l.append('/contactLateralMidfoot_l');
    contact_l.append('/contactLateralToe_l');
    contact_l.append('/contactMedialToe_l');
    contact_l.append('/contactMedialMidfoot_l');

    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, 
                                     fullStride,contact_r,contact_l);
    osim.STOFileAdapter.write(externalForcesTableFlat, 
                                 tag + 'CMATrack_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write(tag + 'CMATrack_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitTrackingSolution to fullstrideGRF:  tag + CMATrack_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n\n')

    # pdb.set_trace()
    # helperOsimFunctions.syncDrives(localDir, destDir)
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \  # TODO check on this
    # osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto'), \
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./tracking2D2Darms_activation/3_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \
    osim.TimeSeriesTable('./tracking2D2Darms_activation/3_OG_muscles_Tracking_solution_FullStride.sto'), \
    testobj

###########################################################################
# test objective function if you want evolving weights in CMA based on iter
###########################################################################
def objective_bilevel_CMATrackCompound(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_bitrack(x, 'CMATrack_')

        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')




        # print('\ngot the returns')
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];

        IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # get vectors of the GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('ground_force_r_vy').to_numpy();
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # flatten size
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()

        # get predicted hip, knee, ankle, lumbar
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy();

        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy();

        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy();

        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy();


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy();

        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy();


        # flatten size
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()

        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()

        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()

        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()

        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()

        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()

        ####################################
        # integrate through time for differences

        # # get the DTW RMSE 
        # distance, path = fastdtw(predvec2, IDvec2) #, dist=euclidean)
        # path_x, path_y = zip(*path)
        # aligned_x = np.array([predvec2[p] for p in path_x])
        # aligned_y = np.array([IDvec2[p] for p in path_y])
        # rmse = np.sqrt(np.mean((aligned_x - aligned_y) ** 2))

        # use DTW to get RMSE across time, and divide by experimental mean - dividing by stuff in the cost function now
        # hgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # vgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # hiperr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # kneeerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # ankleerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # tyvalueerr = 20 * helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)


        # print('\n before the dtw')

        # trying again with normalized by inf norm
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x)) # type: ignore
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y)) # type: ignore
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip)) # type: ignore
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee)) # type: ignore
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle)) # type: ignore
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) #  # type: ignore
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees) # type: ignore
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees) # type: ignore

        # print('\nafter dtw')
        # TODO - revisit this as the error function

        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # TODO add all terms
        kinerr = vgrferr + hgrferr + hiperr + kneeerr + ankleerr + lumbarerr + tyvalueerr + tilterr + err 


        # pull in the iteration that we are on, and then use it to multiply the l1 term
        # I think every 10 iterations the weight should increase maybe
        iterfilepath = 'iteration_.txt'
        iterfile = open(iterfilepath, 'r')
        iteration = int(iterfile.read())
        iterfile.close()
        # now want the weight multiplier to go up by tens
        multil1 = iteration//10
        basel1 = 1e4
        multiplyl1 = basel1 * 10**(multil1)

        # get the weights L1 regularization cost
        l1cost = multiplyl1 * np.sum(np.abs(10**(-x[0]))) # just 0 and 1 idx since they are the tracking weights - others can be whatever, since we want them to fill the void        

        # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        objexp = abs(np.log10(objectiveVal)) - 1
        objErr = 10**(abs(objexp))


        # do some weights error to try and get the tracking weights lower. 
        tot_err = kinerr + l1cost + objErr


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\npelvis tilt err: %f' % tilterr)
        print('\nerr: %f' % err)
        print('\nl1 mult: %f' % multiplyl1)
        print('\nl1cost: %f' % l1cost)
        print('\nobjectiveErr %f' % objErr)
        print('\ntot_err: %f' % tot_err)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\ntilterr: %f' % tilterr)
        outlog.write('\nerr: %f' % err)
        outlog.write('\nl1 mult: %f' % multiplyl1)
        outlog.write('\nl1cost: %f' % l1cost)
        outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\ntot_err: %f' % tot_err)
        outlog.close()
        # print('\nafter file stuff')



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

###########################################################################
# test bed when developing the metabolics stable simulations  x2
###########################################################################
def objective_bilevel_CMATrack_met(x):
    # going to try and recreate Umberger paper 2019
    '''
    e = f( hip angle, knee angle, ankle angle, vgrf, hgrf, and final time)
    '''
    ############################################
    # start by evaluating the lower level of the optimization - TODO return and store other things needed.
    # want to do try catch
    try:
        # gaitPredictionTable is a table of ground reaction forces
        # gaitIDTable is the table of GRF from experimental
        # kinemPred is a MocoTrajectory solution of prediction (has states)
        # kinemIKTable is table of IK results


        # get the outlog file
        outlogfilename = open('met_CMATrack_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()

        # this actually runs the optimization 
        gaitPredictionTable, kinemPred, gaitIDTable, kinemIKTable, objectiveVal = py_overgroundGait2D_bitrack_met(x, 'met_CMATrack_')

        # # pdb.set_trace()
        # # comment this only if you want to skip solving teh problem and just get some curves for debug
        # gaitPredictionTable = osim.TimeSeriesTable('CMA_bi_OG_muscles_Predicted_solutionGRF_fullStride.sto')
        # kinemPred = osim.MocoTrajectory('CMA_bi_OG_muscles_Predicted_solution_fullStride.sto')
        # # want to switch these to the tracking solution - better ankle and foot kinem than the IK
        # # gaitIDTable = osim.TimeSeriesTable('./expData/nat_1_GRF.mot')
        # # kinemIKTable = osim.TimeSeriesTable('./expData/2Darms/nat_1_IK.mot')
        # gaitIDTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto')
        # kinemIKTable = osim.TimeSeriesTable('3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')




        # print('\ngot the returns')
        kinemPredTable = kinemPred.exportToStatesTable()


        # compute end times for the predicion and the experimental reference
        predtime = np.asarray(gaitPredictionTable.getIndependentColumn())
        IDtime = np.asarray(gaitIDTable.getIndependentColumn())
        predictiontime = gaitPredictionTable.getIndependentColumn()[-1];
        gaitIDtime = gaitIDTable.getIndependentColumn()[-1];

        IKtime = np.asarray(kinemIKTable.getIndependentColumn())


        # get vectors of the GRF
        predvec_y = gaitPredictionTable.getDependentColumn('ground_force_r_vy').to_numpy();
        IDvec_y = gaitIDTable.getDependentColumn('ground_force_r_vy').to_numpy();
        predvec_x = gaitPredictionTable.getDependentColumn('ground_force_r_vx').to_numpy();
        IDvec_x = gaitIDTable.getDependentColumn('ground_force_r_vx').to_numpy();
        # flatten size
        predvec2_y = predvec_y.flatten()
        IDvec2_y = IDvec_y.flatten()
        predvec2_x = predvec_x.flatten()
        IDvec2_x = IDvec_x.flatten()

        # get predicted hip, knee, ankle, lumbar
        predvec_hip = (kinemPredTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy());
        IKvec_hip = kinemIKTable.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy();

        predvec_knee = (kinemPredTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy());
        IKvec_knee = kinemIKTable.getDependentColumn('/jointset/knee_r/knee_angle_r/value').to_numpy();

        predvec_ankle = (kinemPredTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy());
        IKvec_ankle = kinemIKTable.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy();

        predvec_lumbar = (kinemPredTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy());
        IKvec_lumbar = kinemIKTable.getDependentColumn('/jointset/lumbar/lumbar/value').to_numpy();


        # get the pelvis height 
        predvec_ty_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy());
        IKvec_ty_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy();

        predvec_tilt_value = (kinemPredTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy());
        IKvec_tilt_value = kinemIKTable.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy();


        # flatten size
        predvec2_hip = predvec_hip.flatten()
        IKvec2_hip = IKvec_hip.flatten()

        predvec2_knee = predvec_knee.flatten()
        IKvec2_knee = IKvec_knee.flatten()

        predvec2_ankle = predvec_ankle.flatten()
        IKvec2_ankle = IKvec_ankle.flatten()

        predvec2_lumbar = predvec_lumbar.flatten()
        IKvec2_lumbar = IKvec_lumbar.flatten()

        predvec2_ty_value = predvec_ty_value.flatten()
        IKvec2_ty_value = IKvec_ty_value.flatten()

        predvec2_tilt_value = predvec_tilt_value.flatten()
        IKvec2_tilt_value = IKvec_tilt_value.flatten()

        ####################################
        # integrate through time for differences

        # # get the DTW RMSE 
        # distance, path = fastdtw(predvec2, IDvec2) #, dist=euclidean)
        # path_x, path_y = zip(*path)
        # aligned_x = np.array([predvec2[p] for p in path_x])
        # aligned_y = np.array([IDvec2[p] for p in path_y])
        # rmse = np.sqrt(np.mean((aligned_x - aligned_y) ** 2))

        # use DTW to get RMSE across time, and divide by experimental mean - dividing by stuff in the cost function now
        # hgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x))
        # vgrferr = 1e-5 * helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y))
        # hiperr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip))
        # kneeerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee))
        # ankleerr = 1e-1 * helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle))
        # tyvalueerr = 20 * helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees)


        # print('\n before the dtw')

        # trying again with normalized by inf norm
        hgrferr = helperOsimFunctions.dtw_rmse(predvec2_x, IDvec2_x, inDegrees) # / abs(np.mean(IDvec_x)) # type: ignore
        vgrferr = helperOsimFunctions.dtw_rmse(predvec2_y, IDvec2_y, inDegrees) # / abs(np.mean(IDvec2_y)) # type: ignore
        hiperr = helperOsimFunctions.dtw_rmse(predvec2_hip, IKvec2_hip, inDegrees) # / abs(np.mean(IKvec2_hip)) # type: ignore
        kneeerr = helperOsimFunctions.dtw_rmse(predvec2_knee, IKvec2_knee, inDegrees) # / abs(np.mean(IKvec2_knee)) # type: ignore
        ankleerr = helperOsimFunctions.dtw_rmse(predvec2_ankle, IKvec2_ankle, inDegrees) # / abs(np.mean(IKvec2_ankle)) # type: ignore
        lumbarerr = helperOsimFunctions.dtw_rmse(predvec2_lumbar, IKvec2_lumbar, inDegrees) #  # type: ignore
        tyvalueerr = helperOsimFunctions.dtw_rmse(predvec2_ty_value, IKvec2_ty_value, inDegrees) # type: ignore
        tilterr = helperOsimFunctions.dtw_rmse(predvec2_tilt_value, IKvec2_tilt_value, inDegrees) # type: ignore

        # print('\nafter dtw')
        # TODO - revisit this as the error function

        #################################
        # then want cost of total duration for the gait cycle
        predictiontime_norm = predictiontime / gaitIDtime
        gaitIDtime_norm = 1.0


        err = 1e5*abs(predictiontime - gaitIDtime);
        # conditional to heavily weight times that are very different
        # if err >= 0.01:
        #     err = err*1e6
        # else: 
        #     err = err*4e4

        # TODO add all terms
        kinerr = vgrferr + hgrferr + hiperr + kneeerr + ankleerr + lumbarerr + tyvalueerr + tilterr + err 

        # get the weights L1 regularization cost
        l1cost = 1e6 * np.sum(np.abs(10**(-x[0]))) # just 0 and 1 idx since they are the tracking weights - others can be whatever, since we want them to fill the void

        # now penalize objective values in moco that are far from ~1 (want to make sure overall objective is close to 1 for tolerances, and convergence in general)
        objexp = abs(np.log10(objectiveVal)) - 1
        objErr = 10**(abs(objexp))


        # do some weights error to try and get the tracking weights lower. 
        tot_err = kinerr + l1cost + objErr


        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        print('\n\ninputs\n')
        print(x)
        print('\n')

        # pdb.set_trace()

        # print errors to see relative magnitudes
        print('\nhgrferr: %f' % hgrferr)
        print('\nvgrferr: %f' % vgrferr)
        print('\nhiperr: %f' % hiperr)
        print('\nkneeerr: %f' % kneeerr)
        print('\nankleerr: %f' % ankleerr)
        print('\nlumbarerr: %f' % lumbarerr)
        print('\npelvis ty err: %f' % tyvalueerr)
        print('\npelvis tilt err: %f' % tilterr)
        print('\nerr: %f' % err)
        print('\nl1cost: %f' % l1cost)
        print('\nobjectiveErr %f' % objErr)
        print('\ntot_err: %f' % tot_err)


        outlog.write('\nhgrferr: %f' % hgrferr)
        outlog.write('\nvgrferr: %f' % vgrferr)
        outlog.write('\nhiperr: %f' % hiperr)
        outlog.write('\nkneeerr: %f' % kneeerr)
        outlog.write('\nankleerr: %f' % ankleerr)
        outlog.write('\nlumbarerr: %f' % lumbarerr)
        outlog.write('\ntyvalueerr: %f' % tyvalueerr)
        outlog.write('\ntilterr: %f' % tilterr)
        outlog.write('\nerr: %f' % err)
        outlog.write('\nl1cost: %f' % l1cost)
        outlog.write('\nobjectiveErr %f' % objErr)
        outlog.write('\ntot_err: %f' % tot_err)
        outlog.close()
        # print('\nafter file stuff')



    except Exception as e:
        # the moco simulation likely failed
        print('\nFailed moco Simulation\n')
        print(str(e))
        print('error %f' % 5e8)
        # assign a large error to move away from something that didn't solve... may need tuned.
        outlogfilename = open('CMA_logfile.txt', 'r')
        outlogfile = outlogfilename.read()
        outlogfilename.close()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nInputs:\n')
        outlog.write(str(x))
        outlog.write('\nFAILED MOCO SIMULATION... error %f' % 5e8)
        outlog.write(str(e))
        outlog.close()
        tot_err = 5e8 # arbitrary based on some of the errors that I got in one pass


    return tot_err

def py_overgroundGait2D_bitrack_met(x, tag): #[effortWeight, effortExponent, activationWeight, headWeight, headExponent, implicitAuxWeight]):
    # # % -------------------------------------------------------------------------- %
    # # % working on getting python versions of my own scripts going.
    # # % this function will be a tracking problem used in CMA to reduce tracking terms
    # # % while also figuring out what other terms need to be included to maintain good
    # # % kinematics that are representative of experimental (known)
    # # % Author: Jon Stingel
    # # % 20230720
    # # % -------------------------------------------------------------------------- %

    # # % imports
    import os
    # os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
    # os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")

    import opensim as osim
    import re
    import numpy as np
    import pdb
    import helperOsimFunctions


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
    # % controlEffortWeight = 1e-8;
    # % stateTrackingWeight = 5e-1;
    # % GRFTrackingWeight   = 1e-1;

    # % # % these are the good weights
    # % controlEffortWeight = 1e-1;
    # % stateTrackingWeight = 1e-1;
    # % GRFTrackingWeight   = 1e-1;

    # stateTrackingWeight = 1e-1;
    # GRFTrackingWeight   = 1e0;
    
    # excitations weight
    controlEffortWeight = 10**(-x[0]) # 1e-3; # 1e-1
    effortExponent = 3

    # activations weights
    activationWeightEach = 10**(-x[1]) # 1e0
    activationWeight = 1e-1

    # metabolic effort
    # metabolicsWeight = 1


    # tracking weights
    trackingWeightTemp = x[2]
    stateTrackingWeight = 10**(-trackingWeightTemp) 
    GRFTrackingWeight = 10**(-trackingWeightTemp+1)


    implicitAuxWeight = 1e-6;
    # now to add all the other weights from the other problems that we are going to include

    # GRF magnitudes - set these to the same weight
    heelForceWeighttemp = x[3] # 1e-20;
    heelForceWeight = 10**(-heelForceWeighttemp)
    heelForceExponent = 3;
    toeForceWeighttemp = x[4] # heelForceWeight # 1e-20;
    toeForceWeight = 10**(-toeForceWeighttemp)
    toeForceExponent = 3;

    # heel acc 
    # heelAccWeighttemp = x[0] # 1e-2;
    # heelAccWeight = 10**(-heelAccWeighttemp)
    # heelAccExponent = 2;

    # head acceleration
    headWeighttemp = x[5]; # 1e-2; # 1e-2 for now - seems okay on the ty motion
    headWeight = 10**(-headWeighttemp)
    headExponent = 2; # 2;
    
    # editing the contact spheres
    newstifftemp = x[6]; #3067776
    newstiff = 10**(newstifftemp)

    convergeTolerance   = 1e-2;
    constraintTolerance = 1e-3;

    stepsize = .03;
    maxiterations = 2500;
    finalTime = 0.6835 / 2;

    guess = True;
    wantguess = False;
    # guessfile = 'goodOverground_muscle_tracking_guess.sto';
    # guessfile = 'guess_2D3DsphereArms_half.sto';
    # guessfile = 'good_guess_2D3DsphereArms_half.sto';
    guessfile = './tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto'


    # % Define the optimal control problem
    # % ==================================
    track = osim.MocoTrack();
    track.setName('CMATrack_muscle_GaitTracking');


    # % Set the OpenSim Model and give it a name
    # % TreadmillModel ='Running267_TM.osim'; %uigetfile('*.osim'); %This code will work for all three speed conditions, choose the model you want to run
    # TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere_arms.osim';
    TreadmillModel = '2DMuscles_OG_complextoes_3Dsphere.osim'; ###########################
    model = osim.Model(TreadmillModel);


    # % Reference data for tracking problem
    tableProcessor = osim.TableProcessor('./expData/2Darms/nat_1_IK.mot'); ###########################
    tableProcessor.append(osim.TabOpLowPassFilter(6));
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames());



    # change the contact sphere stiffnesses
    if 'newstiff' in locals():
        heelr_c = model.getComponent('contactHeel_r')
        heelr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(heelr_c)
        heelr_f.set_stiffness(newstiff)
        heell_c = model.getComponent('contactHeel_l')
        heell_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(heell_c)
        heell_f.set_stiffness(newstiff)

        latmidr_c = model.getComponent('contactLateralMidfoot_r')
        latmidr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(latmidr_c)
        latmidr_f.set_stiffness(newstiff)
        latmidl_c = model.getComponent('contactLateralMidfoot_l')
        latmidl_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(latmidl_c)
        latmidl_f.set_stiffness(newstiff)

        lattoer_c = model.getComponent('contactLateralToe_r')
        lattoer_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(lattoer_c)
        lattoer_f.set_stiffness(newstiff)
        lattoel_c = model.getComponent('contactLateralToe_l')
        lattoel_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(lattoel_c)
        lattoel_f.set_stiffness(newstiff)

        medtoer_c = model.getComponent('contactMedialToe_r')
        medtoer_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medtoer_c)
        medtoer_f.set_stiffness(newstiff)
        medtoel_c = model.getComponent('contactMedialToe_l')
        medtoel_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medtoel_c)
        medtoel_f.set_stiffness(newstiff)

        medmidr_c = model.getComponent('contactMedialMidfoot_r')
        medmidr_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medmidr_c)
        medmidr_f.set_stiffness(newstiff)
        medmidl_c = model.getComponent('contactMedialMidfoot_l')
        medmidl_f = osim.SmoothSphereHalfSpaceForce.safeDownCast(medmidl_c)
        medmidl_f.set_stiffness(newstiff)

        model.initSystem()



    # if doing metabolics in the problem tweak the model
    if 'metabolicsWeight' in locals():
        # adding metabolics effort to the cost
        modelProcessor = osim.ModelProcessor(model);
        premetmodel = modelProcessor.process()
        premetmodel.initSystem()
        muscles = premetmodel.getMuscles()
        numMuscles = muscles.getSize()
        metabolics = osim.Bhargava2004SmoothedMuscleMetabolics()
        metabolics.setName('metabolic_cost')
        metabolics.set_use_smoothing(True)
        # loop and add all the muscles to the model
        for m in range(numMuscles):
            muscle = muscles.get(m)
            muscleName = muscle.getName()
            musclePath = muscle.getAbsolutePathString()
            metabolics.addMuscle(muscleName, osim.Muscle.safeDownCast(premetmodel.getComponent(musclePath)))
        premetmodel.addComponent(metabolics)
        premetmodel.finalizeConnections()
        premetmodel.printToXML('2D_noarms_metabolicsModel.osim')
        modelProcessor = osim.ModelProcessor(premetmodel)
    else:
        modelProcessor = osim.ModelProcessor(model)




    track.setModel(modelProcessor);
    track.setStatesReference(tableProcessor);
    track.set_states_global_tracking_weight(stateTrackingWeight);
    track.set_allow_unused_references(True);
    track.set_track_reference_position_derivatives(True);
    track.set_apply_tracked_states_to_guess(True);
    track.set_initial_time(0.0);
    # track.set_final_time(finalTime); #%0.684 I think. works with half as 0.342
    study = track.initialize();
    problem = study.updProblem();

    # % Goals
    # % =====

    # % # % # % set specific weights for the individual weight set
    # % coordinateweights = MocoWeightSet();
    # % coordinateweights.cloneAndAppend(MocoWeight("lumbar", 10000000))
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tx", 1000000));
    # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_ty", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tz", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_list", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_rotation", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("pelvis_tilt", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("hip_rotation_l", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_r", 0));
    # % # % # % coordinateweights.cloneAndAppend(MocoWeight("ankle_angle_l", 0));
    # % track.set_states_weight_set(coordinateweights);


    # % # % Set different tracking weights for states (weights for states not 
    # % # % explicitly set here have a default value of 1.0). The values below
    # % # % were obtained by trial and error.
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/value', 0.0);
    # % stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_ty/speed', 1000.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/groundPelvis/pelvis_tx/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0.0);
    stateTrackingGoal.setWeightForState('/jointset/lumbar/lumbar/value', 10000.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 10.0);
    # % # % stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 10.0);



    # activation goal setup
    activationGoal = osim.MocoSumSquaredStateGoal('activationGoal')
    if 'activationWeight' in locals():
        activationGoal.setWeight(activationWeight)
        activationGoal.setDivideByDisplacement(True)
        problem.addGoal(activationGoal)



    # % Symmetry (to permit simulating only one step)
    symmetryGoal = osim.MocoPeriodicityGoal('symmetryGoal');
    problem.addGoal(symmetryGoal);
    model = modelProcessor.process();
    model.initSystem();

    # % Symmetric coordinate values (except for pelvis_tx) and speeds
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.startswith(currentStateName , '/jointset'):
            print('\niiii')
            print(currentStateName)
            if '_r' in currentStateName and '_rot' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                               re.sub('_r', '_l', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('first - rights')
                print(currentStateName)
                print(re.sub('_r', '_l', currentStateName))
            if '_l' in currentStateName and '_rot' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                                re.sub('_l', '_r', currentStateName)); 
                symmetryGoal.addStatePair(pair);
                print('second - lefts')
                print(currentStateName)
                print(re.sub('_l', '_r', currentStateName))
            if '_r' in currentStateName and '_rot' in currentStateName and '_l' not in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName,
                    re.sub('_r/', '_l/', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('third - r with rot')
                print(currentStateName)
                print(re.sub('_r/', '_l/', currentStateName))
            if '_l' in currentStateName and '_rot' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                    re.sub('_l/', '_r/', currentStateName));
                symmetryGoal.addStatePair(pair);
                print('forth - l with rot')
                print(currentStateName)
                print(re.sub('_l/', '_r/', currentStateName))
            if ('_r' not in currentStateName and 
                '_l' not in currentStateName and 
                'Treadmill_tx/value' not in currentStateName and 
                'Treadmill_tx/speed' not in currentStateName and 
                'Compensate_tx/value' not in currentStateName and 
                'Compensate_tx/speed' not in currentStateName and
                'pelvis_tx/value' not in currentStateName and
                '/activation' not in currentStateName):
                # 'pelvis_tx/speed' not in currentStateName and
                # % ~contains(currentStateName,'pelvis_tx/value') and 
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(currentStateName));
                print('bottom - leftovers that are not paired. ')
                print(currentStateName)


    # % Symmetric muscle activations
    # muscle activations cost as well
    for i in range(model.getNumStateVariables()):
        currentStateName = str(model.getStateVariableNames().getitem(i));
        if str.endswith(currentStateName,'/activation'):
            
            # activations squared for this actuator
            activationGoal.setWeightForState(currentStateName, activationWeightEach)

            if '_r' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_r','_l', currentStateName));
                symmetryGoal.addStatePair(pair);
            
            if '_l' in currentStateName:
                pair = osim.MocoPeriodicityGoalPair(currentStateName, 
                             re.sub('_l', '_r', currentStateName));
                symmetryGoal.addStatePair(pair);


    # can try adding symmetry for the arms actuators for speed.



    # % Symmetric coordinate actuator controls
    symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/lumbarAct'));
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_flex_Actu_r', '/shoulder_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_add_Actu_r', '/shoulder_add_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_rot_Actu_r', '/shoulder_rot_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/elbow_flex_Actu_r', '/elbow_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/pro_sup_Actu_r', '/pro_sup_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_flex_Actu_r', '/wrist_flex_Actu_l'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_dev_Actu_r', '/wrist_dev_Actu_l'))

    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_flex_Actu_l', '/shoulder_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_add_Actu_l', '/shoulder_add_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/shoulder_rot_Actu_l', '/shoulder_rot_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/elbow_flex_Actu_l', '/elbow_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/pro_sup_Actu_l', '/pro_sup_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_flex_Actu_l', '/wrist_flex_Actu_r'))
    # symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair('/wrist_dev_Actu_l', '/wrist_dev_Actu_r'))



    # % Get a reference to the MocoControlGoal that is added to every MocoTrack
    # % problem by default and change the weight
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
    effort.setWeight(controlEffortWeight);
    effort.setDivideByDisplacement(True);
    effort.setExponent(effortExponent)

    speedGoal = osim.MocoAverageSpeedGoal('speed');
    speedGoal.set_desired_average_speed(2.67);
    speedGoal.setMode('endpoint_constraint');
    problem.addGoal(speedGoal);


    if 'metabolicsWeight' in locals():
        metabolicsGoal = osim.MocoOutputGoal('metabolics', metabolicsWeight) # type: ignore
        metabolicsGoal.setOutputPath('/metabolic_cost|total_metabolic_rate')
        metabolicsGoal.setDivideByDisplacement(True)
        metabolicsGoal.setDivideByMass(True)
        problem.addGoal(metabolicsGoal)



    # % head accelerations
    if 'headWeight' in locals():        
        headGoal = osim.MocoOutputGoal('headacc');
        headGoal.setOutputPath('bodyset/torso/head|acceleration');
        headGoal.setExponent(headExponent);
        headGoal.setWeight(headWeight);
        problem.addGoal(headGoal);

    if 'heelForceWeight' in locals():
        # trying a contact force goal to not slam feet - taking out the left foot, I think at really high weights, the numbers shrink and make the cost function weird.
        heelForceGoal_r = osim.MocoOutputGoal('heelforce_r');
        heelForceGoal_r.setOutputPath('contactHeel_r|sphere_force');
        heelForceGoal_r.setExponent(heelForceExponent);
        heelForceGoal_r.setWeight(heelForceWeight);

        lateralmidfootforcegoal_r = osim.MocoOutputGoal('lateralmidfoot_r');
        lateralmidfootforcegoal_r.setOutputPath('contactLateralMidfoot_r|sphere_force');
        lateralmidfootforcegoal_r.setExponent(toeForceExponent);
        lateralmidfootforcegoal_r.setWeight(toeForceWeight);

        medialmidfootforcegoal_r = osim.MocoOutputGoal('medialmidfoot_r');
        medialmidfootforcegoal_r.setOutputPath('contactMedialMidfoot_r|sphere_force');
        medialmidfootforcegoal_r.setExponent(toeForceExponent);
        medialmidfootforcegoal_r.setWeight(toeForceWeight);

        medialtoeforcegoal_r = osim.MocoOutputGoal('medialtoe_r');
        medialtoeforcegoal_r.setOutputPath('contactMedialToe_r|sphere_force');
        medialtoeforcegoal_r.setExponent(toeForceExponent);
        medialtoeforcegoal_r.setWeight(toeForceWeight);

        lateraltoeforcegoal_r = osim.MocoOutputGoal('lateraltoe_r');
        lateraltoeforcegoal_r.setOutputPath('contactLateralToe_r|sphere_force');
        lateraltoeforcegoal_r.setExponent(toeForceExponent);
        lateraltoeforcegoal_r.setWeight(toeForceWeight);

        problem.addGoal(heelForceGoal_r);
        problem.addGoal(lateralmidfootforcegoal_r);
        problem.addGoal(medialmidfootforcegoal_r);
        problem.addGoal(medialtoeforcegoal_r);
        problem.addGoal(lateraltoeforcegoal_r);


        # heelForceGoal_l = osim.MocoOutputGoal('heelforce_l');
        # heelForceGoal_l.setOutputPath('contactHeel_l|sphere_force');
        # heelForceGoal_l.setExponent(heelForceExponent);
        # heelForceGoal_l.setWeight(heelForceWeight);

        # lateralmidfootforcegoal_l = osim.MocoOutputGoal('lateralmidfoot_l');
        # lateralmidfootforcegoal_l.setOutputPath('contactLateralMidfoot_l|sphere_force');
        # lateralmidfootforcegoal_l.setExponent(toeForceExponent);
        # lateralmidfootforcegoal_l.setWeight(toeForceWeight);

        # medialmidfootforcegoal_l = osim.MocoOutputGoal('medialmidfoot_l');
        # medialmidfootforcegoal_l.setOutputPath('contactMedialMidfoot_l|sphere_force');
        # medialmidfootforcegoal_l.setExponent(toeForceExponent);
        # medialmidfootforcegoal_l.setWeight(toeForceWeight);

        # medialtoeforcegoal_l = osim.MocoOutputGoal('medialtoe_l');
        # medialtoeforcegoal_l.setOutputPath('contactMedialToe_l|sphere_force');
        # medialtoeforcegoal_l.setExponent(toeForceExponent);
        # medialtoeforcegoal_l.setWeight(toeForceWeight);

        # lateraltoeforcegoal_l = osim.MocoOutputGoal('lateraltoe_l');
        # lateraltoeforcegoal_l.setOutputPath('contactLateralToe_l|sphere_force');
        # lateraltoeforcegoal_l.setExponent(toeForceExponent);
        # lateraltoeforcegoal_l.setWeight(toeForceWeight);
        
        # problem.addGoal(heelForceGoal_l);
        # problem.addGoal(lateralmidfootforcegoal_l);
        # problem.addGoal(medialmidfootforcegoal_l);
        # problem.addGoal(medialtoeforcegoal_l);
        # problem.addGoal(lateraltoeforcegoal_l);


    if 'heelAccWeight' in locals():
        # now trying the heel/ foot accelerations cost
        # try heel acclerations
        heelGoalr = osim.MocoOutputGoal('heelracc');
        heelGoalr.setOutputPath('bodyset/calcn_r/heelr|acceleration');
        heelGoalr.setExponent(heelAccExponent); # type: ignore
        heelGoalr.setWeight(heelAccWeight); # type: ignore

        heelGoall = osim.MocoOutputGoal('heellacc');
        heelGoall.setOutputPath('bodyset/calcn_l/heell|acceleration');
        heelGoall.setExponent(heelAccExponent); # type: ignore
        heelGoall.setWeight(heelAccWeight); # type: ignore

        problem.addGoal(heelGoalr)
        problem.addGoal(heelGoall)


    if 'toeAccWeight' in locals():
        # now toe frame accelerations
        toeGoalr = osim.MocoOutputGoal('toeracc');
        toeGoalr.setOutputPath('bodyset/toes_r/toer|acceleration');
        toeGoalr.setExponent(toeAccExponent); # type: ignore
        toeGoalr.setWeight(toeAccWeight); # type: ignore

        toeGoall = osim.MocoOutputGoal('toelacc');
        toeGoall.setOutputPath('bodyset/toes_l/toel|acceleration');
        toeGoall.setExponent(toeAccExponent); # type: ignore
        toeGoall.setWeight(toeAccWeight); # type: ignore

        problem.addGoal(toeGoalr)
        problem.addGoal(toeGoall)



    # % Optionally, add a contact tracking goal.
    if GRFTrackingWeight != 0:
        # % Track the right and left vertical and fore-aft ground reaction forces.
        contactTracking = osim.MocoContactTrackingGoal('contact', GRFTrackingWeight);
        contactTracking.setExternalLoadsFile('grf_walk - Copy.xml');
        
        forceNamesRightFoot = osim.StdVectorString();
        forceNamesRightFoot.append('/contactHeel_r');
        # forceNamesRightFoot.append('/forceset/contactLateralRearfoot_r');
        forceNamesRightFoot.append('/contactLateralMidfoot_r');
        forceNamesRightFoot.append('/contactLateralToe_r');
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
        forceNamesLeftFoot.append('/contactLateralToe_l');
        forceNamesLeftFoot.append('/contactMedialToe_l');
        forceNamesLeftFoot.append('/contactMedialMidfoot_l');
        # contactTracking.addContactGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft = osim.MocoContactTrackingGoalGroup(forceNamesLeftFoot, 'Left_GRF');
        contactTrackingSplitLeft.append_alternative_frame_paths('/bodyset/toes_l')
        contactTracking.addContactGroup(contactTrackingSplitLeft);


        contactTracking.setProjection('plane');
        contactTracking.setProjectionVector(osim.Vec3(0, 0, 1));
        problem.addGoal(contactTracking);


    # % Bounds
    # % ======
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tilt/value', [-20*np.pi/180, 10*np.pi/180]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_tx/value', [0, 5], [0]);
    problem.setStateInfo('/jointset/groundPelvis/pelvis_ty/value', [0.75, 1.25]);
    problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-20*np.pi/180, 60*np.pi/180]);
    problem.setStateInfo('/jointset/knee_l/knee_angle_l/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', [-150*np.pi/180, 0]);
    problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-30*np.pi/180, 30*np.pi/180]);
    problem.setStateInfo('/jointset/lumbar/lumbar/value', [-30*np.pi/180, 20*np.pi/180]);

    problem.setTimeBounds(0, [0.30, 0.35]);


    # % Configure the solver
    # % ====================
    solver = study.initCasADiSolver();
    solver.set_optim_finite_difference_scheme('forward')
    solver.set_parameters_require_initsystem(False);
    # duration = track.get_final_time() - track.get_initial_time();
    duration = finalTime - track.get_initial_time()
    num_mesh = round(duration/stepsize);
    solver.set_num_mesh_intervals(num_mesh);
    solver.set_verbosity(2);
    solver.set_optim_solver('ipopt');
    solver.set_optim_convergence_tolerance(convergeTolerance);
    solver.set_optim_constraint_tolerance(constraintTolerance);
    solver.set_optim_max_iterations(maxiterations);
    solver.set_scale_variables_using_bounds(True);
    solver.set_minimize_implicit_auxiliary_derivatives(True);
    solver.set_implicit_auxiliary_derivatives_weight(implicitAuxWeight);

    if guess:
        randomguess = solver.createGuess('bounds')
        newguess = helperOsimFunctions.fillGuess(randomguess, guessfile)
        solver.setGuess(newguess)



    # %{%
    # % Solve the problem
    # % =================
    gaitTrackingSolution = study.solve();
    testobj = gaitTrackingSolution.getObjective()
    gaitTrackingSolution.write(tag + 'CMATrack_OG_muscles_Tracking_solution__Halfgaitcycle.sto');
    # gaitTrackingSolution.write('./results/1_nat_muscles_Tracking_solution__Halfgaitcycle.sto');
    # % keyboard
    # study.visualize(gaitTrackingSolution)
    # pdb.set_trace()
    # gaitTrackingSolution = osim.MocoTrajectory('3_2D3D_OG_muscles_Tracking_solution__Halfgaitcycle.sto')

    # % Create a full stride from the periodic single step solution.
    # % For details, navigate to
    # % User Guide > Utilities > Model and trajectory utilities
    # % in the Moco Documentation.
    addPatterns = osim.StdVectorString();
    addPatterns.append('.*pelvis_tx/value');
    fullStride = osim.createPeriodicTrajectory(gaitTrackingSolution, addPatterns);
    fullStride.write(tag + 'CMATrack_OG_muscles_Tracking_solution_FullStride.sto');

    # % Uncomment next line to visualize the result
    # % study.visualize(fullStride);


    # % Extract ground reaction forces
    # % ==============================
    contact_r = osim.StdVectorString();
    contact_l = osim.StdVectorString();
    contact_r.append('/contactHeel_r');
    # contact_r.append('/contactLateralRearfoot_r');
    contact_r.append('/contactLateralMidfoot_r');
    contact_r.append('/contactLateralToe_r');
    contact_r.append('/contactMedialToe_r');
    contact_r.append('/contactMedialMidfoot_r');

    contact_l.append('/contactHeel_l');
    # contact_l.append('/contactLateralRearfoot_l');
    contact_l.append('/contactLateralMidfoot_l');
    contact_l.append('/contactLateralToe_l');
    contact_l.append('/contactMedialToe_l');
    contact_l.append('/contactMedialMidfoot_l');

    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, 
                                     fullStride,contact_r,contact_l);
    osim.STOFileAdapter.write(externalForcesTableFlat, 
                                 tag + 'CMATrack_OG_muscles_Tracking_solutionGRF_FullStride.sto');
    fullstrideGRF = fullStride;
    fullstrideGRF.insertStatesTrajectory(externalForcesTableFlat);
    fullstrideGRF.write(tag + 'CMATrack_OG_muscles_Tracking_solution_fullStride_wGRF.sto');


    print('\n\nchange wantguess to True if we want to overwrite the guess to this solution!')
    print('\ngaitTrackingSolution to fullstrideGRF:  tag + CMATrack_OG_muscles_Tracking_solution_fullStride_wGRF.sto\n\n')


    # helperOsimFunctions.syncDrives(localDir, destDir)
    return externalForcesTableFlat, \
    fullStride, \
    osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solutionGRF_FullStride.sto'), \
    osim.TimeSeriesTable('./tracking2DNoArms/3_2D3D_OG_muscles_Tracking_solution_FullStride.sto')
