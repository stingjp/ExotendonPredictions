## creating a module of helper functions for opensim scripts
# jon stingel
# 20230410
##################################################################
import os
from pyexpat import model
from telnetlib import EXOPL
from turtle import right
from weakref import ref

from scipy import stats
from exceptiongroup import catch
from matplotlib import table
from matplotlib.pylab import f
from pygments import highlight
import mpld3
from IPython.display import display_html

# os.add_dll_directory("C:/OpenSim 4.4/bin")
# os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
# os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")
# os.add_dll_directory('C:/Users/jonstingel/opensim/opensim-core-4.5-2024-05-15-a1a2282/bin')
os.add_dll_directory('C:/Users/jonstingel/opensim-core-4.5.1-2024-08-23-cf3ef35/bin')
import opensim as osim
import pdb
import numpy as np
import shutil
import time
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat
import pandas as pd
from IPython.display import display, HTML
import numpy as np
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation
import textwrap
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# highlight text for the outputs
def highlight_text(text):
    display(HTML(f'<span style="background-color: black; font-weight: bold;">{text}</span>'))

# handsfield max isometric muscle force scaling
def scaleModelMaxIsometricForces(subjectmass, subjectheight, genericmodelfile, subjectmodelfile, newsubjfile):
    # input the subject's mass and height and return a model that has the muscle max 
    # forces scaled based on handsfield muscle volume regression

    # get the generic model
    generic_model = osim.Model(genericmodelfile)
    generic_mset = generic_model.getMuscles()    

    # get the subject specific model
    subj_model = osim.Model(subjectmodelfile)
    subj_mset = subj_model.getMuscles()



    # regression functions
    def total_muscle_volume_regression_mass(mass):
        """cm^3"""
        return 91.0*mass + 588.0

    def total_muscle_volume_regression_massandheight(mass, height):
        """cm^3"""
        return 47.05*mass*height + 1289.6


    # if we have nonzero height, we use it to scale
    if subjectheight != 0:
        # % # TMV: total muscle volume (cm^3).
        generic_TMV_massandheight = total_muscle_volume_regression_massandheight(75.337, 1.7);
        subj_TMV_massandheight = total_muscle_volume_regression_massandheight(subjectmass, subjectheight);

        # go through and scale each muscle
        for im in range(subj_model.getMuscles().getSize()):
            subj_muscle_name = subj_mset.get(im).getName()
            subj_muscle = subj_mset.get(subj_muscle_name)
            generic_muscle = generic_mset.get(subj_muscle_name)

            # OFL: optimal fiber length (cm).
            generic_OFL = generic_muscle.get_optimal_fiber_length() * 100
            subj_OFL = subj_muscle.get_optimal_fiber_length() * 100

            scale_factor = (subj_TMV_massandheight/generic_TMV_massandheight) * (generic_OFL/subj_OFL)
            print("Scaling '%s' muscle force by %f." % (subj_muscle_name, scale_factor))

            generic_force = generic_muscle.get_max_isometric_force()
            scaled_force = generic_force*scale_factor
            subj_muscle.set_max_isometric_force(scaled_force)

    # use only the mass to scale the forces in the model
    else:
        # TMV: total muscle volume (cm^3).
        generic_TMV_mass = total_muscle_volume_regression_mass(92.383)
        subj_TMV_mass = total_muscle_volume_regression_mass(subjectmass)

        # go through and scale each muscle
        for im in range(subj_model.getMuscles().getSize()):
            subj_muscle_name = subj_mset.get(im).getName()
            subj_muscle = subj_mset.get(subj_muscle_name)
            generic_muscle = generic_mset.get(subj_muscle_name)

            # OFL: optimal fiber length (cm).
            generic_OFL = generic_muscle.get_optimal_fiber_length() * 100
            subj_OFL = subj_muscle.get_optimal_fiber_length() * 100

            scale_factor = (subj_TMV_mass/generic_TMV_mass) * (generic_OFL/subj_OFL)
            print("Scaling '%s' muscle force by %f." % (subj_muscle_name, scale_factor))

            generic_force = generic_muscle.get_max_isometric_force()
            scaled_force = generic_force*scale_factor
            subj_muscle.set_max_isometric_force(scaled_force)

    subj_model.printToXML(newsubjfile)
    return subj_model

# create a quick method for plotting norm fiber velocities
def plotNormFiberVelocityPaths(path1, path2, tag1, tag2):
    # okay the goal for this function is to take a file or solution and plot the norm fiber velocity
    # analysisDir = './goodResults/'
    # analysisFile = 'quickAnalysis_' + tag1 + '_MuscleAnalysis_NormalizedFiberLength.sto'
    analysisFile = 'quickAnalysis_' + tag1 + '_Normalized_Fiber_Velocity.sto'


    fiberTable = osim.TimeSeriesTable(os.path.join(path1, analysisFile))
    time = fiberTable.getIndependentColumn()
    fibercols = fiberTable.getColumnLabels()
    numcols = len(fibercols)

        
    # if second tag, get second table
    if path2 != "":
        # analysisFile2 = 'quickAnalysis_' + tag2 + '_MuscleAnalysis_NormalizedFiberLength.sto'
        analysisFile2 = 'quickAnalysis_' + tag2 + '_Normalized_Fiber_Velocity.sto'
        fiberTable2 = osim.TimeSeriesTable(os.path.join(path2, analysisFile2))
        time2 = fiberTable2.getIndependentColumn()
        fibercols2 = fiberTable2.getColumnLabels()
        numcols2 = len(fibercols2)
    
    # want a 3x6 plot
    fig1, ax1 = plt.subplots(3, 6, figsize=(15,7))
    for i, ax in enumerate(fig1.axes):
        # get a column and plot it. 
        tempcol = fiberTable.getDependentColumn(fibercols[i]).to_numpy()
        ax.plot(time, tempcol, label=tag1)
        
        # second curve?
        if tag2 != "":
            tempcol2 = fiberTable2.getDependentColumn(fibercols[i]).to_numpy()
            ax.plot(time2, tempcol2, label=tag2)
        
        if i > 11:
            ax.set_xlabel('time(s)')
        if i==0 or i==6 or i==12:
            ax.set_ylabel('Norm. Fiber Velocity')

        # print(fibercols[i])
        tempname = fibercols[i].split('|')
        tempname2 = tempname[0].split('/')

        # print(tempname2)
        # pdb.set_trace()
        ax.title.set_text(tempname2[2]) # fibercols[i]
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

    return

# create a quick method for plotting the norm fiber lengths
def plotNormFiberLengthsPaths(path1, path2, tag1, tag2):
    # okay the goal for this function is to take a file or solution and plot the norm fiber lengths
    # analysisDir = './goodResults/'
    # analysisFile = 'quickAnalysis_' + tag1 + '_MuscleAnalysis_NormalizedFiberLength.sto'
    analysisFile = 'quickAnalysis_' + tag1 + '_Normalized_Fiber_Length.sto'


    fiberTable = osim.TimeSeriesTable(os.path.join(path1, analysisFile))
    time = fiberTable.getIndependentColumn()
    fibercols = fiberTable.getColumnLabels()
    numcols = len(fibercols)

        
    # if second tag, get second table
    if path2 != "":
        # analysisFile2 = 'quickAnalysis_' + tag2 + '_MuscleAnalysis_NormalizedFiberLength.sto'
        analysisFile2 = 'quickAnalysis_' + tag2 + '_Normalized_Fiber_Length.sto'
        fiberTable2 = osim.TimeSeriesTable(os.path.join(path2, analysisFile2))
        time2 = fiberTable2.getIndependentColumn()
        fibercols2 = fiberTable2.getColumnLabels()
        numcols2 = len(fibercols2)
    
    # want a 3x6 plot
    fig1, ax1 = plt.subplots(3, 6, figsize=(15,7))
    for i, ax in enumerate(fig1.axes):
        # get a column and plot it. 
        tempcol = fiberTable.getDependentColumn(fibercols[i]).to_numpy()
        ax.plot(time, tempcol, label=tag1)
        
        # second curve?
        if tag2 != "":
            tempcol2 = fiberTable2.getDependentColumn(fibercols[i]).to_numpy()
            ax.plot(time2, tempcol2, label=tag2)
        
        if i > 11:
            ax.set_xlabel('time(s)')
        if i==0 or i==6 or i==12:
            ax.set_ylabel('Norm. Fiber length')

        # print(fibercols[i])
        tempname = fibercols[i].split('|')
        tempname2 = tempname[0].split('/')

        # print(tempname2)
        # pdb.set_trace()
        ax.title.set_text(tempname2[2]) # fibercols[i]
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

    return

# create a quick method for plotting the norm fiber lengths
def plotNormFiberLengths(tag1, tag2):
    # okay the goal for this function is to take a file or solution and plot the norm fiber lengths
    analysisDir = './analysesTools/'
    analysisFile = 'quickAnalysis_' + tag1 + '_MuscleAnalysis_NormalizedFiberLength.sto'

    fiberTable = osim.TimeSeriesTable(os.path.join(analysisDir, analysisFile))
    time = fiberTable.getIndependentColumn()
    fibercols = fiberTable.getColumnLabels()
    numcols = len(fibercols)

        
        # if second tag, get second table
    if tag2 != "":
        analysisFile2 = 'quickAnalysis_' + tag2 + '_MuscleAnalysis_NormalizedFiberLength.sto'
        fiberTable2 = osim.TimeSeriesTable(os.path.join(analysisDir, analysisFile2))
        time2 = fiberTable2.getIndependentColumn()
        fibercols2 = fiberTable2.getColumnLabels()
        numcols2 = len(fibercols2)
    
    # want a 3x6 plot
    fig1, ax1 = plt.subplots(3, 6, figsize=(15,7))
    for i, ax in enumerate(fig1.axes):
        # get a column and plot it. 
        tempcol = fiberTable.getDependentColumn(fibercols[i]).to_numpy()
        ax.plot(time, tempcol, label=tag1)
        
        # second curve?
        if tag2 != "":
            tempcol2 = fiberTable2.getDependentColumn(fibercols[i]).to_numpy()
            ax.plot(time2, tempcol2, label=tag2)
        
        if i > 11:
            ax.set_xlabel('time(s)')
        if i==0 or i==6 or i==12:
            ax.set_ylabel('Norm. Fiber length')
        ax.title.set_text(fibercols[i])
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

    return

# create a plot of active and passive muscle forces (and exotendon tension)
def plotMuscleForces(arg1, arg2, arg3, arg4, argtitle, opt_arg5=None, opt_arg6=None):
    table1_active = osim.TimeSeriesTable(arg1)
    table2_passive = osim.TimeSeriesTable(arg2)

    times1_active = np.array(table1_active.getIndependentColumn())
    times1 = np.zeros([len(times1_active), 1])
    for i in range(len(times1_active)):
        times1[i] = times1_active[i]
    timespercent1 = (times1 - times1[0]) / (times1[-1] - times1[0]) * 100
    timespercent101 = np.linspace(0,100,101).reshape((101,1))
    numcols1 = table1_active.getNumColumns()
    labels1 = table1_active.getColumnLabels()
    labels2 = table2_passive.getColumnLabels()
    d1 = {}
    d2 = {}
    # loop through and get all of the data
    for i in range(len(labels1)):
        temp1 = table1_active.getDependentColumn(labels1[i]).to_numpy().reshape((len(times1_active),1))
        temp2 = table2_passive.getDependentColumn(labels2[i]).to_numpy().reshape((len(times1_active),1))
        # interpolate 
        if len(timespercent1) != len(timespercent101):
            try:
                tempinterp1 = np.interp(timespercent101, timespercent1, temp1) # 
                tempinterp2 = np.interp(timespercent101, timespercent1, temp2)
            except:
                tempinterp1 = np.interp(timespercent101.flatten(), timespercent1.flatten(), temp1.flatten()) # 
                tempinterp2 = np.interp(timespercent101.flatten(), timespercent1.flatten(), temp2.flatten())
        else:
            tempinterp1 = temp1
            tempinterp2 = temp2
        # add to the dictionary
        d1[labels1[i]] = tempinterp1
        d2[labels1[i]] = tempinterp2
    
    if opt_arg5 != None:
        table5_hobl = osim.TimeSeriesTable(opt_arg5)
        temp5 = table5_hobl.getDependentColumn('/forceset/HOBL|tension').to_numpy().reshape((len(times1_active),1))
        time5 = table5_hobl.getIndependentColumn()
        if len(timespercent1) != len(timespercent101):
            try:
                tempinterp5 = np.interp(timespercent101, timespercent1, temp5)
            except:
                tempinterp5 = np.interp(timespercent101.flatten(), timespercent1.flatten(), temp5.flatten())
        else:
            tempinterp5 = temp5
        d5 = {}
        d5['tension'] = tempinterp5

    
    table3_active = osim.TimeSeriesTable(arg3)
    table4_passive = osim.TimeSeriesTable(arg4)
    
    times3_active = np.array(table3_active.getIndependentColumn())
    times3 = np.zeros([len(times3_active), 1])
    for i in range(len(times3_active)):
        times3[i] = times3_active[i]
    timespercent3 = (times3 - times3[0]) / (times3[-1] - times3[0]) * 100
    timespercent301 = np.linspace(0,100,101).reshape((101,1))
    numcols3 = table3_active.getNumColumns()
    labels3 = table3_active.getColumnLabels()
    labels4 = table4_passive.getColumnLabels()
    d3 = {}
    d4 = {}
    # loop through and get all of the data
    for i in range(len(labels3)):
        temp3 = table3_active.getDependentColumn(labels3[i]).to_numpy().reshape((len(times3_active),1))
        temp4 = table4_passive.getDependentColumn(labels4[i]).to_numpy().reshape((len(times3_active),1))
        # interpolate 
        if len(timespercent3) != len(timespercent301):
            try:
                tempinterp3 = np.interp(timespercent301, timespercent3, temp3)
                tempinterp4 = np.interp(timespercent301, timespercent3, temp4)
            except:
                tempinterp3 = np.interp(timespercent301.flatten(), timespercent3.flatten(), temp3.flatten())
                tempinterp4 = np.interp(timespercent301.flatten(), timespercent3.flatten(), temp4.flatten())
        else:
            tempinterp3 = temp3
            tempinterp4 = temp4
        # add to the dictionary
        d3[labels3[i]] = tempinterp3
        d4[labels3[i]] = tempinterp4
    
    if opt_arg6 != None:
        table6_hobl = osim.TimeSeriesTable(opt_arg6)
        temp6 = table6_hobl.getDependentColumn('/forceset/HOBL|tension').to_numpy().reshape((len(times3_active),1))
        time6 = table6_hobl.getIndependentColumn()
        if len(timespercent3) != len(timespercent301):
            try:
                tempinterp6 = np.interp(timespercent301, timespercent3, temp6)
            except:
                tempinterp6 = np.interp(timespercent301.flatten(), timespercent3.flatten(), temp6.flatten())
        else:
            tempinterp6 = temp6
        d6 = {}
        d6['tension'] = tempinterp6

    # plot the data
    # now probably do some plotting with each of these to compare passive and active forces to what we might expect. 
    fig_size = (15, 8)
    dpi = 300
    # create the figure
    fig1, ax1 = plt.subplots(nrows=4, ncols=5, figsize=fig_size, dpi=dpi)
    musclekeys = list(d1.keys())
    
    # Loop over the axes array and create a plot in each subplot
    for i, ax in enumerate(ax1.flatten()):
        # y = np.sin(x + i)  # Example: different sine wave for each subplot
        if i < 18:
            y_p = d2[musclekeys[i]]
            y_a = d1[musclekeys[i]]
            z_p = d4[musclekeys[i]]
            z_a = d3[musclekeys[i]]
            # plot each of the curves
            ax.plot(timespercent101, y_p, label='T1 Passive', linestyle='-', color='#fdb863')
            ax.plot(timespercent101, y_a, label='T1 Active', linestyle='-', color='#e66101')
            ax.plot(timespercent301, z_p, label='T2 Passive', linestyle='-', color='#b2abd2')
            ax.plot(timespercent301, z_a, label='T2 Active', linestyle='-', color='#5e3c99')
            ax.set_title(musclekeys[i], fontsize=12)
            ax.set_xlabel('GC%', fontsize=12)
            ax.set_ylabel('Force [N]', fontsize=12)
        elif i == 18 and opt_arg5 != None:
            y_h = d5['tension']
            ax.plot(timespercent101, y_h, label='T1 Exo Tension', linestyle='-', color='#e66101')
            if opt_arg6 != None:
                z_h = d6['tension']
                ax.plot(timespercent301, z_h, label='T2 Exo Tension', linestyle='-', color='#5e3c99')
            ax.set_title('Exotendon Tension', fontsize=12)
            ax.set_xlabel('GC%', fontsize=12)
            ax.set_ylabel('Force [N]', fontsize=12)
        else:
            y_p = d2[musclekeys[0]] 
            y_a = d1[musclekeys[0]]
            z_p = d4[musclekeys[0]] 
            z_a = d3[musclekeys[0]]
            ax.plot(timespercent101, y_p, label='T1 Passive', linestyle='-', color='#fdb863')
            ax.plot(timespercent101, y_a, label='T1 Active', linestyle='-', color='#e66101')
            ax.plot(timespercent301, z_p, label='T2 Passive', linestyle='-', color='#b2abd2')
            ax.plot(timespercent301, z_a, label='T2 Active', linestyle='-', color='#5e3c99')
            ax.set_title(musclekeys[0], fontsize=12)
            ax.set_xlabel('GC%', fontsize=12)
            ax.set_ylabel('Force [N]', fontsize=12)
            ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # figurePath = localDir + '\\analysesTools\\mk13_analysis_figs\\'
    plt.savefig(argtitle)
    # Show the plot
    # plt.show()
    return

# setting up file name based on trial and speed
def setupNames(arg1, arg3, beginName): 
    speed1 = arg1.split('/')[2]
    speed3 = arg3.split('/')[2]
    trial1 = arg1.split('/')[-2]
    trial3 = arg3.split('/')[-2]
    testname = beginName + '_' + speed1 + '_' + trial1 + '_' + speed3 + '_' +  trial3 + '.png'
    return testname

# extract and print the values of any optimized parameters from the solution file
def parameterOptExtract(solutionFile, paramNames):
    tempsolution = osim.TimeSeriesTable(solutionFile)
    for each in paramNames:
        tempparam = tempsolution.getDependentColumn(each).to_numpy()
        tempparam = tempparam[~np.isnan(tempparam)]
        print(each + ' :: ' + str(tempparam))
    return

# function for doing a quick analysis of any input scripts (metabolics etc.)
def quickAnalyze(solutionFile, modelFile, grfFile, tag):

    # solutionFile = '3Act_2D3D_OG_muscles_Tracking_solution_FullStride.sto'
    # modelFile = '2DMuscles_OG_complextoes_3Dsphere_arms2D - Copy_stiff.osim'
    # statesFile = 'testingmetcoststates.sto'
    # controlsFile = 'testingmetcostControls.sto'
    # grfFile = 'grf_walk_nat_1.xml'
    # pdb.set_trace()

    analyze = osim.AnalyzeTool()
    analyze.setName('quickAnalysis_' + tag)

    solFileTrim = solutionFile[0:-4]
    statesFile = solFileTrim + '_states.sto'
    controlsFile = solFileTrim + '_controls.sto'

    sol = osim.MocoTrajectory(solutionFile)
    osim.STOFileAdapter.write(sol.exportToStatesTable(), statesFile)
    osim.STOFileAdapter.write(sol.exportToControlsTable(), controlsFile)

    statesStorage = osim.Storage(statesFile)
    analyze.setStatesStorage(statesStorage)
    analyze.updControllerSet().cloneAndAppend(osim.PrescribedController(controlsFile))

    # create analysis objects
    muscA = osim.MuscleAnalysis()
    probA = osim.ProbeReporter()
    forcA = osim.ForceReporter()
    bodyA = osim.BodyKinematics()
    joinA = osim.JointReaction()


    joinA.setName('jra')
    wherestr = osim.ArrayStr(); wherestr.append('child')
    joinA.setInFrame(wherestr)
    # add the analyses that we want
    analyze.updAnalysisSet().cloneAndAppend(muscA)
    analyze.updAnalysisSet().cloneAndAppend(probA)
    analyze.updAnalysisSet().cloneAndAppend(forcA)
    analyze.updAnalysisSet().cloneAndAppend(bodyA)
    analyze.updAnalysisSet().cloneAndAppend(joinA)

    # get times from the solution/states files
    states = osim.TimeSeriesTable(statesFile)
    time = states.getIndependentColumn()
    analyze.setInitialTime(time[0])
    analyze.setFinalTime(time[-1])
    analyze.setResultsDir('./analysesTools/')

    modelprocessor = osim.ModelProcessor(modelFile)
    if grfFile != "":
        modelprocessor.append(osim.ModOpAddExternalLoads(grfFile))
    
    model = modelprocessor.process()
    model.addAnalysis(muscA)
    model.addAnalysis(probA)
    model.addAnalysis(forcA)
    model.addAnalysis(bodyA)
    model.addAnalysis(joinA)
    
    analyze.setModel(model)
    # analyze.printToXML('testingmetcostanalyze.xml')

    analyze.run()    

# Updating the muscle ratio values.  = 
def getMuscleFiberRatios(muscleName, short):
    muscleKeys = ['addbrev_r',
                    'addlong_r',
                    'addmagDist_r',
                    'addmagIsch_r',
                    'addmagMid_r',
                    'addmagProx_r',
                    'bflh_r',
                    'bfsh_r',
                    'edl_r',
                    'ehl_r',
                    'fdl_r',
                    'fhl_r',
                    'gaslat_r',
                    'gasmed_r',
                    'glmax1_r',
                    'glmax2_r',
                    'glmax3_r',
                    'glmed1_r',
                    'glmed2_r',
                    'glmed3_r',
                    'glmin1_r',
                    'glmin2_r',
                    'glmin3_r',
                    'grac_r',
                    'iliacus_r',
                    'perbrev_r',
                    'perlong_r',
                    'piri_r',
                    'psoas_r',
                    'recfem_r',
                    'sart_r',
                    'semimem_r',
                    'semiten_r',
                    'soleus_r',
                    'tfl_r',
                    'tibant_r',
                    'tibpost_r',
                    'vasint_r',
                    'vaslat_r',
                    'vasmed_r',
                    'addbrev_l',
                    'addlong_l',
                    'addmagDist_l',
                    'addmagIsch_l',
                    'addmagMid_l',
                    'addmagProx_l',
                    'bflh_l',
                    'bfsh_l',
                    'edl_l',
                    'ehl_l',
                    'fdl_l',
                    'fhl_l',
                    'gaslat_l',
                    'gasmed_l',
                    'glmax1_l',
                    'glmax2_l',
                    'glmax3_l',
                    'glmed1_l',
                    'glmed2_l',
                    'glmed3_l',
                    'glmin1_l',
                    'glmin2_l',
                    'glmin3_l',
                    'grac_l',
                    'iliacus_l',
                    'perbrev_l',
                    'perlong_l',
                    'piri_l',
                    'psoas_l',
                    'recfem_l',
                    'sart_l',
                    'semimem_l',
                    'semiten_l',
                    'soleus_l',
                    'tfl_l',
                    'tibant_l',
                    'tibpost_l',
                    'vasint_l',
                    'vaslat_l',
                    'vasmed_l'];

    muscleKeys_long = ['add_brev_r',
                    'add_long_r',
                    'add_mag3_r',
                    'add_mag4_r',
                    'add_mag2_r',
                    'add_mag1_r',
                    'bifemlh_r',
                    'bifemsh_r',
                    'ext_dig_r',
                    'ext_hal_r',
                    'flex_dig_r',
                    'flex_hal_r',
                    'lat_gas_r',
                    'med_gas_r',
                    'glut_max1_r',
                    'glut_max2_r',
                    'glut_max3_r',
                    'glut_med1_r',
                    'glut_med2_r',
                    'glut_med3_r',
                    'glut_min1_r',
                    'glut_min2_r',
                    'glut_min3_r',
                    'grac_r',
                    'iliacus_r',
                    'per_brev_r',
                    'per_long_r',
                    'peri_r',
                    'psoas_r',
                    'rect_fem_r',
                    'sar_r',
                    'semimem_r',
                    'semiten_r',
                    'soleus_r',
                    'tfl_r',
                    'tib_ant_r',
                    'tib_post_r',
                    'vas_int_r',
                    'vas_lat_r',
                    'vas_med_r',
                    'add_brev_l',
                    'add_long_l',
                    'add_mag3_l',
                    'add_mag4_l',
                    'add_mag2_l',
                    'add_mag1_l',
                    'bifemlh_l',
                    'bifemsh_l',
                    'ext_dig_l',
                    'ext_hal_l',
                    'flex_dig_l',
                    'flex_hal_l',
                    'lat_gas_l',
                    'med_gas_l',
                    'glut_max1_l',
                    'glut_max2_l',
                    'glut_max3_l',
                    'glut_med1_l',
                    'glut_med2_l',
                    'glut_med3_l',
                    'glut_min1_l',
                    'glut_min2_l',
                    'glut_min3_l',
                    'grac_l',
                    'iliacus_l',
                    'per_brev_l',
                    'per_long_l',
                    'peri_l',
                    'psoas_l',
                    'rect_fem_l',
                    'sar_l',
                    'semimem_l',
                    'semiten_l',
                    'soleus_l',
                    'tfl_l',
                    'tib_ant_l',
                    'tib_post_l',
                    'vas_int_l',
                    'vas_lat_l',
                    'vas_med_l'];

    muscleKeys_short = ['hamstrings_r',
                        'bifemsh_r',
                        'glut_max_r',
                        'iliopsoas_r',
                        'rect_fem_r',
                        'vasti_r',
                        'gastroc_r',
                        'soleus_r',
                        'tib_ant_r',
                        'hamstrings_l',
                        'bifemsh_l',
                        'glut_max_l',
                        'iliopsoas_l',
                        'rect_fem_l',
                        'vasti_l',
                        'gastroc_l',
                        'soleus_l',
                        'tib_ant_l'];

    muscleKeys_short2 = ['semimem_r',
                        'bfsh_r',
                        'glmax2_r',
                        'psoas_r',
                        'recfem_r',
                        'vasint_r',
                        'gasmed_r',
                        'soleus_r',
                        'tibant_r',
                        'semimem_l',
                        'bfsh_l',
                        'glmax2_l',
                        'psoas_l',
                        'recfem_l',
                        'vasint_l',
                        'gasmed_l',
                        'soleus_l',
                        'tibant_l'];


    ratioVals_short = [0.49249,
                    0.5290,
                    0.55,
                    0.5,
                    0.3865,
                    0.455,
                    0.53,
                    0.8035,
                    0.6999,
                    0.49249,
                    0.5290,
                    0.55,
                    0.5,
                    0.3865,
                    0.455,
                    0.53,
                    0.8035,
                    0.6999];


    ratioVals = [0.5,
        0.5,
        0.55200000000000005,
        0.55200000000000005,
        0.55200000000000005,
        0.55200000000000005,
        0.54249999999999998,
        0.52900000000000003,
        0.75,
        0.75,
        0.59999999999999998,
        0.59999999999999998,
        0.50700000000000001,
        0.56599999999999995,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.5,
        0.5,
        0.59999999999999998,
        0.59999999999999998,
        0.5,
        0.5,
        0.38650000000000001,
        0.5,
        0.49249999999999999,
        0.42499999999999999,
        0.80350000000000005,
        0.5,
        0.69999999999999996,
        0.59999999999999998,
        0.54350000000000004,
        0.45500000000000002,
        0.503,
        0.5,
        0.5,
        0.55200000000000005,
        0.55200000000000005,
        0.55200000000000005,
        0.55200000000000005,
        0.54249999999999998,
        0.52900000000000003,
        0.75,
        0.75,
        0.59999999999999998,
        0.59999999999999998,
        0.50700000000000001,
        0.56599999999999995,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.55000000000000004,
        0.5,
        0.5,
        0.59999999999999998,
        0.59999999999999998,
        0.5,
        0.5,
        0.38650000000000001,
        0.5,
        0.49249999999999999,
        0.42499999999999999,
        0.80350000000000005,
        0.5,
        0.69999999999999996,
        0.59999999999999998,
        0.54350000000000004,
        0.45500000000000002,
        0.503];

    # now just index the list of names for the muscle we want
    # and use th index to get the right ratio value. 
    if short == 'short2': 
        for i in range(len(muscleKeys_short2)):
            if muscleKeys_short2[i] in muscleName:
                tempratio = ratioVals_short[i]
                # print(muscleKeys_short[i])
                # print(tempratio)            
    if short == 'short': 
        for i in range(len(muscleKeys_short)):
            if muscleKeys_short[i] in muscleName:
                tempratio = ratioVals_short[i]
                # print(muscleKeys_short[i])
                # print(tempratio)            
    if short == 'long':
        for i in range(len(muscleKeys)):
            if muscleKeys[i] in muscleName:
                tempratio = ratioVals[i]
                # print(muscleKeys_short[i])
                # print(tempratio

    return tempratio

##TODO: finish this - not even tested once
# creating a version of the function to compute and save the contact forces from the simulation 
def create_external_loads_table_for_gait(model, trajectory, force_paths_right_foot, force_paths_left_foot):
    model.initSystem()
    external_forces_table = osim.TimeSeriesTableVec3()
    count = 0
    
    pdb.set_trace()
    numStates = trajectory.getNumRows()

    for state in trajectory:
        model.realizeDynamics(state)
        
        sphere_forces_right = osim.Vec3(0)
        sphere_torques_right = osim.Vec3(0)
        half_space_forces_right = osim.Vec3(0)
        half_space_torques_right = osim.Vec3(0)
        
        for smooth_force in force_paths_right_foot:
            force_values = model.getComponent('Force', smooth_force).getRecordValues(state)
            sphere_forces_right += osim.Vec3(force_values[0], force_values[1], force_values[2])
            sphere_torques_right += osim.Vec3(force_values[3], force_values[4], force_values[5])
            half_space_forces_right += osim.Vec3(force_values[6], force_values[7], force_values[8])
            half_space_torques_right += osim.Vec3(force_values[9], force_values[10], force_values[11])
        
        sphere_forces_left = osim.Vec3(0)
        sphere_torques_left = osim.Vec3(0)
        half_space_forces_left = osim.Vec3(0)
        half_space_torques_left = osim.Vec3(0)
        
        for smooth_force in force_paths_left_foot:
            force_values = model.getComponent('Force', smooth_force).getRecordValues(state)
            sphere_forces_left += osim.Vec3(force_values[0], force_values[1], force_values[2])
            sphere_torques_left += osim.Vec3(force_values[3], force_values[4], force_values[5])
            half_space_forces_left += osim.Vec3(force_values[6], force_values[7], force_values[8])
            half_space_torques_left += osim.Vec3(force_values[9], force_values[10], force_values[11])
        
        cop_right = osim.Vec3(0)
        cop_right[0] = half_space_torques_right[2] / half_space_forces_right[1]
        cop_right[2] = -half_space_torques_right[0] / half_space_forces_right[1]
        
        cop_left = osim.Vec3(0)
        cop_left[0] = half_space_torques_left[2] / half_space_forces_left[1]
        cop_left[2] = -half_space_torques_left[0] / half_space_forces_left[1]
        
        row = osim.RowVectorVec3(6)
        row[0] = sphere_forces_right
        row[1] = cop_right
        row[2] = sphere_forces_left
        row[3] = cop_left
        row[4] = sphere_torques_right
        row[5] = sphere_torques_left
        
        external_forces_table.appendRow(state.getTime(), row)
        count += 1
    
    labels = ["ground_force_r_v", "ground_force_r_p", "ground_force_l_v", "ground_force_l_p", "ground_torque_r_", "ground_torque_l_"]
    external_forces_table.setColumnLabels(labels)
    external_forces_table_flat = external_forces_table.flatten(["x", "y", "z"])
    
    return external_forces_table_flat

# compute the error of times for the simulation
def timeErr(x, y):
    err = abs(y - x)
    return err

# compute error for the various signals of the simulation
def dtw_rmse(x, y, indegrees):
    if indegrees:
        # have to convert y to rad, which is what x will be in
        y = y * np.pi / 180
    # sim results are x, and IK or comparison is y
    x_new = np.interp(np.linspace(0,100,len(y)), np.linspace(0,100,len(x)), x)
    rmse = np.sqrt(np.mean((x_new - y)**2))
    # now compute a normalized to one version. 
    normalized_x = x_new / np.max(np.abs(y))
    normalized_y = y / np.max(np.abs(y))
    normalized_rmse = np.sqrt(np.mean((normalized_x - normalized_y)**2))

    return 100*normalized_rmse

    ## this is all old methods. leaving for now.   
        # f = interp1d(np.arange(len(y)), y)
        # downsampling_indices = np.linspace(0,len(y) - 1, len1, dtype=int)
        # downsampled_vec = f(downsampling_indices)
        # y_new = downsampled_vec

        # xl1 = np.linalg.norm(x, ord=np.inf)
        # yl1 = np.linalg.norm(y_new, ord=np.inf)

        # x_norm = x / yl1
        # y_norm = y_new / yl1
        # len1 = len(x_norm)
        # len2 = len(y_norm)
        
        # yrange = np.max(y_new) - np.min(y_new)
        # xrange = np.max(x) - np.min(x)

        # if len1 < len2:
        #     kept_diff = np.zeros((len2,1))
        #     kept_diff2 = np.zeros((len2,1))    
        #     for i in range(len1):
        #         squared_diff += (x_norm[i] - y_norm[i]) ** 2
        #         kept_diff[i] = (x_norm[i] - y_norm[i]) ** 2

        #         squared_diff2 += (x[i] - y_new[i]) ** 2
        #         kept_diff2[i] = (x[i] - y_new[i]) ** 2

        #     for i in range(len1, len2):
        #         squared_diff += y_norm[i] ** 2  # Add remaining points of y
        #         kept_diff[i] = y_norm[i] ** 2

        #         squared_diff2 += y_new[i] ** 2  # Add remaining points of y
        #         kept_diff2[i] = y_new[i] ** 2

        # else:
        #     kept_diff = np.zeros((len1,1))
        #     kept_diff2 = np.zeros((len1,1))
        #     for i in range(len2):
        #         kept_diff[i] = (x_norm[i] - y_norm[i]) ** 2
        #         squared_diff += (x_norm[i] - y_norm[i]) ** 2

        #         kept_diff2[i] = (x[i] - y_new[i]) ** 2
        #         squared_diff2 += (x[i] - y_new[i]) ** 2
        #     for i in range(len2, len1):
        #         kept_diff[i] = x_norm[i] ** 2
        #         squared_diff += x_norm[i] ** 2  # Add remaining points of x

        #         kept_diff2[i] = x[i] ** 2
        #         squared_diff2 += x[i] ** 2  # Add remaining points of x

        # if np.isnan(squared_diff):
        #     return 0
        # else:
        #     return 10*squared_diff

# compute errors and plot the curves compared to experimental data
def coordSTDCompare(y_sim, y_ik, indegrees, stdmeasure, fig=None, ax=None, count=None):
    if indegrees:
        # have to convert y to rad, which is what x will be in
        y_ik = y_ik * np.pi / 180
        # stdmeasure = stdmeasure * np.pi / 180

    # okay y is the tracked or IK kinematics for this trial
    # x is the simulation kinematic curve. 
    # stdmeasure is the max standard deviation for this coordinate, as measured from the
    # whole dataset of participants in the main exotendon study

    # start with resampling to make the same length
    len1 = len(y_sim)
    len2 = len(y_ik)

    x_sim = np.linspace(0, 100, len1)
    x_ik = np.linspace(0, 100, len2)
    y_ik_new = np.interp(x_sim, x_ik, y_ik)
    stdmeasure_new = np.interp(x_sim, np.linspace(0,100,len(stdmeasure)), stdmeasure)

    # f = interp1d(np.arange(len(y)), y)
    # downsampling_indices = np.linspace(0,len(y) - 1, len1)
    # downsampled_vec = f(downsampling_indices)
    # y_new = downsampled_vec
    maxdiff = 0
    # pdb.set_trace()
    # loop through length
    outRange = np.zeros((len(y_ik_new),1))
    for i in range(len(y_ik_new)):
        # for each time point see if the value is within 2 std of the input kinematics
        diff = abs(y_sim[i] - y_ik_new[i])
        if diff > maxdiff:
            maxdiff = diff
        # if it is not then we need to flag it!
        if diff > stdmeasure_new[i]:
            # print('we got one')
            outRange[i] = 1
        else:
            outRange[i] = 0
    if fig == None:
        return [np.sum(outRange), maxdiff, stdmeasure_new]
    else:
        # now we have a vector of trues and falses representing when the vector exceeds 2std of the input
        ax[count].plot(x_sim, y_ik_new, label='ik/id', color='orange')
        ax[count].plot(x_sim, y_sim, label='simulation', color='blue')
        # xaxis = np.linspace(0,100,len(y_new)); 
        y_minus = y_ik_new - stdmeasure_new
        y_plus = y_ik_new + stdmeasure_new
        ax[count].fill_between(x_sim, y_minus, y_plus, color='orange', alpha=0.2, label='2std from dataset')
        # ax[count].legend()
        # pdb.set_trace()
        return [np.sum(outRange), maxdiff, stdmeasure_new, fig, ax] 

# compute and visualize differences in the grf between twto trials
def grfCompare(grf1, grf2, ref1, ref2, sim, prevstd_nat, prevstd_exo):
    subjectmass = 73.48
    if sim=='CMA':
        # load the GRF files
        grf1table = osim.TimeSeriesTable(grf1)
        grf2table = osim.TimeSeriesTable(grf2)
        ref1table = osim.TimeSeriesTable(ref1)  # './expdata/nat50_1.mot')
        ref2table = osim.TimeSeriesTable(ref2)  # './expdata/exo50_1.mot')
        # grfDevs_nat = pd.read_csv('std_externalForces_nat.csv')
        # grfDevs_both = pd.read_csv('std_externalForces_both.csv')
        # grfDevs_exo = pd.read_csv('std_externalForces_exo.csv')
        grfDevs_nat = prevstd_nat
        grfDevs_exo = prevstd_exo

        # get the x and y for each 
        grf1x = grf1table.getDependentColumn('ground_force_r_vx').to_numpy()
        grf1y = grf1table.getDependentColumn('ground_force_r_vy').to_numpy()
        grf2x = grf2table.getDependentColumn('ground_force_r_vx').to_numpy()
        grf2y = grf2table.getDependentColumn('ground_force_r_vy').to_numpy()
        try: 
            ref1x = ref1table.getDependentColumn('rF_x').to_numpy()
            ref1y = ref1table.getDependentColumn('rF_y').to_numpy()
            ref2x = ref2table.getDependentColumn('rF_x').to_numpy()
            ref2y = ref2table.getDependentColumn('rF_y').to_numpy()
        except:
            ref1x = ref1table.getDependentColumn('ground_force_r_vx').to_numpy()
            ref1y = ref1table.getDependentColumn('ground_force_r_vy').to_numpy()
            ref2x = ref2table.getDependentColumn('ground_force_r_vx').to_numpy()
            ref2y = ref2table.getDependentColumn('ground_force_r_vy').to_numpy()
        # get time for each 
        grf1time = grf1table.getIndependentColumn()
        grf2time = grf2table.getIndependentColumn()
        ref1time = ref1table.getIndependentColumn()
        ref2time = ref2table.getIndependentColumn()
        # resize everything if needed
        xdes_ref1 = np.linspace(ref1time[0],ref1time[-1],101)
        xdes_ref2 = np.linspace(ref2time[0],ref2time[-1],101)
        xdes_grf1 = np.linspace(grf1time[0],grf1time[-1],101)
        xdes_grf2 = np.linspace(grf2time[0],grf2time[-1],101)
        grf1x = np.interp(xdes_grf1, grf1time, grf1x)
        grf1y = np.interp(xdes_grf1, grf1time, grf1y)
        grf2x = np.interp(xdes_grf2, grf2time, grf2x)
        grf2y = np.interp(xdes_grf2, grf2time, grf2y)
        ref1x = np.interp(xdes_ref1, ref1time, ref1x)
        ref1y = np.interp(xdes_ref1, ref1time, ref1y)
        ref2x = np.interp(xdes_ref2, ref2time, ref2x)
        ref2y = np.interp(xdes_ref2, ref2time, ref2y)

        # normalize to the body weight. 
        grf1y = grf1y / (subjectmass*9.81)
        grf2y = grf2y / (subjectmass*9.81)
        ref1y = ref1y / (subjectmass*9.81)
        ref2y = ref2y / (subjectmass*9.81)
        grf1x = grf1x / (subjectmass*9.81)
        grf2x = grf2x / (subjectmass*9.81)
        ref1x = ref1x / (subjectmass*9.81)
        ref2x = ref2x / (subjectmass*9.81)

        # get reference upper and lower 2 std bounds
        ref1yminus = ref1y - np.interp(xdes_ref1, np.linspace(ref1time[0], ref1time[-1], len(grfDevs_nat['calcn_r_Right_GRF_Fy'])), grfDevs_nat['calcn_r_Right_GRF_Fy']) 
        ref1yplus = ref1y + np.interp(xdes_ref1, np.linspace(ref1time[0], ref1time[-1], len(grfDevs_nat['calcn_r_Right_GRF_Fy'])), grfDevs_nat['calcn_r_Right_GRF_Fy'])
        ref1xminus = ref1x - np.interp(xdes_ref1, np.linspace(ref1time[0], ref1time[-1], len(grfDevs_nat['calcn_r_Right_GRF_Fx'])), grfDevs_nat['calcn_r_Right_GRF_Fx'])
        ref1xplus = ref1x + np.interp(xdes_ref1, np.linspace(ref1time[0], ref1time[-1], len(grfDevs_nat['calcn_r_Right_GRF_Fx'])), grfDevs_nat['calcn_r_Right_GRF_Fx'])    
        ref2yminus = ref2y - np.interp(xdes_ref2, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fy'])), grfDevs_exo['calcn_r_Right_GRF_Fy'])
        ref2yplus = ref2y + np.interp(xdes_ref2, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fy'])), grfDevs_exo['calcn_r_Right_GRF_Fy'])
        ref2xminus = ref2x - np.interp(xdes_ref2, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fx'])), grfDevs_exo['calcn_r_Right_GRF_Fx'])
        ref2xplus = ref2x + np.interp(xdes_ref2, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fx'])), grfDevs_exo['calcn_r_Right_GRF_Fx'])
    
        # plot the forces together, as well as the references. 
        fig, ax = plt.subplots(1,2, figsize=(12,6), dpi=300) # , dpi=300
        ax = ax.flatten()
        # put in the std for the references first. 
        ax[0].fill_between(xdes_ref1, ref1yminus, ref1yplus, color='orange', alpha=0.2, label='1std (previous simulations)')
        ax[0].fill_between(xdes_ref2, ref2yminus, ref2yplus, color='purple', alpha=0.2, label='1std (previous simulations)')    
        ax[0].plot(xdes_ref1, ref1y, label='nat', color='orange')
        ax[0].plot(xdes_ref2, ref2y, label='exo', color='purple')
        ax[0].plot(xdes_grf1, grf1y, label='sim nat', color='orange', linestyle='--', linewidth=2)
        ax[0].plot(xdes_grf2, grf2y, label='sim exo', color='purple', linestyle='--', linewidth=2)
        ax[0].set_title('GRF Y', fontsize=14)
        ax[0].set_xlabel('Gait Cycle (sec)', fontsize=14)
        ax[0].set_ylabel('Force (BW)', fontsize=14)
        ax[1].fill_between(xdes_ref1, ref1xminus, ref1xplus, color='orange', alpha=0.2, label='1std (previous simulations)')
        ax[1].fill_between(xdes_ref2, ref2xminus, ref2xplus, color='purple', alpha=0.2, label='1std (previous simulations)')
        ax[1].plot(xdes_ref1, ref1x, label='Exp. nat', color='orange')
        ax[1].plot(xdes_ref2, ref2x, label='Exp. exo', color='purple')
        ax[1].plot(xdes_grf1, grf1x, label='sim nat', color='orange', linestyle='--', linewidth=2)
        ax[1].plot(xdes_grf2, grf2x, label='sim exo', color='purple', linestyle='--', linewidth=2)
        ax[1].set_title('GRF X', fontsize=14)
        ax[1].set_xlabel('Gait Cycle (sec)', fontsize=14)
        ax[1].legend()
        plt.tight_layout()
        
        # compute some quick peak differences. 
        natRMSE = np.sqrt(np.mean((grf1y - ref1y)**2))
        print('\nRMSE for nat y GRF: ' + str(natRMSE))
        exoRMSE = np.sqrt(np.mean((grf2y - ref2y)**2))
        print('RMSE for exo y GRF: ' + str(exoRMSE))
        
        natRMASEx = np.sqrt(np.mean((grf1x - ref1x)**2))
        print('RMSE for nat x GRF: ' + str(natRMASEx))
        exoRMSEx = np.sqrt(np.mean((grf2x - ref2x)**2))
        print('RMSE for exo x GRF: ' + str(exoRMSEx))

        plt.show()
        
        # print('looking at the peak vertical values')
        highlight_text('looking at the peak vertical values')
        print('max ref nat y: ' + str(np.max(ref1y))) 
        print('max ref exo y: ' + str(np.max(ref2y)))
        refpeakdiff = np.max(ref1y) - np.max(ref2y)
        # print('peak difference between references: ' + str(refpeakdiff))
        highlight_text('peak difference between references: ' + str(refpeakdiff))
        print('\nmax sim nat y: ' + str(np.max(grf1y)))
        print('max sim exo y: ' + str(np.max(grf2y)))
        grfpeakdiff = np.max(grf1y) - np.max(grf2y)
        # print('peak difference between simulations: ' + str(grfpeakdiff))
        highlight_text('peak difference between simulations: ' + str(grfpeakdiff))

        natpeakdiff = np.max(ref1y) - np.max(grf1y)
        exopeakdiff = np.max(ref2y) - np.max(grf2y)
        print('\nerror between nat ref and sim peaks: ' + str(natpeakdiff))
        print('error between exo ref and sim peaks: ' + str(exopeakdiff) + '\n\n')

    if sim=='CMA_4':
        # load the GRF files
        grf1table = osim.TimeSeriesTable(grf1)
        grf2table = osim.TimeSeriesTable(grf2)
        ref1table = osim.TimeSeriesTable(ref1)  # './expdata/nat50_1.mot')
        # ref2table = osim.TimeSeriesTable(ref2)  # './expdata/exo50_1.mot')
        # grfDevs_nat = pd.read_csv('std_externalForces_nat.csv')
        # grfDevs_both = pd.read_csv('std_externalForces_both.csv')
        # grfDevs_exo = pd.read_csv('std_externalForces_exo.csv')
        grfDevs_nat = prevstd_nat
        # grfDevs_exo = prevstd_exo
        # get the x and y for each 
        grf1x = grf1table.getDependentColumn('ground_force_r_vx').to_numpy()
        grf1y = grf1table.getDependentColumn('ground_force_r_vy').to_numpy()
        grf2x = grf2table.getDependentColumn('ground_force_r_vx').to_numpy()
        grf2y = grf2table.getDependentColumn('ground_force_r_vy').to_numpy()
        try: 
            ref1x = ref1table.getDependentColumn('rF_x').to_numpy()
            ref1y = ref1table.getDependentColumn('rF_y').to_numpy()
            # ref2x = ref2table.getDependentColumn('rF_x').to_numpy()
            # ref2y = ref2table.getDependentColumn('rF_y').to_numpy()
        except:
            ref1x = ref1table.getDependentColumn('R_ground_force_vx').to_numpy()
            ref1y = ref1table.getDependentColumn('R_ground_force_vy').to_numpy()
            # ref1x = ref1table.getDependentColumn('ground_force_r_vx').to_numpy()
            # ref1y = ref1table.getDependentColumn('ground_force_r_vy').to_numpy()
            # ref2x = ref2table.getDependentColumn('ground_force_r_vx').to_numpy()
            # ref2y = ref2table.getDependentColumn('ground_force_r_vy').to_numpy()
        # get time for each 
        grf1time = grf1table.getIndependentColumn()
        grf2time = grf2table.getIndependentColumn()
        ref1time = ref1table.getIndependentColumn()
        # ref2time = ref2table.getIndependentColumn()
        # normalize all the curves
        grf1x = grf1x / (subjectmass*9.81)
        grf1y = grf1y / (subjectmass*9.81)
        grf2x = grf2x / (subjectmass*9.81)
        grf2y = grf2y / (subjectmass*9.81)
        ref1x = ref1x / (subjectmass*9.81)
        ref1y = ref1y / (subjectmass*9.81)

        # need to standardize the lengths of all the vectors for easy comparisons. 
        xdes_ref1 = np.linspace(0,100,len(grf1y))
        xdes_grf1 = np.linspace(0,100,len(grf1y))
        xdes_grf2 = np.linspace(0,100,len(grf1y))
        grf1x = np.interp(xdes_grf1, np.linspace(0,100, len(grf1time)), grf1x)
        grf1y = np.interp(xdes_grf1, np.linspace(0,100, len(grf1time)), grf1y)
        grf2x = np.interp(xdes_grf2, np.linspace(0,100, len(grf2time)), grf2x)
        grf2y = np.interp(xdes_grf2, np.linspace(0,100, len(grf2time)), grf2y)
        ref1x = np.interp(xdes_ref1, np.linspace(0,100, len(ref1time)), ref1x)
        ref1y = np.interp(xdes_ref1, np.linspace(0,100, len(ref1time)), ref1y)

        # get reference upper and lower 2 std bounds
        ref1yminus = ref1y - np.interp(xdes_ref1, np.linspace(0,100, len(grfDevs_nat['calcn_r_Right_GRF_Fy'])), grfDevs_nat['calcn_r_Right_GRF_Fy']) 
        ref1yplus = ref1y + np.interp(xdes_ref1, np.linspace(0, 100, len(grfDevs_nat['calcn_r_Right_GRF_Fy'])), grfDevs_nat['calcn_r_Right_GRF_Fy'])
        ref1xminus = ref1x - np.interp(xdes_ref1, np.linspace(0, 100, len(grfDevs_nat['calcn_r_Right_GRF_Fx'])), grfDevs_nat['calcn_r_Right_GRF_Fx'])
        ref1xplus = ref1x + np.interp(xdes_ref1, np.linspace(0, 100, len(grfDevs_nat['calcn_r_Right_GRF_Fx'])), grfDevs_nat['calcn_r_Right_GRF_Fx'])    
        # ref2yminus = ref2y - np.interp(ref2time, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fy'])), grfDevs_exo['calcn_r_Right_GRF_Fy'])
        # ref2yplus = ref2y + np.interp(ref2time, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fy'])), grfDevs_exo['calcn_r_Right_GRF_Fy'])
        # ref2xminus = ref2x - np.interp(ref2time, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fx'])), grfDevs_exo['calcn_r_Right_GRF_Fx'])
        # ref2xplus = ref2x + np.interp(ref2time, np.linspace(ref2time[0], ref2time[-1], len(grfDevs_exo['calcn_r_Right_GRF_Fx'])), grfDevs_exo['calcn_r_Right_GRF_Fx'])
        
        # plot the forces together, as well as the references. 
        fig, ax = plt.subplots(1,2, figsize=(12,6), dpi=300) # , dpi=300
        ax = ax.flatten()
        # put in the std for the references first. 
        # ax[0].fill_between(ref1time, ref1yminus, ref1yplus, color='orange', alpha=0.2, label='2std from nat')
        # ax[0].fill_between(ref2time, ref2yminus, ref2yplus, color='purple', alpha=0.2, label='2std from exo')    
        ax[0].plot(xdes_ref1, ref1y, label='nat', color='orange')
        # ax[0].plot(ref2time, ref2y, label='exo', color='purple')
        ax[0].plot(xdes_grf1, grf1y, label='sim nat', color='orange', linestyle='--', linewidth=2)
        ax[0].plot(xdes_grf2, grf2y, label='sim exo', color='purple', linestyle='--', linewidth=2)
        ax[0].set_title('GRF Y', fontsize=14)
        ax[0].set_xlabel('Gait Cycle (sec)', fontsize=14)
        ax[0].set_ylabel('Force (N)', fontsize=14)
        # ax[1].fill_between(ref1time, ref1xminus, ref1xplus, color='orange', alpha=0.2, label='2std from nat')
        # ax[1].fill_between(ref2time, ref2xminus, ref2xplus, color='purple', alpha=0.2, label='2std from exo')
        ax[1].plot(xdes_ref1, ref1x, label='Exp. nat', color='orange')
        # ax[1].plot(ref2time, ref2x, label='exo', color='purple')
        ax[1].plot(xdes_grf1, grf1x, label='sim nat', color='orange', linestyle='--', linewidth=2)
        ax[1].plot(xdes_grf2, grf2x, label='sim exo', color='purple', linestyle='--', linewidth=2)
        ax[1].set_title('GRF X', fontsize=14)
        ax[1].set_xlabel('Gait Cycle (sec)', fontsize=14)
        ax[1].legend()
        plt.tight_layout()

        # compute some quick peak differences. 
        print('looking at the peak vertical values')
        print('max ref nat y: ' + str(np.max(ref1y))) 
        # print('max ref exo y: ' + str(np.max(ref2y)))
        # refpeakdiff = np.max(ref1y) - np.max(ref2y)
        # print('peak difference between references: ' + str(refpeakdiff))
        print('\nmax sim nat y: ' + str(np.max(grf1y)))
        print('max sim exo y: ' + str(np.max(grf2y)))
        grfpeakdiff = np.max(grf1y) - np.max(grf2y)
        print('peak difference between simulations: ' + str(grfpeakdiff))

        natpeakdiff = np.max(ref1y) - np.max(grf1y)
        # exopeakdiff = np.max(ref2y) - np.max(grf2y)
        print('\nerror between nat ref and sim peaks: ' + str(natpeakdiff))
        # print('error between exo ref and sim peaks: ' + str(exopeakdiff))

        natRMSE = np.sqrt(np.mean((grf1y - ref1y)**2))
        print('\nRMSE for nat y GRF: ' + str(natRMSE))
        # exoRMSE = np.sqrt(np.mean((grf2y - ref2y)**2))
        # print('RMSE for exo y GRF: ' + str(exoRMSE))
        
        natRMASEx = np.sqrt(np.mean((grf1x - ref1x)**2))
        print('RMSE for nat x GRF: ' + str(natRMASEx))
        # exoRMSEx = np.sqrt(np.mean((grf2x - ref2x)**2))
        # print('RMSE for exo x GRF: ' + str(exoRMSEx))

        plt.show()
    return 

# compute and visualize differences in metabolics between stance and swing
def metabolicStanceSwing(natmet, exomet, modelfile, GRFnat, GRFexo, sim):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the GRF results
    natGRF = osim.TimeSeriesTable(GRFnat)
    exoGRF = osim.TimeSeriesTable(GRFexo)
    # get the time vectors
    natGRFtime = np.array(natGRF.getIndependentColumn())
    exoGRFtime = np.array(exoGRF.getIndependentColumn())
    # get the vertical forces
    natGRFy = natGRF.getDependentColumn('ground_force_r_vy').to_numpy()
    exoGRFy = exoGRF.getDependentColumn('ground_force_r_vy').to_numpy()
    # get previous study data
    prevstance = pd.read_csv('stanceStudy1Savings.csv')
    prevswing = pd.read_csv('swingStudy1Savings.csv')    
    prevcycle = pd.read_csv('musclesStudy1Savings.csv')


    # Find the peak in the first half of the vector nat
    half_index_nat = len(natGRFy) // 2
    first_half_nat = natGRFy[:half_index_nat]
    peak_index_nat = np.argmax(first_half_nat)
    threshold = 0.5
    # Find the point where the vector first rises above the threshold before the peak
    start_of_hump_index_nat = np.where(natGRFy[:peak_index_nat] > threshold)[0][0]
    return_to_zero_index_nat = peak_index_nat + np.where(natGRFy[peak_index_nat:] < threshold)[0][0]
    # Split the vector into three parts
    part_before_hump_naty = natGRFy[:start_of_hump_index_nat]
    part_main_hump_naty = natGRFy[start_of_hump_index_nat:return_to_zero_index_nat + 1]
    part_after_hump_naty = natGRFy[return_to_zero_index_nat + 1:]
    part_before_hump_nattime = natGRFtime[:start_of_hump_index_nat]
    part_main_hump_nattime = natGRFtime[start_of_hump_index_nat:return_to_zero_index_nat + 1]
    part_after_hump_nattime = natGRFtime[return_to_zero_index_nat + 1:]

    # Find the peak in the first half of the vector exo
    half_index_exo = len(exoGRFy) // 2
    first_half_exo = exoGRFy[:half_index_exo]
    peak_index_exo = np.argmax(first_half_exo)
    threshold = 0.5
    # Find the point where the vector first rises above the threshold before the peak
    start_of_hump_index_exo = np.where(exoGRFy[:peak_index_exo] > threshold)[0][0]
    return_to_zero_index_exo = peak_index_exo + np.where(exoGRFy[peak_index_exo:] < threshold)[0][0]
    # Split the vector into three parts
    part_before_hump_exoy = exoGRFy[:start_of_hump_index_exo]
    part_main_hump_exoy = exoGRFy[start_of_hump_index_exo:return_to_zero_index_exo + 1]
    part_after_hump_exoy = exoGRFy[return_to_zero_index_exo + 1:]
    part_before_hump_exotime = exoGRFtime[:start_of_hump_index_exo]
    part_main_hump_exotime = exoGRFtime[start_of_hump_index_exo:return_to_zero_index_exo + 1]
    part_after_hump_exotime = exoGRFtime[return_to_zero_index_exo + 1:]
    # # Plotting the GRF for stance and swing phases   # load the metabolic results
    # fig, ax = plt.subplots(1, 3, figsize=(12, 6), dpi=300)
    # # Plotting the stance phase
    # ax[0].set_title('before Stance Phase')
    # ax[0].set_xlabel('Time (s)')
    # ax[0].set_ylabel('GRF (N)')
    # ax[0].legend(); 
    # ax[0].plot(part_before_hump_nattime, part_before_hump_naty, label='nat', color='orange')
    # ax[0].plot(part_before_hump_exotime, part_before_hump_exoy, label='exo', color='purple')
    # ax[1].set_title('Stance Phase')
    # ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('GRF (N)')
    # ax[1].legend()
    # ax[1].plot(part_main_hump_nattime, part_main_hump_naty, label='nat', color='orange')
    # ax[1].plot(part_main_hump_exotime, part_main_hump_exoy, label='exo', color='purple')
    # ax[2].set_title('Swing Phase')
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_ylabel('GRF (N)')
    # ax[2].legend()
    # ax[2].plot(part_after_hump_nattime, part_after_hump_naty, label='nat', color='orange')
    # ax[2].plot(part_after_hump_exotime, part_after_hump_exoy, label='exo', color='purple')
    # plt.tight_layout()
    # plt.show()
    # load the metabolic results
    natmet = osim.TimeSeriesTable(natmet)
    exomet = osim.TimeSeriesTable(exomet)
    # check that the time vectors are the same
    natmettime = np.array(natmet.getIndependentColumn())
    exomettime = np.array(exomet.getIndependentColumn())
    if not np.array_equal(natmettime, natGRFtime):
        print('Time vectors do not match')
        return
    
    # TODO get all of the individual muscle costs for both stance and swing... 
    # get the total metabolic rate for each phase
    total_stance_nat = np.zeros((len(part_main_hump_nattime),9))
    total_stance_exo = np.zeros((len(part_main_hump_exotime),9))
    total_swing_nat = np.zeros((len(part_after_hump_nattime),9))
    total_swing_exo = np.zeros((len(part_after_hump_exotime),9))
    count = 0
    musclenames = []
    for each in natmet.getColumnLabels():
        if 'total_metabolic_rate' in each and '_r_' in each:
            shortname = each.split('_r_')[0][1:]; musclenames.append(shortname)
            # get the data
            natdata = natmet.getDependentColumn(each).to_numpy()
            exodata = exomet.getDependentColumn(each).to_numpy()
            # get the stance and swing phases
            natstance = natdata[start_of_hump_index_nat:return_to_zero_index_nat + 1]
            exostance = exodata[start_of_hump_index_exo:return_to_zero_index_exo + 1]
            natswing = natdata[return_to_zero_index_nat + 1:]
            exoswing = exodata[return_to_zero_index_exo + 1:]
            total_stance_nat[:,count] = natstance
            total_stance_exo[:,count] = exostance
            total_swing_nat[:,count] = natswing
            total_swing_exo[:,count] = exoswing
            count += 1

    # compute stance and swing for each muscle costs individually, and print... 
    # for nat and for exo
    natstanceind = {}
    exostanceind = {}
    natswingind = {}
    exoswingind = {}
    for m in musclenames:
        natstanceind[m] = np.trapz(total_stance_nat[:,musclenames.index(m)], x=part_main_hump_nattime) / (part_main_hump_nattime[-1] - part_main_hump_nattime[0]) / modelmass
        exostanceind[m] = np.trapz(total_stance_exo[:,musclenames.index(m)], x=part_main_hump_exotime) / (part_main_hump_exotime[-1] - part_main_hump_exotime[0]) / modelmass
        natswingind[m] = np.trapz(total_swing_nat[:,musclenames.index(m)], x=part_after_hump_nattime) / (part_after_hump_nattime[-1] - part_after_hump_nattime[0]) / modelmass
        exoswingind[m] = np.trapz(total_swing_exo[:,musclenames.index(m)], x=part_after_hump_exotime) / (part_after_hump_exotime[-1] - part_after_hump_exotime[0]) / modelmass    
        # create a figure with subplots for each muscle that plots the time series of the metabolic cost for natural and exo during stance.
    # Create a figure with subplots for each muscle
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
    fig.suptitle('Metabolic Cost Time Series for Individual Muscles During Stance Phase')
    
    # Plot the time series for each muscle
    for i, muscle in enumerate(musclenames):
        row = i // 3
        col = i % 3
        axs[row, col].plot(part_main_hump_nattime, total_stance_nat[:, i], label='Nat', color='orange')
        axs[row, col].plot(part_main_hump_exotime, total_stance_exo[:, i], label='Exo', color='purple')
        axs[row, col].set_title(muscle)
        axs[row, col].set_xlabel('Time (s)')
        axs[row, col].set_ylabel('Metabolic Cost (W/kg)')
        axs[row, col].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # Create a figure with subplots for each muscle that plots the time series of the metabolic cost for natural and exo during swing.
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
    fig.suptitle('Metabolic Cost Time Series for Individual Muscles During swing Phase')
    
    # Plot the time series for each muscle
    for i, muscle in enumerate(musclenames):
        row = i // 3
        col = i % 3
        axs[row, col].plot(part_after_hump_nattime, total_swing_nat[:, i], label='Nat', color='orange')
        axs[row, col].plot(part_after_hump_exotime, total_swing_exo[:, i], label='Exo', color='purple')
        axs[row, col].set_title(muscle)
        axs[row, col].set_xlabel('Time (s)')
        axs[row, col].set_ylabel('Metabolic Cost (W/kg)')
        axs[row, col].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()    

    # Plotting the natstanceind and exostanceind values for stance and swing phases
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    fig.suptitle('Metabolic Rates for Individual Muscles During Stance and Swing Phases')
    # Plotting the stance phase
    # Define the width of the bars
    bar_width = 0.35
    # Get the positions for the bars
    indices = np.arange(len(natstanceind))
    # Plot the bars
    axs[0].bar(indices, natstanceind.values(), bar_width, color='orange', label='Nat Stance')
    axs[0].bar(indices + bar_width, exostanceind.values(), bar_width, color='purple', alpha=0.7, label='Exo Stance')
    axs[0].set_xticks(indices + bar_width / 2)
    axs[0].set_xticklabels(natstanceind.keys())
    axs[0].set_title('Stance Phase')
    axs[0].set_xlabel('Muscles')
    axs[0].set_ylabel('Metabolic Rate (W/kg)')
    axs[0].legend()
    axs[0].tick_params(axis='x', labelrotation=30, length=7, width=1)
    # Set the horizontal alignment of the tick labels to 'right'
    for label in axs[0].get_xticklabels():
        label.set_ha('right')
    # Plotting the swing phase
    axs[1].bar(indices, natswingind.values(), bar_width, color='orange', label='Nat Swing')
    axs[1].bar(indices + bar_width, exoswingind.values(), bar_width, color='purple', alpha=0.7, label='Exo Swing')
    axs[1].set_xticks(indices + bar_width / 2)
    axs[1].set_xticklabels(natswingind.keys())
    axs[1].set_title('Swing Phase')
    axs[1].set_xlabel('Muscles')
    axs[1].set_ylabel('Metabolic Rate (W/kg)')
    axs[1].legend()
    axs[1].tick_params(axis='x', labelrotation=30, length=7, width=1)
    for label in axs[1].get_xticklabels():
        label.set_ha('right')
    # Ensure the y-axis limits are the same for both subplots
    y_min = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
    y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim([y_min, y_max])
    axs[1].set_ylim([y_min, y_max])    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Create a list to store the data
    datastanceind = []
    dataswingind = []
    # Populate the list with muscle names, differences, and percent changes
    for muscle in natstanceind.keys():
        difference_stance = exostanceind[muscle] - natstanceind[muscle]
        percent_change_stance = (difference_stance / natstanceind[muscle]) * 100
        datastanceind.append([muscle, difference_stance, percent_change_stance])
        difference_swing = exoswingind[muscle] - natswingind[muscle]
        percent_change_swing = (difference_swing / natswingind[muscle]) * 100
        dataswingind.append([muscle, difference_swing, percent_change_swing])
    # Create a DataFrame from the data
    dfstanceind = pd.DataFrame(datastanceind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    dfswingind = pd.DataFrame(dataswingind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    # Sort the DataFrame by the 'Difference (W/kg)' column
    df_sorted_stanceind = dfstanceind.sort_values(by='Difference (W/kg)', ascending=True)
    df_sorted_swingind = dfswingind.sort_values(by='Difference (W/kg)', ascending=True)
    # Print the sorted DataFrame
    print('\n>>> Stance Phase <<<')
    print(f"{'Muscle':<20} {'Difference (W/kg)':>25} {'Percent Change (%)':>25}")
    for _, row in df_sorted_stanceind.iterrows():
        print(f"{row['Muscle']:<20} {row['Difference (W/kg)']:>25.4f} {row['Percent Change (%)']:>25.2f} %")
    print('\n>>> Swing Phase <<<')
    print(f"{'Muscle':<20} {'Difference (W/kg)':>25} {'Percent Change (%)':>25}")
    for _, row in df_sorted_swingind.iterrows():
        print(f"{row['Muscle']:<20} {row['Difference (W/kg)']:>25.4f} {row['Percent Change (%)']:>25.2f} %")

    # print out the dataframes for use later. 
    if sim=='CMA_4':
        df_sorted_stanceind.to_csv('stanceStudy2Savings4ms.csv', index=False)
        df_sorted_swingind.to_csv('swingStudy2Savings4ms.csv', index=False)
    if sim=='CMA':
        df_sorted_stanceind.to_csv('stanceStudy2Savings.csv', index=False)
        df_sorted_swingind.to_csv('swingStudy2Savings.csv', index=False)

    # sum the metabolic rates for each phase
    total_stance_nat = np.sum(total_stance_nat, axis=1)
    total_stance_exo = np.sum(total_stance_exo, axis=1)
    total_swing_nat = np.sum(total_swing_nat, axis=1)
    total_swing_exo = np.sum(total_swing_exo, axis=1)
    natstanceavg = np.trapz(total_stance_nat, x=part_main_hump_nattime) / (part_main_hump_nattime[-1] - part_main_hump_nattime[0]) / modelmass
    exostanceavg = np.trapz(total_stance_exo, x=part_main_hump_exotime) / (part_main_hump_exotime[-1] - part_main_hump_exotime[0]) / modelmass
    natswingavg = np.trapz(total_swing_nat, x=part_after_hump_nattime) / (part_after_hump_nattime[-1] - part_after_hump_nattime[0]) / modelmass
    exoswingavg = np.trapz(total_swing_exo, x=part_after_hump_exotime) / (part_after_hump_exotime[-1] - part_after_hump_exotime[0]) / modelmass
    print('\n\n>>> Single leg musculature (think similar to whole body) <<<')
    print('Average metabolic rate for nat stance (W/kg): ' + str(natstanceavg))
    print('Average metabolic rate for exo stance (W/kg): ' + str(exostanceavg))
    print('Average metabolic rate for nat swing (W/kg): ' + str(natswingavg))
    print('Average metabolic rate for exo swing (W/kg): ' + str(exoswingavg))

    print('\nDifference in metabolic rate for stance (W/kg): ' + str(exostanceavg - natstanceavg))
    # print('Percent change in metabolic rate for stance: ' + str((exostanceavg - natstanceavg) / natstanceavg * 100) + ' %')
    highlight_text('Percent change in metabolic rate for stance: ' + str((exostanceavg - natstanceavg) / natstanceavg * 100) + ' %')
    print('\nDifference in metabolic rate for swing (W/kg): ' + str(exoswingavg - natswingavg))
    # print('Percent change in metabolic rate for swing: ' + str((exoswingavg - natswingavg) / natswingavg * 100) + ' %')                                                                       
    highlight_text('Percent change in metabolic rate for swing: ' + str((exoswingavg - natswingavg) / natswingavg * 100) + ' %')

    # # get the total rate in stance and swing for each simulation
    # nattotal = natmet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy()
    # exototal = exomet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy()
    # # get the average total rate for stance and swing
    # natmetstance = nattotal[start_of_hump_index_nat:return_to_zero_index_nat + 1]
    # exometstance = exototal[start_of_hump_index_exo:return_to_zero_index_exo + 1]
    # natmetswing = nattotal[return_to_zero_index_nat + 1:]
    # exometswing = exototal[return_to_zero_index_exo + 1:]
    # # get the average metabolic rate for each phase
    # natstanceavg = np.trapz(natmetstance, x=part_main_hump_nattime) / (part_main_hump_nattime[-1] - part_main_hump_nattime[0]) / modelmass
    # exostanceavg = np.trapz(exometstance, x=part_main_hump_exotime) / (part_main_hump_exotime[-1] - part_main_hump_exotime[0]) / modelmass        
    # natswingavg = np.trapz(natmetswing, x=part_after_hump_nattime) / (part_after_hump_nattime[-1] - part_after_hump_nattime[0]) / modelmass
    # exoswingavg = np.trapz(exometswing, x=part_after_hump_exotime) / (part_after_hump_exotime[-1] - part_after_hump_exotime[0]) / modelmass
    # print('Average metabolic rate for nat stance (W/kg): ' + str(natstanceavg))
    # print('Average metabolic rate for exo stance (W/kg): ' + str(exostanceavg))
    # print('Average metabolic rate for nat swing (W/kg): ' + str(natswingavg))
    # print('Average metabolic rate for exo swing (W/kg): ' + str(exoswingavg))

    # print('\nDifference in metabolic rate for stance (W/kg): ' + str(exostanceavg - natstanceavg))
    # print('Percent change in metabolic rate for stance: ' + str((exostanceavg - natstanceavg) / natstanceavg * 100) + ' %')
    
    # print('\nDifference in metabolic rate for swing (W/kg): ' + str(exoswingavg - natswingavg))
    # print('Percent change in metabolic rate for swing: ' + str((exoswingavg - natswingavg) / natswingavg * 100) + ' %')
    
    # print('\n**Note: Previous study found the following changes at 2.7 m/s:')
    # print('Stance: ~1.5 W or ~12% reduction ')
    # print('Swing: ~0.3 or ~2.5% reduction')
    highlight_text('**Note: Previous study found the following changes at 2.7 m/s:')
    highlight_text('Stance: ~1.5 W or ~12% reduction ')
    highlight_text('Swing: ~0.3 or ~2.5% reduction')
    return

# gather and plot different metabolic values for individual muscles. 
def muscleMetabolicsCompare(sim1, sim2, modelfile):
    # load the metabolic results
    sim1met = osim.TimeSeriesTable(sim1)
    sim2met = osim.TimeSeriesTable(sim2)
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    
    # first lets create a plot for all the right muscles and total met rate. 
    fig, ax = plt.subplots(2,5, figsize=(15,6), dpi=300) # , dpi=300
    ax = ax.flatten()
    count = 0
    metvalues = []
    # loop and get the right muscle total rates
    for each in sim1met.getColumnLabels():
        if '_r_' in each and 'total_metabolic_rate' in each:
            tempname = each.split('_r_')[0]
            # print(tempname)
            # get the data
            sim1data = sim1met.getDependentColumn(each).to_numpy()
            sim2data = sim2met.getDependentColumn(each).to_numpy()
            # get the time
            time = sim1met.getIndependentColumn()
            x1 = np.linspace(0,100,len(sim1data))
            x2 = np.linspace(0,100,len(sim2data))
            sim2data = np.interp(x1, x2, sim2data)
            # plot it
            ax[count].plot(x1, sim1data, label='sim1', color='orange')
            ax[count].plot(x2, sim2data, label='sim2', color='purple')
            ax[count].set_title(tempname[1:])
            ax[count].set_ylabel('Metabolic rate W/kg')
            ax[count].set_xlabel('Gait cycle (%)')
            count += 1

            # now get the average values for each one 
            sim1_avg = np.trapz(sim1data, x=x1) / (x1[-1] - x1[0]) / modelmass
            sim2_avg = np.trapz(sim2data, x=x2) / (x2[-1] - x2[0]) / modelmass
            # print(f'Avg of sim1 (nat) for {tempname[1:]}: {sim1_avg}')
            # print(f'Avg of sim2 (exo) for {tempname[1:]}: {sim2_avg}')
            # now print out the difference and percent change between the two
            diff = sim2_avg - sim1_avg
            percent_change = (diff / sim1_avg) * 100
            # print(f'Difference: {diff}')
            # print(f'Percent change: {percent_change}%\n')            
            metvalues.append({'Muscle': tempname, 'Nat. Met': sim1_avg, 'Exo Met': sim2_avg, 'Difference': diff, 'Percent Change': percent_change})
    plt.tight_layout()
    figurePath = os.getcwd() + '\\..\\..\\analysis\\'
    plt.savefig(figurePath + 'individualMetabolics_Sim' + '2_7' + '.png')
    met_df = pd.DataFrame(metvalues)
    met_df.to_csv(figurePath + 'muscleMetabolics_Sim' + '2_7' + '.csv')
    # Sort the DataFrame based on the 'Difference' column
    met_df = met_df.sort_values(by='Difference', ascending=True)
    print(met_df)
    plt.show()
    return 

# function for quick met cost 
def simMetCost(metTable, modelFile):
    # load in the table
    mettime = metTable.getIndependentColumn()
    metAll = metTable.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy()

    # need the model mass
    model = osim.Model(modelFile)
    state = model.initSystem()
    mass = model.getTotalMass(state)

    # now what am I going to do with it. 
    metAllAvg = (np.trapz(metAll, x=mettime) / (mettime[-1] - mettime[0])) / mass
    # metabolics_all_avg = ((trapz(time, metabolics_all)) / (time(end)-time(1))) / model_mass;
    print('\n\nAvg. Metabolic Rate :: %f \n' % metAllAvg)
    return metAllAvg

# function for quick met cost 
def quickMetCost(probeFile, modelFile):
    # load in the table
    metTable = osim.TimeSeriesTable(probeFile)
    mettime = metTable.getIndependentColumn()
    metAll = metTable.getDependentColumn('all_metabolics_TOTAL').to_numpy()

    # need the model mass
    model = osim.Model(modelFile)
    state = model.initSystem()
    mass = model.getTotalMass(state)

    # now what am I going to do with it. 
    metAllAvg = (np.trapz(metAll, x=mettime) / (mettime[-1] - mettime[0])) / mass
    # metabolics_all_avg = ((trapz(time, metabolics_all)) / (time(end)-time(1))) / model_mass;
    print('\n\nAvg. Metabolic Rate:: %f \n' % metAllAvg)
    return metAllAvg

# compute and visualize differences in metabolics between stance and swing
def wholeBodyMetabolics(natmet, exomet, modelfile):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the metabolic results
    natmet = osim.TimeSeriesTable(natmet)
    exomet = osim.TimeSeriesTable(exomet)
    # check that the time vectors are the same
    natmettime = np.array(natmet.getIndependentColumn())
    exomettime = np.array(exomet.getIndependentColumn())
    # get the total body rate for natural and exotendon cases. 
    natmettotal = natmet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy()
    exomettotal = exomet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy()
    # get the average total rate for both conditions
    natmetavg = np.trapz(natmettotal, x=natmettime) / (natmettime[-1] - natmettime[0]) / modelmass
    exometavg = np.trapz(exomettotal, x=exomettime) / (exomettime[-1] - exomettime[0]) / modelmass
    print('Average metabolic rate for nat (W/kg): ' + str(natmetavg))
    print('Average metabolic rate for exo (W/kg): ' + str(exometavg))
    # print('Difference in metabolic rate (W/kg): ' + str(exometavg - natmetavg))
    # print('Percent change in metabolic rate: ' + str((exometavg - natmetavg) / natmetavg * 100) + ' %')
    highlight_text('Difference in metabolic rate (W/kg): ' + str(exometavg - natmetavg))
    highlight_text('Percent change in metabolic rate: ' + str((exometavg - natmetavg) / natmetavg * 100) + ' %')
    
    # print('\n**Note: Previous study found the following changes at 2.7 m/s:')
    # print('Simpson experimental difference: -6.4% +/- 2.8%')
    # print('Stingel experimental difference: -0.9 +/- 0.2 W/kg or -8.3 +/- 1.3% ')
    # print('Stingel simulation difference: -1.4 +/- 0.2 W/kg or -12.0 +/- 1.8%')
    highlight_text('**Note: Previous study found the following changes at 2.7 m/s:')
    highlight_text('Simpson experimental difference: -6.4% +/- 2.8%')
    highlight_text('Stingel experimental difference: -0.9 +/- 0.2 W/kg or -8.3 +/- 1.3% ')
    highlight_text('Stingel simulation difference: -1.4 +/- 0.2 W/kg or -12.0 +/- 1.8%')
    return

# gather and plot the joint moments from a set of solutions vs the experimental 3D data. 
def jointMomentCompare(sim1, sim2, ref1, ref2, sim, stdmomnat, stdmomexo):
    subjectmass = 73.48
    if sim == 'CMA':
        # load the tables and plot each of the entries across both sim and ref
        idrefnat = osim.TimeSeriesTable(ref1)
        idrefexo = osim.TimeSeriesTable(ref2)
        idsimnat = osim.TimeSeriesTable(sim1)
        idsimexo = osim.TimeSeriesTable(sim2)
        # get the time vectors
        idrefnat_time = idrefnat.getIndependentColumn()
        idrefexo_time = idrefexo.getIndependentColumn()
        idsimnat_time = idsimnat.getIndependentColumn()
        idsimexo_time = idsimexo.getIndependentColumn()
        # get the labels for the columns
        idrefnat_labels = idrefnat.getColumnLabels()
        idrefexo_labels = idrefexo.getColumnLabels()
        idsimnat_labels = idsimnat.getColumnLabels()
        idsimexo_labels = idsimexo.getColumnLabels()
        # loop through the labels and plot each one
        fig, ax = plt.subplots(5,4, figsize=(15,15), dpi=200) # , dpi=300
        ax = ax.flatten()
        count = 0
        rmse_values = []
        peak_diffs = []
        for lab in idsimnat_labels:
            if 'beta' not in lab: 
                yrefnat = idrefnat.getDependentColumn(lab).to_numpy()
                yrefexo = idrefexo.getDependentColumn(lab).to_numpy()
                ysimnat = idsimnat.getDependentColumn(lab).to_numpy()
                ysimexo = idsimexo.getDependentColumn(lab).to_numpy()
                
                # resize everything if needed
                xsimnat = np.linspace(0, 100, len(ysimnat))
                xsimexo = np.linspace(0, 100, len(ysimexo))
                
                if 'arm' not in lab and 'elbow' not in lab and 'xdes_refnat' not in locals():
                    xdes_refnat = np.linspace(0, 100, len(stdmomnat[lab]))
                    xdes_refexo = np.linspace(0, 100, len(stdmomexo[lab]))
                
                xrefnat = np.linspace(0, 100, len(yrefnat))
                xrefexo = np.linspace(0, 100, len(yrefexo))
                
                yrefnat = np.interp(xdes_refnat, xrefnat, yrefnat)
                yrefexo = np.interp(xdes_refexo, xrefexo, yrefexo)

                ysimnat = np.interp(xdes_refnat, xsimnat, ysimnat)
                ysimexo = np.interp(xdes_refexo, xsimexo, ysimexo)

                # TODO splice the kinematics to show what we are really matching
                # if yn.size != 101:
                if 'pelvis' not in lab and 'lumbar' not in lab:
                    if '_r' in lab or '_l' in lab:
                        # TODO figure out how to mirror the left to the right and stitch it together for a better representation of what we are tracking.
                        yrefnat_new = spliceKinematics(yrefnat, idrefnat, lab)
                        yrefexo_new = spliceKinematics(yrefexo, idrefexo, lab)
                    else:
                        yrefnat_new = yrefnat
                        yrefexo_new = yrefexo
                else: 
                    yrefnat_new = yrefnat
                    yrefexo_new = yrefexo

                # normalize the moments to the body mass of the subject. 
                yrefnat = yrefnat_new / subjectmass
                yrefexo = yrefexo_new / subjectmass
                ysimnat = ysimnat / subjectmass
                ysimexo = ysimexo / subjectmass

                if 'arm' not in lab and 'elbow' not in lab:
                    # get the plus and minus from the standard deviation data. 
                    yrefplusnat = yrefnat + stdmomnat[lab]
                    yrefminusnat = yrefnat - stdmomnat[lab]
                    yrefplusexo = yrefexo + stdmomexo[lab]
                    yrefminusexo = yrefexo - stdmomexo[lab]
                    # plot std data
                    ax[count].fill_between(xdes_refnat, yrefminusnat, yrefplusnat, color='orange', alpha=0.2, label='1std (previous simulations)')
                    ax[count].fill_between(xdes_refexo, yrefminusexo, yrefplusexo, color='purple', alpha=0.2, label='1std (previous simulations)')

                # plot the data
                ax[count].plot(xdes_refnat, yrefnat, label='tracked nat', color='orange')
                ax[count].plot(xdes_refexo, yrefexo, label='tracked exo', color='purple')
                ax[count].plot(xdes_refnat, ysimnat, label='sim nat', color='orange', linestyle='--', linewidth=2)
                ax[count].plot(xdes_refexo, ysimexo, label='sim exo', color='purple', linestyle='--', linewidth=2)
                ax[count].set_title(lab)
                ax[count].set_ylabel('Moment (Nm/kg)')
                ax[count].set_xlabel('time (sec)')
                count += 1
                # interpolate the ref data to match the sim data. 
                # xrefnat = np.linspace(idrefnat_time[0], idrefnat_time[-1], 100)
                # xrefexo = np.linspace(idrefexo_time[0], idrefexo_time[-1], 100)
                # xsimnat = np.linspace(idsimnat_time[0], idsimnat_time[-1], 100)
                # xsimexo = np.linspace(idsimexo_time[0], idsimexo_time[-1], 100)
                # yrefnat2 = np.interp(xrefnat, idrefnat_time, yrefnat)
                # yrefexo2 = np.interp(xrefexo, idrefexo_time, yrefexo)
                # ysimnat2 = np.interp(xsimnat, idsimnat_time, ysimnat)
                # ysimexo2 = np.interp(xsimexo, idsimexo_time, ysimexo)
                # print and do some RMSE calculations
                # print(lab)
                natRMSE = np.sqrt(np.mean((ysimnat - yrefnat)**2))
                exoRMSE = np.sqrt(np.mean((ysimexo - yrefexo)**2))
                # print('RMSE for nat: ' + str(natRMSE))
                # print('RMSE for exo: ' + str(exoRMSE))
                rmse_values.append({'Coordinate Moment': lab, 'Nat RMSE': natRMSE, 'Exo RMSE': exoRMSE})
                if 'knee' in lab and '_r_' in lab:
                    expdiff_ext = np.min(yrefexo) - np.min(yrefnat)
                    simdiff_ext = np.min(ysimexo) - np.min(ysimnat)
                    expdiff_flex = np.max(yrefexo) - np.max(yrefnat)
                    simdiff_flex = np.max(ysimexo) - np.max(ysimnat)
                    peak_diffs.append({'Moment': 'knee extension', 'Exp Peak Diff': expdiff_ext, 'Sim Peak Diff': simdiff_ext})
                    highlight_text('knee extension moment peak difference between simulations: ' + str(simdiff_ext))
                    highlight_text('NOTE: knee extension moment peak difference between reference data: ' + str(expdiff_ext))
                    peak_diffs.append({'Moment': 'knee flexion', 'Exp Peak Diff': expdiff_flex, 'Sim Peak Diff': simdiff_flex})
                if 'hip' in lab and '_r_' in lab: 
                    expdiff_flex = np.min(yrefexo) - np.min(yrefnat)
                    simdiff_flex = np.min(ysimexo) - np.min(ysimnat)
                    expdiff_ext = np.max(yrefexo) - np.max(yrefnat)
                    simdiff_ext = np.max(ysimexo) - np.max(ysimnat)
                    peak_diffs.append({'Moment': 'hip extension', 'Exp Peak Diff': expdiff_ext, 'Sim Peak Diff': simdiff_ext})
                    peak_diffs.append({'Moment': 'hip flexion', 'Exp Peak Diff': expdiff_flex, 'Sim Peak Diff': simdiff_flex})


        rmse_df = pd.DataFrame(rmse_values)
        figurePath = os.getcwd() + '\\..\\..\\analysis\\'
        rmse_df.to_csv(figurePath + 'moments_RMSE_SimVS3DID' + '2_7' + '.csv')
        peak_df = pd.DataFrame(peak_diffs)
        peak_df.to_csv(figurePath + 'moments_PeakDiff_SimVS3DID' + '2_7' + '.csv')
        print('Peak differences in some of the key saggital plane moments')
        print(peak_df)
        print('\nRMSE values for the moments')
        print(rmse_df)
        plt.tight_layout()
        plt.savefig(figurePath + 'moments_compare_27.png')
        plt.show()

    if sim == 'CMA_4':
        # load the tables and plot each of the entries across both sim and ref
        idrefnat = osim.TimeSeriesTable(ref1)
        # idrefexo = osim.TimeSeriesTable(ref2)
        idsimnat = osim.TimeSeriesTable(sim1)
        idsimexo = osim.TimeSeriesTable(sim2)
        # get the time vectors
        idrefnat_time = idrefnat.getIndependentColumn()
        # idrefexo_time = idrefexo.getIndependentColumn()
        idsimnat_time = idsimnat.getIndependentColumn()
        idsimexo_time = idsimexo.getIndependentColumn()
        # get the labels for the columns
        idrefnat_labels = idrefnat.getColumnLabels()
        # idrefexo_labels = idrefexo.getColumnLabels()
        idsimnat_labels = idsimnat.getColumnLabels()
        idsimexo_labels = idsimexo.getColumnLabels()
        # loop through the labels and plot each one
        fig, ax = plt.subplots(5,4, figsize=(15,15), dpi=200) # , dpi=300
        ax = ax.flatten()
        count = 0
        rmse_values = []
        peak_diffs = []
        peak_names = []
        for lab in idsimnat_labels:
            if 'beta' not in lab:
                yrefnat = idrefnat.getDependentColumn(lab).to_numpy()
                # yrefexo = idrefexo.getDependentColumn(lab).to_numpy()
                ysimnat = idsimnat.getDependentColumn(lab).to_numpy()
                ysimexo = idsimexo.getDependentColumn(lab).to_numpy()

                if 'knee' in lab:
                    ysimnat = -ysimnat
                    ysimexo = -ysimexo
                # # interpolate the sim data to match the ref data
                xdes = np.linspace(0,100,101)
                xdes_refnat = np.linspace(idrefnat_time[0], idrefnat_time[-1], 101)
                xdes_simnat = np.linspace(idsimnat_time[0], idsimnat_time[-1], 101)
                xdes_simexo = np.linspace(idsimexo_time[0], idsimexo_time[-1], 101)
                ysimnat2 = np.interp(xdes_simnat, idsimnat_time, ysimnat)
                ysimexo2 = np.interp(xdes_simexo, idsimexo_time, ysimexo)
                yrefnat2 = np.interp(xdes_refnat, idrefnat_time, yrefnat)
                
                # TODO splice the kinematics to show what we are really matching
                # if yn.size != 101:
                if 'pelvis' not in lab and 'lumbar' not in lab:
                    print(lab)
                    if '_r' in lab or '_l' in lab:
                        # TODO figure out how to mirror the left to the right and stitch it together for a better representation of what we are tracking.
                        yrefnat_new = spliceKinematics(yrefnat2, idrefnat, lab)
                    else:
                        yrefnat_new = yrefnat2
                else: 
                    yrefnat_new = yrefnat2
                
                
                # normalize all the moments to body mass
                yrefnat_new = yrefnat_new / 65 # hamner subject 19 reference was 65 kg
                ysimnat2 = ysimnat2 / subjectmass
                ysimexo2 = ysimexo2 / subjectmass
                
                
                # plot the data
                ax[count].plot(xdes, yrefnat_new, label='tracked nat', color='orange')
                # ax[count].plot(idrefexo_time, yrefexo, label='tracked exo', color='purple')
                ax[count].plot(xdes, ysimnat2, label='sim nat', color='orange', linestyle='--', linewidth=2)
                ax[count].plot(xdes, ysimexo2, label='sim exo', color='purple', linestyle='--', linewidth=2)
                ax[count].set_title(lab)
                ax[count].set_ylabel('Moment (Nm)')
                ax[count].set_xlabel('time (sec)')
                count += 1
                # # print and do some RMSE calculations
                # print(lab)
                natRMSE = np.sqrt(np.mean((ysimnat2 - yrefnat_new)**2))
                # exoRMSE = np.sqrt(np.mean((ysimexo2 - yrefexo2)**2))
                # print('RMSE for nat: ' + str(natRMSE))
                # print('RMSE for exo: ' + str(exoRMSE))
                rmse_values.append({'Coordinate Moment': lab, 'Nat RMSE': natRMSE}) # , 'Exo RMSE': exoRMSE})
                if 'knee' in lab and '_r_' in lab:
                    # expdiff_ext = np.min(yrefexo2) - np.min(yrefnat_new)
                    simdiff_ext = np.min(ysimexo2) - np.min(ysimnat2)
                    # expdiff_flex = np.max(yrefexo2) - np.max(yrefnat_new)
                    simdiff_flex = np.max(ysimexo2) - np.max(ysimnat2)
                    peak_diffs.append({'Moment': 'knee extension', 'Sim Peak Diff': simdiff_ext})
                    highlight_text('knee extension moment peak difference between simulations: ' + str(simdiff_ext))
                    # highlight_text('NOTE: knee extension moment peak difference between reference data: ' + str(expdiff_ext))
                    peak_diffs.append({'Moment': 'knee flexion', 'Sim Peak Diff': simdiff_flex})
                if 'hip' in lab and '_r_' in lab: 
                    # expdiff_flex = np.min(yrefexo2) - np.min(yrefnat_new)
                    simdiff_flex = np.min(ysimexo2) - np.min(ysimnat2)
                    # expdiff_ext = np.max(yrefexo2) - np.max(yrefnat_new)
                    simdiff_ext = np.max(ysimexo2) - np.max(ysimnat2)
                    peak_diffs.append({'Moment': 'hip extension', 'Sim Peak Diff': simdiff_ext})
                    peak_diffs.append({'Moment': 'hip flexion', 'Sim Peak Diff': simdiff_flex})
        rmse_df = pd.DataFrame(rmse_values)
        figurePath = os.getcwd() + '\\..\\..\\analysis\\'
        rmse_df.to_csv(figurePath + 'moments_RMSE_SimVS3DID' + '_40' + '.csv')
        peak_df = pd.DataFrame(peak_diffs)
        peak_df.to_csv(figurePath + 'moments_PeakDiff_SimVS3DID' + '_40' + '.csv')
        print('Peak differences in some of the key saggital plane moments')
        print(peak_df)
        print('\nRMSE values for the moments')
        print(rmse_df)
        plt.tight_layout()
        plt.show()

    return

# get sim and experimental data column labels matching
def getColumnNames(coordinates, labels2D, simLabels):
    # get only the sim labels that match the coordinates that we are looking at 
    coordinates_sim = np.empty(len(coordinates), dtype=object)
    coordinates_sim_clean = np.empty(len(labels2D), dtype=object)
    for c, each in enumerate(coordinates):
        for i, temp in enumerate(simLabels):
            if each in temp and 'speed' not in temp:
                coordinates_sim[c] = str(temp)
    for c, each in enumerate(labels2D):
        if 'knee_angle' in each and 'beta' not in each: 
            # case for knee_angle_r
            for i, temp in enumerate(simLabels):
                if each in temp and 'speed' not in temp and 'beta' not in temp:
                    coordinates_sim_clean[c] = str(temp)
        elif 'beta' in each:
            # case for knee_angle_r beta 
            for i, temp in enumerate(simLabels):
                if each in temp and 'speed' not in temp:
                    coordinates_sim_clean[c] = str(temp)
        else:
            for i, temp in enumerate(simLabels):
                if each in temp and 'speed' not in temp:
                    coordinates_sim_clean[c] = str(temp)
    return coordinates_sim, coordinates_sim_clean

# splice half left and half right kinematics together for plotting - for a single coordinate. 
def spliceKinematics(yn, ikNat2D, coord):
    if '_r' in coord:
        # print(coord)
        # print(coord.replace('_r', '_l'))
        # get the left side
        yn_new = np.zeros(len(yn))
        yl = ikNat2D.getDependentColumn(coord.replace('_r', '_l')).to_numpy()
        yl = np.interp(np.linspace(0, 100, len(yn)), np.linspace(0, 100, len(yl)), yl)
        yn_new[:50] = yn[:50]
        yn_new[50:] = yl[:51]#[::-1]

        # fig, ax = plt.subplots(figsize=(10, 6))
        # # Plot the original yn, yl, and yn_new
        # x = np.linspace(0, 100, len(yn))
        # ax.plot(x, yn, label='Original yn', color='blue')
        # ax.plot(x, yl, label='Original yl', color='green')
        # ax.plot(x, yn_new, label='Spliced yn_new', color='red', linestyle='--')
        # ax.set_title('Spliced Kinematics')
        # ax.set_xlabel('Gait Cycle (%)')
        # ax.set_ylabel('Coordinate Value')
        # ax.legend()
        # plt.show()
        # pdb.set_trace()

    elif '_l' in coord:
        # get the right side
        yn_new = np.zeros(len(yn))
        yr = ikNat2D.getDependentColumn(coord.replace('_l', '_r')).to_numpy()
        yr = np.interp(np.linspace(0, 100, len(yn)), np.linspace(0, 100, len(yr)), yr)
        yn_new[:50] = yn[:50]
        yn_new[50:] = yr[:51]#[::-1]

    return yn_new

# script for plotting the coordinates for the 2.7 solutions. 
def plotCoordinates27(simNat, simExo, ikNat2D, ikexo2D, labels2D, coordinates_sim_clean, std_nat, std_exo):
    simlen = len(simNat.getIndependentColumn())
    rmse_values = []
    peak_diff = []
    fig_size = (15, 15)
    dpi = 200
    fig1, ax1 = plt.subplots(nrows=5, ncols=5, figsize=fig_size, dpi=dpi)
    # loop each coordinate and plot them
    for i, ax in enumerate(ax1.flatten()):
        if i<len(labels2D):
            coord = labels2D[i]    
            # now do the plotting - reference data first
            yn = ikNat2D.getDependentColumn(coord).to_numpy()
            ye = ikexo2D.getDependentColumn(coord).to_numpy()

            # if yn.size != 101:
            yn = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yn)), yn)
            ye = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(ye)), ye)
            x = np.linspace(0,100,len(yn))
            if 'pelvis' not in coord and 'lumbar' not in coord:
                if '_r' in coord or '_l' in coord:
                    # TODO figure out how to mirror the left to the right and stitch it together for a better representation of what we are tracking.
                    yn_new = spliceKinematics(yn, ikNat2D, coord)
                    ye_new = spliceKinematics(ye, ikexo2D, coord)
                else:
                    yn_new = yn
                    ye_new = ye
            else: 
                yn_new = yn
                ye_new = ye
            

            # handle pelvis ty - if it is ty then we want to divide by the height of the subject
            if 'pelvis_ty' in coord:
                subjectheight = 1.78
                yn_new = yn_new / subjectheight
                ye_new = ye_new / subjectheight                
            lowerbound_n = yn_new - np.interp(x, np.linspace(0,100,len(((np.array(std_nat[coord]))))), ((np.array(std_nat[coord]))))
            upperbound_n = yn_new + np.interp(x, np.linspace(0,100,len(((np.array(std_nat[coord]))))), ((np.array(std_nat[coord]))))
            lowerbound_e = ye_new - np.interp(x, np.linspace(0,100,len(((np.array(std_nat[coord]))))), ((np.array(std_exo[coord]))))
            upperbound_e = ye_new + np.interp(x, np.linspace(0,100,len(((np.array(std_nat[coord]))))), ((np.array(std_exo[coord]))))
            ax.fill_between(x, lowerbound_n, upperbound_n, color='orange', alpha=0.15)
            ax.fill_between(x, lowerbound_e, upperbound_e, color='purple', alpha=0.15)
            ax.plot(x, yn_new, color='orange')
            ax.plot(x, ye_new, color='purple')
            # now do the plotting - for the simulation
            coord_sim = coordinates_sim_clean[i]
            if coord_sim:            
                sn = simNat.getDependentColumn(coord_sim).to_numpy()
                se = simExo.getDependentColumn(coord_sim).to_numpy()
                # yt1 = simTightNat1.getDependentColumn(coord_sim).to_numpy()
                # yt2 = simTightNat2.getDependentColumn(coord_sim).to_numpy()
                # yt1 = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yt1)), yt1)
                # yt2 = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yt2)), yt2)
                if '_tx' not in coord_sim and '_ty' not in coord_sim:
                    sn = sn*180/np.pi
                    se = se*180/np.pi
                    ax.plot(x, sn, color='orange', linestyle='--')
                    ax.plot(x, se, color='purple', linestyle='--')
                    # ax.plot(x, yt1*180/np.pi, color='black')
                    # ax.plot(x, yt2*180/np.pi, color='grey', label='9Tight Solution')
                else:    
                    if '_ty' in coord_sim:
                        sn = sn / subjectheight
                        se = se / subjectheight
                    ax.plot(x, sn, color='orange', linestyle='--')
                    ax.plot(x, se, color='purple', linestyle='--')
                    # ax.plot(x, yt1, color='black')
                    # ax.plot(x, yt2, color='grey', label='9Tight Solution')
                # figure out some error metrics here - and print
                # print('Coordinate: ', coord)
                tempcoordRMSE_nat = np.sqrt(np.mean((sn - yn_new)**2))
                tempcoordRMSE_exo = np.sqrt(np.mean((se - ye_new)**2))
                # print('Nat RMSE: ' + str(tempcoordRMSE_nat))
                # print('Exo RMSE: ' + str(tempcoordRMSE_exo) + '\n')

                rmse_values.append({'Coordinate': coord, 'Nat RMSE': tempcoordRMSE_nat, 'Exo RMSE': tempcoordRMSE_exo})
                if 'knee_angle_r' in coord_sim and 'beta' not in coord_sim:
                    # get the peak differences
                    # Find peaks in the simulation data
                    peaks_sim_nat, _ = find_peaks(sn[:len(sn)//2])
                    peaks_sim_exo, _ = find_peaks(se[:len(se)//2])

                    # Find peaks in the experimental data
                    peaks_exp_nat, _ = find_peaks(yn_new[:len(yn_new)//2])
                    peaks_exp_exo, _ = find_peaks(ye_new[:len(ye_new)//2])

                    # Calculate the peak differences
                    sim_diff = np.max(se[peaks_sim_exo]) - np.max(sn[peaks_sim_nat])
                    exp_diff = np.max(ye_new[peaks_exp_exo]) - np.max(yn_new[peaks_exp_nat])
                    peak_diff.append({'Coordinate': coord, 'Experimental difference': exp_diff, 'Simulation difference': sim_diff})
                    highlight_text('Peak difference in knee flexion angle in stance between simulations: ' + str(sim_diff))
                    highlight_text('NOTE: Peak difference in knee flexion angle in stance between reference data: ' + str(exp_diff))
                    # Plot the peaks
                    ax.plot(x[peaks_sim_nat], sn[peaks_sim_nat], "x", color='blue')
                    ax.plot(x[peaks_sim_exo], se[peaks_sim_exo], "x", color='red')
                    ax.plot(x[peaks_exp_nat], yn_new[peaks_exp_nat], "o", color='blue')
                    ax.plot(x[peaks_exp_exo], ye_new[peaks_exp_exo], "o", color='red')
                    
                if 'ty' in coord_sim: 
                    # get the peak differences
                    nat_sim = np.max(sn) - np.min(sn)
                    exo_sim = np.max(se) - np.min(se)
                    nat_exp = np.max(yn_new) - np.min(yn_new)
                    exo_exp = np.max(ye_new) - np.min(ye_new)
                    highlight_text('Vertical pelvis translation |  Sim. Nat: ' + str(nat_sim) + ' | Exp. Nat: ' + str(nat_exp))
                    highlight_text('Vertical pelvis translation |  Sim. Exo: ' + str(exo_sim) + ' | Exp. Exo: ' + str(exo_exp))
            # format the final parts
            ax.set_title(coord, fontsize=12)
            ax.set_xlabel('GC%')
            ax.set_ylabel('Coordinate Value (deg or m)')

        if i >= len(labels2D):
            # now do the plotting
            # yn_new = np.array(mean_nat[coord])
            # ye_new = np.array(mean_exo[coord])
            # now do the plotting - reference data first
            yn_new = ikNat2D.getDependentColumn(coord).to_numpy()
            ye_new = ikexo2D.getDependentColumn(coord).to_numpy()
            # if yn_new.size != 101:
            yn_new = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yn_new)), yn_new)
            ye_new = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(ye_new)), ye_new)
            lowerbound_n = yn_new - np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
            upperbound_n = yn_new + np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
            lowerbound_e = ye_new - np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_exo[coord]))))
            upperbound_e = ye_new + np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_exo[coord]))))
            ax.fill_between(x, lowerbound_n, upperbound_n, color='orange', alpha=0.1, label='Pop. 2 StDev. Natural')
            ax.fill_between(x, lowerbound_e, upperbound_e, color='purple', alpha=0.1, label='Pop. 2 StDev. Exo')
            ax.plot(x, yn_new, color='orange', label='Natural Tracked Data')
            ax.plot(x, ye_new, color='purple', label='Exo experimental Data')
            ax.plot(x, sn, color='orange', linestyle='--', label='Nat Sim.')
            ax.plot(x, se, color='purple', linestyle='--', label='Exo Sim.')
            ax.set_title(coord, fontsize=12)
            ax.set_xlabel('GC%')
            ax.set_ylabel('Coordinate Value (deg or m)')
            ax.legend()
    figurePath = os.getcwd() + '\\..\\..\\analysis\\'
    rmse_df = pd.DataFrame(rmse_values)
    rmse_df.to_csv(figurePath + 'coordinate_RMSE_SimVS2DIK' + '2_7' + '.csv')
    print(rmse_df)

    rmse_df_end = rmse_df[~rmse_df['Coordinate'].str.contains('beta')]
    rmse_df_end2 = rmse_df_end[~rmse_df_end['Coordinate'].str.contains('mtp')]
    
    rmse_df_angles = rmse_df_end2[~rmse_df_end2['Coordinate'].str.contains('pelvis_tx')]
    rmse_df_angles2 = rmse_df_angles[~rmse_df_angles['Coordinate'].str.contains('pelvis_ty')]
    
    rmse_df_trans = rmse_df_end2[rmse_df_end2['Coordinate'].str.contains('pelvis_ty')]
    # rmse_df_trans2 = rmse_df_trans[~rmse_df_trans['Coordinate'].str.contains('pelvis_tilt')]

    # angle rmse average and std
    avgRMSEnat = np.mean(rmse_df_angles2['Nat RMSE'])
    stdRMSEnat = np.std(rmse_df_angles2['Nat RMSE'])
    avgRMSEexo = np.mean(rmse_df_angles2['Exo RMSE'])
    stdRMSEexo = np.std(rmse_df_angles2['Exo RMSE'])

    print(np.std(std_nat['pelvis_ty']))
    print(np.std(std_exo['pelvis_ty']))

    # translation rmse average and std
    avgRMSEnat_trans = np.mean(rmse_df_trans['Nat RMSE'])
    stdRMSEnat_trans = np.std(rmse_df_trans['Nat RMSE'])
    avgRMSEexo_trans = np.mean(rmse_df_trans['Exo RMSE'])
    stdRMSEexo_trans = np.std(rmse_df_trans['Exo RMSE'])


    print('Average RMSE for natural: ' + str(avgRMSEnat) + ' +/- ' + str(stdRMSEnat))
    print('Average RMSE for exo: ' + str(avgRMSEexo) + ' +/- ' + str(stdRMSEexo))
    print('Average RMSE for natural translation: ' + str(avgRMSEnat_trans) + ' +/- ' + str(stdRMSEnat_trans))
    print('Average RMSE for exo translation: ' + str(avgRMSEexo_trans) + ' +/- ' + str(stdRMSEexo_trans))
    plt.tight_layout()
    plt.savefig(figurePath + 'coordinate_variations_SimVS2DIK_2STDEV' + '2_7' + '.png')
    plt.show()
    
    return

# function to plot the coordinates for the 4.0 solutions.
def plotCoordinates40(simNat, simExo, ikNat2D, labels2D, coordinates_sim_clean, std_nat):
    simlen = len(simNat.getIndependentColumn())
    x = np.linspace(0,100,101)
    rmse_values = []
    rmse_values_asSTD = []
    corr_values = []
    # want to compute the RMSE as a function of the STDs?? maybe
    
    # also need some correlations

    fig_size = (15, 15)
    dpi = 200
    fig1, ax1 = plt.subplots(nrows=5, ncols=5, figsize=fig_size, dpi=dpi)
    # loop each coordinate and plot them
    for i, ax in enumerate(ax1.flatten()):
        if i<len(labels2D):
            coord = labels2D[i]
            if 'beta' not in coord:
                # now do the plotting - reference data first
                yn = ikNat2D.getDependentColumn(coord).to_numpy()
                # ye = ikexo2D.getDependentColumn(coord).to_numpy()
                # if yn.size != 101:
                yn = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yn)), yn)
                # ye = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(ye)), ye)

                x = np.linspace(0,100,len(yn))
                if 'pelvis' not in coord and 'lumbar' not in coord:
                    if '_r' in coord or '_l' in coord:
                        # TODO figure out how to mirror the left to the right and stitch it together for a better representation of what we are tracking.
                        yn_new = spliceKinematics(yn, ikNat2D, coord)
                    else:
                        yn_new = yn
                else: 
                    yn_new = yn


                lowerbound_n = yn_new - np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
                upperbound_n = yn_new + np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
                # lowerbound_e = ye - np.interp(x, np.linspace(0,100,101), (2*(np.array(std_exo[coord]))))
                # upperbound_e = ye + np.interp(x, np.linspace(0,100,101), (2*(np.array(std_exo[coord]))))
                ax.fill_between(x, lowerbound_n, upperbound_n, color='orange', alpha=0.15)
                # ax.fill_between(x, lowerbound_e, upperbound_e, color='purple', alpha=0.15)
                ax.plot(x, yn_new, color='orange')
                # ax.plot(x, ye, color='purple')
                # now do the plotting - for the simulation
                coord_sim = coordinates_sim_clean[i]
                if coord_sim:            
                    sn = simNat.getDependentColumn(coord_sim).to_numpy()
                    se = simExo.getDependentColumn(coord_sim).to_numpy()
                    # yt1 = simTightNat1.getDependentColumn(coord_sim).to_numpy()
                    # yt2 = simTightNat2.getDependentColumn(coord_sim).to_numpy()
                    # yt1 = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yt1)), yt1)
                    # yt2 = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yt2)), yt2)
                    if '_tx' not in coord_sim and '_ty' not in coord_sim:
                        sn = sn*180/np.pi
                        se = se*180/np.pi
                        ax.plot(x, sn, color='orange', linestyle='--')
                        ax.plot(x, se, color='purple', linestyle='--')
                        # ax.plot(x, yt1*180/np.pi, color='black')
                        # ax.plot(x, yt2*180/np.pi, color='grey', label='9Tight Solution')
                    else:    
                        ax.plot(x, sn, color='orange', linestyle='--')
                        ax.plot(x, se, color='purple', linestyle='--')
                        # ax.plot(x, yt1, color='black')
                        # ax.plot(x, yt2, color='grey', label='9Tight Solution')
                    # figure out some error metrics here - and print
                    # print('Coordinate: ', coord)
                    tempcoordRMSE_nat = np.sqrt(np.mean((sn - yn_new)**2))
                    tempcoordcorr_nat = np.corrcoef(sn, yn_new)[0,1]
                    # import pdb; pdb.set_trace()
                    # tempcoordRMSE_asSTD_nat = (sn - yn_new)/(upperbound_n/2)
                    tempcoordRMSE_nat_asSTD = np.sqrt(np.mean((np.abs((sn - yn_new))/np.abs((upperbound_n/2)))**2))
                    
                    # tempcoordRMSE_exo = np.sqrt(np.mean((se - ye)**2))
                    # print('Nat RMSE: ' + str(tempcoordRMSE_nat))
                    # print('Exo RMSE: ' + str(tempcoordRMSE_exo) + '\n')
                    rmse_values.append({'Coordinate': coord, 'Nat RMSE': tempcoordRMSE_nat}) # , 'Exo RMSE': tempcoordRMSE_exo})
                    rmse_values_asSTD.append({'Coordinate': coord, 'Nat RMSE as STD': tempcoordRMSE_nat_asSTD}) # , 'Exo RMSE as STD': tempcoordRMSE_exo_asSTD})
                    corr_values.append({'Coordinate': coord, 'Nat Correlation': tempcoordcorr_nat}) # , 'Exo Correlation': tempcoordcorr_exo})
            # format the final parts
            ax.set_title(coord, fontsize=12)
            ax.set_xlabel('GC%')
            ax.set_ylabel('Coordinate Value (deg or m)')

        if i >= len(labels2D):
            # now do the plotting
            # yn_new = np.array(mean_nat[coord])
            # ye = np.array(mean_exo[coord])
            # now do the plotting - reference data first
            yn_new = ikNat2D.getDependentColumn(coord).to_numpy()
            # ye = ikexo2D.getDependentColumn(coord).to_numpy()
            # if yn_new.size != 101:
            yn_new = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(yn_new)), yn_new)
            # ye = np.interp(np.linspace(0, 100, simlen), np.linspace(0, 100, len(ye)), ye)
            lowerbound_n = yn_new - np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
            upperbound_n = yn_new + np.interp(x, np.linspace(0,100,len((2*(np.array(std_nat[coord]))))), (2*(np.array(std_nat[coord]))))
            # lowerbound_e = ye - np.interp(x, np.linspace(0,100,101), (2*(np.array(std_exo[coord]))))
            # upperbound_e = ye + np.interp(x, np.linspace(0,100,101), (2*(np.array(std_exo[coord]))))
            ax.fill_between(x, lowerbound_n, upperbound_n, color='orange', alpha=0.1, label='Pop. 2 StDev. Natural')
            # ax.fill_between(x, lowerbound_e, upperbound_e, color='purple', alpha=0.1, label='Pop. 2 StDev. Exo')
            ax.plot(x, yn_new, color='orange', label='Natural Tracked Data')
            # ax.plot(x, ye, color='purple', label='Exo Tracked Data')
            ax.plot(x, sn, color='orange', linestyle='--', label='Nat Sim.')
            ax.plot(x, se, color='purple', linestyle='--', label='Exo Sim.')
            ax.set_title(coord, fontsize=12)
            ax.set_xlabel('GC%')
            ax.set_ylabel('Coordinate Value (deg or m)')
            # ax.legend()
            
    figurePath = os.getcwd() + '\\..\\..\\analysis\\'
    rmse_df = pd.DataFrame(rmse_values)
    rmse_df.to_csv(figurePath + 'coordinate_RMSE_SimVS2DIK' + '_40' + '.csv')
    print(rmse_df)

    corr_df = pd.DataFrame(corr_values)
    corr_df.to_csv(figurePath + 'coordinate_Correlation_SimVS2DIK' + '_40' + '.csv')
    print(corr_df)

    rmseSTD_df = pd.DataFrame(rmse_values_asSTD)
    rmseSTD_df.to_csv(figurePath + 'coordinate_RMSE_asSTD_SimVS2DIK' + '_40' + '.csv')
    print(rmseSTD_df)

    plt.tight_layout()
    plt.savefig(figurePath + 'coordinate_variations_SimVS2DIK_2STDEV' + '_40' + '.png')
    plt.show()
    return

# create a function for plotting a nice figure for use as a paper validation figure. Focus on saggital metrics. 
def saggitalValidationFigure27(simNat, simExo, iknat2D, ikexo2D, labels2D, coordinates_sim_clean, mean_nat, std_nat, mean_exo, std_exo, GRFsimnat, GRFsimexo, GRFrefnat, GRFrefexo, meangrfnat, stdgrfnat, meangrfexo, stdgrfexo, natmomentfile, exomomentfile, idnat, idexo, meanmomnat, stdmomnat, meanmomexo, stdmomexo, modelfile):
    # load the model and get the mass
    model = osim.Model(modelfile)
    mass = model.getTotalMass(model.initSystem())
    height = 1.78
    ## starting with the kinematics
    # get the length of the simulation data
    kin_simlen_nat = len(simNat.getIndependentColumn())
    kin_simlen_exo = len(simExo.getIndependentColumn())
    if kin_simlen_nat != kin_simlen_exo:
        print('Simulation kinematic data lengths do not match. Exiting.')
        return
    # get the sim kinematics
    sim_hip_angle_nat = simNat.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_hip_angle_exo = simExo.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_nat = simNat.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_knee_angle_exo = simExo.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_nat = simNat.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_ankle_angle_exo = simExo.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_ty_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    sim_ty_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    sim_hip_angle_nat_l = simNat.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_hip_angle_exo_l = simExo.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_nat_l = simNat.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_knee_angle_exo_l = simExo.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_nat_l = simNat.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ankle_angle_exo_l = simExo.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_pelvis_tilt_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvis_tilt_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvis_list_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvis_list_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvis_rotation_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_pelvis_rotation_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbar_extension_nat = simNat.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()
    sim_lumbar_extension_exo = simExo.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()
    sim_lumbar_bending_nat = simNat.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbar_bending_exo = simExo.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbar_rotation_nat = simNat.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_lumbar_rotation_exo = simExo.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_arm_flex_nat = simNat.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy()
    sim_arm_flex_exo = simExo.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy()
    sim_arm_flex_nat_l = simNat.getDependentColumn('/jointset/acromial_l/arm_flex_l/value').to_numpy()
    sim_arm_flex_exo_l = simExo.getDependentColumn('/jointset/acromial_l/arm_flex_l/value').to_numpy()
    sim_elbow_flex_nat = simNat.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy()
    sim_elbow_flex_exo = simExo.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy()
    sim_elbow_flex_nat_l = simNat.getDependentColumn('/jointset/elbow_l/elbow_flex_l/value').to_numpy()
    sim_elbow_flex_exo_l = simExo.getDependentColumn('/jointset/elbow_l/elbow_flex_l/value').to_numpy()
    
    
    # get the reference data
    ref_hip_angle_nat = iknat2D.getDependentColumn('hip_flexion_r').to_numpy()
    ref_hip_angle_exo = ikexo2D.getDependentColumn('hip_flexion_r').to_numpy()
    ref_knee_angle_nat = iknat2D.getDependentColumn('knee_angle_r').to_numpy()
    ref_knee_angle_exo = ikexo2D.getDependentColumn('knee_angle_r').to_numpy()
    ref_ankle_angle_nat = iknat2D.getDependentColumn('ankle_angle_r').to_numpy()
    ref_ankle_angle_exo = ikexo2D.getDependentColumn('ankle_angle_r').to_numpy()
    ref_ty_nat = iknat2D.getDependentColumn('pelvis_ty').to_numpy()
    ref_ty_exo = ikexo2D.getDependentColumn('pelvis_ty').to_numpy()
    ref_hip_angle_nat_l = iknat2D.getDependentColumn('hip_flexion_l').to_numpy()
    ref_hip_angle_exo_l = ikexo2D.getDependentColumn('hip_flexion_l').to_numpy()
    ref_knee_angle_nat_l = iknat2D.getDependentColumn('knee_angle_l').to_numpy()
    ref_knee_angle_exo_l = ikexo2D.getDependentColumn('knee_angle_l').to_numpy()
    ref_ankle_angle_nat_l = iknat2D.getDependentColumn('ankle_angle_l').to_numpy()
    ref_ankle_angle_exo_l = ikexo2D.getDependentColumn('ankle_angle_l').to_numpy()
    ref_pelvis_tilt_nat = iknat2D.getDependentColumn('pelvis_tilt').to_numpy()
    ref_pelvis_tilt_exo = ikexo2D.getDependentColumn('pelvis_tilt').to_numpy()
    ref_pelvis_list_nat = iknat2D.getDependentColumn('pelvis_list').to_numpy()
    ref_pelvis_list_exo = ikexo2D.getDependentColumn('pelvis_list').to_numpy()
    ref_pelvis_rotation_nat = iknat2D.getDependentColumn('pelvis_rotation').to_numpy()
    ref_pelvis_rotation_exo = ikexo2D.getDependentColumn('pelvis_rotation').to_numpy()
    ref_lumbar_extension_nat = iknat2D.getDependentColumn('lumbar_extension').to_numpy()
    ref_lumbar_extension_exo = ikexo2D.getDependentColumn('lumbar_extension').to_numpy()
    ref_lumbar_bending_nat = iknat2D.getDependentColumn('lumbar_bending').to_numpy()
    ref_lumbar_bending_exo = ikexo2D.getDependentColumn('lumbar_bending').to_numpy()
    ref_lumbar_rotation_nat = iknat2D.getDependentColumn('lumbar_rotation').to_numpy()
    ref_lumbar_rotation_exo = ikexo2D.getDependentColumn('lumbar_rotation').to_numpy()
    ref_arm_flex_nat = iknat2D.getDependentColumn('arm_flex_r').to_numpy()
    ref_arm_flex_exo = ikexo2D.getDependentColumn('arm_flex_r').to_numpy()
    ref_arm_flex_nat_l = iknat2D.getDependentColumn('arm_flex_l').to_numpy()
    ref_arm_flex_exo_l = ikexo2D.getDependentColumn('arm_flex_l').to_numpy()
    ref_elbow_flex_nat = iknat2D.getDependentColumn('elbow_flex_r').to_numpy()
    ref_elbow_flex_exo = ikexo2D.getDependentColumn('elbow_flex_r').to_numpy()
    ref_elbow_flex_nat_l = iknat2D.getDependentColumn('elbow_flex_l').to_numpy()
    ref_elbow_flex_exo_l = ikexo2D.getDependentColumn('elbow_flex_l').to_numpy()
    
    # get the length of the reference data
    kin_reflen_nat = len(ref_hip_angle_nat)
    kin_reflen_exo = len(ref_hip_angle_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation. 
    kin_xsim = np.linspace(0,100,kin_simlen_nat)
    kin_xref_nat = np.linspace(0,100,kin_reflen_nat)
    kin_xref_exo = np.linspace(0,100,kin_reflen_exo)



    
    # interpolate the reference data to the simulation data length
    ref_hip_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_nat)
    ref_hip_angle_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_exo)
    ref_knee_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_nat)
    ref_knee_angle_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_exo)
    ref_ankle_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_nat)
    ref_ankle_angle_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_exo)
    ref_ty_nat = np.interp(kin_xsim, kin_xref_nat, ref_ty_nat)
    ref_ty_exo = np.interp(kin_xsim, kin_xref_exo, ref_ty_exo)
    ref_hip_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_nat_l)
    ref_hip_angle_exo_l = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_exo_l)
    ref_knee_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_nat_l)
    ref_knee_angle_exo_l = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_exo_l)
    ref_ankle_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_nat_l)
    ref_ankle_angle_exo_l = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_exo_l)
    ref_pelvis_tilt_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvis_tilt_nat)
    ref_pelvis_tilt_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvis_tilt_exo)
    ref_pelvis_list_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvis_list_nat)
    ref_pelvis_list_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvis_list_exo)
    ref_pelvis_rotation_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvis_rotation_nat)
    ref_pelvis_rotation_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvis_rotation_exo)
    ref_lumbar_extension_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbar_extension_nat)
    ref_lumbar_extension_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbar_extension_exo)
    ref_lumbar_bending_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbar_bending_nat)
    ref_lumbar_bending_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbar_bending_exo)
    ref_lumbar_rotation_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbar_rotation_nat)
    ref_lumbar_rotation_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbar_rotation_exo)
    ref_arm_flex_nat = np.interp(kin_xsim, kin_xref_nat, ref_arm_flex_nat)
    ref_arm_flex_exo = np.interp(kin_xsim, kin_xref_exo, ref_arm_flex_exo)
    ref_arm_flex_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_arm_flex_nat_l)
    ref_arm_flex_exo_l = np.interp(kin_xsim, kin_xref_exo, ref_arm_flex_exo_l)
    ref_elbow_flex_nat = np.interp(kin_xsim, kin_xref_nat, ref_elbow_flex_nat)
    ref_elbow_flex_exo = np.interp(kin_xsim, kin_xref_exo, ref_elbow_flex_exo)
    ref_elbow_flex_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_elbow_flex_nat_l)
    ref_elbow_flex_exo_l = np.interp(kin_xsim, kin_xref_exo, ref_elbow_flex_exo_l)

    # normalize the ty values to the height of the subject
    ref_ty_nat = ref_ty_nat / height
    ref_ty_exo = ref_ty_exo / height
    sim_ty_nat = sim_ty_nat / height
    sim_ty_exo = sim_ty_exo / height

    ## get the GRF data
    grfsimnat = osim.TimeSeriesTable(GRFsimnat)
    grfsimexo = osim.TimeSeriesTable(GRFsimexo)
    grfrefnat = osim.TimeSeriesTable(GRFrefnat)
    grfrefexo = osim.TimeSeriesTable(GRFrefexo)
    # get the length of the simulation data
    grf_simlen_nat = len(grfsimnat.getIndependentColumn())
    grf_simlen_exo = len(grfsimexo.getIndependentColumn())
    if grf_simlen_nat != grf_simlen_exo:
        print('Simulation GRF data lengths do not match. Exiting.')
        return
    # get the sim GRF data
    sim_grf_y_nat = grfsimnat.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_y_exo = grfsimexo.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_x_nat = grfsimnat.getDependentColumn('ground_force_r_vx').to_numpy()
    sim_grf_x_exo = grfsimexo.getDependentColumn('ground_force_r_vx').to_numpy()
    # get the reference GRF data
    ref_grf_y_nat = grfrefnat.getDependentColumn('rF_y').to_numpy()
    ref_grf_y_exo = grfrefexo.getDependentColumn('rF_y').to_numpy()
    ref_grf_x_nat = grfrefnat.getDependentColumn('rF_x').to_numpy()
    ref_grf_x_exo = grfrefexo.getDependentColumn('rF_x').to_numpy()
    # ref_grf_y_nat = meangrfnat['calcn_r_Right_GRF_Fy']
    # ref_grf_y_exo = meangrfexo['calcn_r_Right_GRF_Fy']
    # ref_grf_x_nat = meangrfnat['calcn_r_Right_GRF_Fx']
    # ref_grf_x_exo = meangrfexo['calcn_r_Right_GRF_Fx']
    # get the length of the reference data
    grf_reflen_nat = len(ref_grf_y_nat)
    grf_reflen_exo = len(ref_grf_y_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    grf_xsim = np.linspace(0,100,grf_simlen_nat)
    grf_xref_nat = np.linspace(0,100,grf_reflen_nat)
    grf_xref_exo = np.linspace(0,100,grf_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_grf_y_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_y_nat)
    ref_grf_y_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_y_exo)
    ref_grf_x_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_x_nat)
    ref_grf_x_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_x_exo)
    # divide all of the GRF data based on the mass of the model
    sim_grf_y_nat = sim_grf_y_nat/(mass*9.81)
    sim_grf_y_exo = sim_grf_y_exo/(mass*9.81)
    sim_grf_x_nat = sim_grf_x_nat/(mass*9.81)
    sim_grf_x_exo = sim_grf_x_exo/(mass*9.81)
    ref_grf_y_nat = ref_grf_y_nat/(mass*9.81)
    ref_grf_y_exo = ref_grf_y_exo/(mass*9.81)
    ref_grf_x_nat = ref_grf_x_nat/(mass*9.81)
    ref_grf_x_exo = ref_grf_x_exo/(mass*9.81)
    
    ## get the moment data
    natmoment = osim.TimeSeriesTable(natmomentfile)
    exomoment = osim.TimeSeriesTable(exomomentfile)
    natrefmoment = osim.TimeSeriesTable(idnat)
    exorefmoment = osim.TimeSeriesTable(idexo)
    # get the length of the simulation data
    moment_simlen_nat = len(natmoment.getIndependentColumn())
    moment_simlen_exo = len(exomoment.getIndependentColumn())
    if moment_simlen_nat != moment_simlen_exo:
        print('Simulation moment data lengths do not match. Exiting.')
        return
    
    # get the sim moment data
    sim_hip_moment_nat = natmoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_hip_moment_exo = exomoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_knee_moment_nat = natmoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_knee_moment_exo = exomoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_ankle_moment_nat = natmoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_ankle_moment_exo = exomoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_hip_moment_nat_l = natmoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_hip_moment_exo_l = exomoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_knee_moment_nat_l = natmoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_knee_moment_exo_l = exomoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_ankle_moment_nat_l = natmoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_ankle_moment_exo_l = exomoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_pelvis_tilt_moment_nat = natmoment.getDependentColumn('pelvis_tilt_moment').to_numpy()
    sim_pelvis_tilt_moment_exo = exomoment.getDependentColumn('pelvis_tilt_moment').to_numpy()
    sim_pelvis_list_moment_nat = natmoment.getDependentColumn('pelvis_list_moment').to_numpy()
    sim_pelvis_list_moment_exo = exomoment.getDependentColumn('pelvis_list_moment').to_numpy()
    sim_pelvis_rotation_moment_nat = natmoment.getDependentColumn('pelvis_rotation_moment').to_numpy()
    sim_pelvis_rotation_moment_exo = exomoment.getDependentColumn('pelvis_rotation_moment').to_numpy()
    sim_lumbar_extension_moment_nat = natmoment.getDependentColumn('lumbar_extension_moment').to_numpy()
    sim_lumbar_extension_moment_exo = exomoment.getDependentColumn('lumbar_extension_moment').to_numpy()
    sim_lumbar_bending_moment_nat = natmoment.getDependentColumn('lumbar_bending_moment').to_numpy()
    sim_lumbar_bending_moment_exo = exomoment.getDependentColumn('lumbar_bending_moment').to_numpy()
    sim_lumbar_rotation_moment_nat = natmoment.getDependentColumn('lumbar_rotation_moment').to_numpy()
    sim_lumbar_rotation_moment_exo = exomoment.getDependentColumn('lumbar_rotation_moment').to_numpy()
    sim_arm_flex_moment_nat = natmoment.getDependentColumn('arm_flex_r_moment').to_numpy()
    sim_arm_flex_moment_exo = exomoment.getDependentColumn('arm_flex_r_moment').to_numpy()
    sim_arm_flex_moment_nat_l = natmoment.getDependentColumn('arm_flex_l_moment').to_numpy()
    sim_arm_flex_moment_exo_l = exomoment.getDependentColumn('arm_flex_l_moment').to_numpy()
    sim_elbow_flex_moment_nat = natmoment.getDependentColumn('elbow_flex_r_moment').to_numpy()
    sim_elbow_flex_moment_exo = exomoment.getDependentColumn('elbow_flex_r_moment').to_numpy()
    sim_elbow_flex_moment_nat_l = natmoment.getDependentColumn('elbow_flex_l_moment').to_numpy()
    sim_elbow_flex_moment_exo_l = exomoment.getDependentColumn('elbow_flex_l_moment').to_numpy()
    
    
    # get the reference moment data
    ref_hip_moment_nat = idnat.getDependentColumn('hip_flexion_r_moment').to_numpy()
    ref_hip_moment_exo = idexo.getDependentColumn('hip_flexion_r_moment').to_numpy()
    ref_knee_moment_nat = idnat.getDependentColumn('knee_angle_r_moment').to_numpy()
    ref_knee_moment_exo = idexo.getDependentColumn('knee_angle_r_moment').to_numpy()
    ref_ankle_moment_nat = idnat.getDependentColumn('ankle_angle_r_moment').to_numpy()
    ref_ankle_moment_exo = idexo.getDependentColumn('ankle_angle_r_moment').to_numpy()
    ref_hip_moment_nat_l = idnat.getDependentColumn('hip_flexion_l_moment').to_numpy()
    ref_hip_moment_exo_l = idexo.getDependentColumn('hip_flexion_l_moment').to_numpy()
    ref_knee_moment_nat_l = idnat.getDependentColumn('knee_angle_l_moment').to_numpy()
    ref_knee_moment_exo_l = idexo.getDependentColumn('knee_angle_l_moment').to_numpy()
    ref_ankle_moment_nat_l = idnat.getDependentColumn('ankle_angle_l_moment').to_numpy()
    ref_ankle_moment_exo_l = idexo.getDependentColumn('ankle_angle_l_moment').to_numpy()
    ref_pelvis_tilt_moment_nat = idnat.getDependentColumn('pelvis_tilt_moment').to_numpy()
    ref_pelvis_tilt_moment_exo = idexo.getDependentColumn('pelvis_tilt_moment').to_numpy()
    ref_pelvis_list_moment_nat = idnat.getDependentColumn('pelvis_list_moment').to_numpy()
    ref_pelvis_list_moment_exo = idexo.getDependentColumn('pelvis_list_moment').to_numpy()
    ref_pelvis_rotation_moment_nat = idnat.getDependentColumn('pelvis_rotation_moment').to_numpy()
    ref_pelvis_rotation_moment_exo = idexo.getDependentColumn('pelvis_rotation_moment').to_numpy()
    ref_lumbar_extension_moment_nat = idnat.getDependentColumn('lumbar_extension_moment').to_numpy()
    ref_lumbar_extension_moment_exo = idexo.getDependentColumn('lumbar_extension_moment').to_numpy()
    ref_lumbar_bending_moment_nat = idnat.getDependentColumn('lumbar_bending_moment').to_numpy()
    ref_lumbar_bending_moment_exo = idexo.getDependentColumn('lumbar_bending_moment').to_numpy()
    ref_lumbar_rotation_moment_nat = idnat.getDependentColumn('lumbar_rotation_moment').to_numpy()
    ref_lumbar_rotation_moment_exo = idexo.getDependentColumn('lumbar_rotation_moment').to_numpy()
    ref_arm_flex_moment_nat = idnat.getDependentColumn('arm_flex_r_moment').to_numpy()
    ref_arm_flex_moment_exo = idexo.getDependentColumn('arm_flex_r_moment').to_numpy()
    ref_arm_flex_moment_nat_l = idnat.getDependentColumn('arm_flex_l_moment').to_numpy()
    ref_arm_flex_moment_exo_l = idexo.getDependentColumn('arm_flex_l_moment').to_numpy()
    ref_elbow_flex_moment_nat = idnat.getDependentColumn('elbow_flex_r_moment').to_numpy()
    ref_elbow_flex_moment_exo = idexo.getDependentColumn('elbow_flex_r_moment').to_numpy()
    ref_elbow_flex_moment_nat_l = idnat.getDependentColumn('elbow_flex_l_moment').to_numpy()
    ref_elbow_flex_moment_exo_l = idexo.getDependentColumn('elbow_flex_l_moment').to_numpy()

    # get the length of the reference data
    moment_reflen_nat = len(ref_hip_moment_nat)
    moment_reflen_exo = len(ref_hip_moment_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    moment_xsim = np.linspace(0,100,moment_simlen_nat)
    moment_xref_nat = np.linspace(0,100,moment_reflen_nat)
    moment_xref_exo = np.linspace(0,100,moment_reflen_exo)
    
    # interpolate the reference data to the simulation data length
    ref_hip_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_nat)
    ref_hip_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_exo)
    ref_knee_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_nat)
    ref_knee_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_exo)
    ref_ankle_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_nat)
    ref_ankle_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_exo)
    ref_hip_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_nat_l)
    ref_hip_moment_exo_l = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_exo_l)
    ref_knee_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_nat_l)
    ref_knee_moment_exo_l = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_exo_l)
    ref_ankle_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_nat_l)
    ref_ankle_moment_exo_l = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_exo_l)
    ref_pelvis_tilt_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvis_tilt_moment_nat)
    ref_pelvis_tilt_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_pelvis_tilt_moment_exo)
    ref_pelvis_list_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvis_list_moment_nat)
    ref_pelvis_list_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_pelvis_list_moment_exo)
    ref_pelvis_rotation_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvis_rotation_moment_nat)
    ref_pelvis_rotation_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_pelvis_rotation_moment_exo)
    ref_lumbar_extension_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbar_extension_moment_nat)
    ref_lumbar_extension_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_lumbar_extension_moment_exo)
    ref_lumbar_bending_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbar_bending_moment_nat)
    ref_lumbar_bending_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_lumbar_bending_moment_exo)
    ref_lumbar_rotation_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbar_rotation_moment_nat)
    ref_lumbar_rotation_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_lumbar_rotation_moment_exo)
    ref_arm_flex_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_arm_flex_moment_nat)
    ref_arm_flex_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_arm_flex_moment_exo)
    ref_arm_flex_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_arm_flex_moment_nat_l)
    ref_arm_flex_moment_exo_l = np.interp(moment_xsim, moment_xref_exo, ref_arm_flex_moment_exo_l)
    ref_elbow_flex_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_elbow_flex_moment_nat)
    ref_elbow_flex_moment_exo = np.interp(moment_xsim, moment_xref_exo, ref_elbow_flex_moment_exo)
    ref_elbow_flex_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_elbow_flex_moment_nat_l)
    ref_elbow_flex_moment_exo_l = np.interp(moment_xsim, moment_xref_exo, ref_elbow_flex_moment_exo_l)
    
    # normalize all of it to body mass
    sim_hip_moment_nat = sim_hip_moment_nat/mass    
    sim_hip_moment_exo = sim_hip_moment_exo/mass
    sim_knee_moment_nat = sim_knee_moment_nat/mass
    sim_knee_moment_exo = sim_knee_moment_exo/mass
    sim_ankle_moment_nat = sim_ankle_moment_nat/mass
    sim_ankle_moment_exo = sim_ankle_moment_exo/mass
    sim_hip_moment_nat_l = sim_hip_moment_nat_l/mass
    sim_hip_moment_exo_l = sim_hip_moment_exo_l/mass
    sim_knee_moment_nat_l = sim_knee_moment_nat_l/mass
    sim_knee_moment_exo_l = sim_knee_moment_exo_l/mass
    sim_ankle_moment_nat_l = sim_ankle_moment_nat_l/mass
    sim_ankle_moment_exo_l = sim_ankle_moment_exo_l/mass
    sim_pelvis_tilt_moment_nat = sim_pelvis_tilt_moment_nat/mass
    sim_pelvis_tilt_moment_exo = sim_pelvis_tilt_moment_exo/mass
    sim_pelvis_list_moment_nat = sim_pelvis_list_moment_nat/mass
    sim_pelvis_list_moment_exo = sim_pelvis_list_moment_exo/mass
    sim_pelvis_rotation_moment_nat = sim_pelvis_rotation_moment_nat/mass
    sim_pelvis_rotation_moment_exo = sim_pelvis_rotation_moment_exo/mass
    sim_lumbar_extension_moment_nat = sim_lumbar_extension_moment_nat/mass
    sim_lumbar_extension_moment_exo = sim_lumbar_extension_moment_exo/mass
    sim_lumbar_bending_moment_nat = sim_lumbar_bending_moment_nat/mass
    sim_lumbar_bending_moment_exo = sim_lumbar_bending_moment_exo/mass
    sim_lumbar_rotation_moment_nat = sim_lumbar_rotation_moment_nat/mass
    sim_lumbar_rotation_moment_exo = sim_lumbar_rotation_moment_exo/mass
    sim_arm_flex_moment_nat = sim_arm_flex_moment_nat/mass
    sim_arm_flex_moment_exo = sim_arm_flex_moment_exo/mass
    sim_arm_flex_moment_nat_l = sim_arm_flex_moment_nat_l/mass
    sim_arm_flex_moment_exo_l = sim_arm_flex_moment_exo_l/mass
    sim_elbow_flex_moment_nat = sim_elbow_flex_moment_nat/mass
    sim_elbow_flex_moment_exo = sim_elbow_flex_moment_exo/mass
    sim_elbow_flex_moment_nat_l = sim_elbow_flex_moment_nat_l/mass
    sim_elbow_flex_moment_exo_l = sim_elbow_flex_moment_exo_l/mass

    ref_hip_moment_nat = ref_hip_moment_nat/mass
    ref_hip_moment_exo = ref_hip_moment_exo/mass
    ref_knee_moment_nat = ref_knee_moment_nat/mass
    ref_knee_moment_exo = ref_knee_moment_exo/mass
    ref_ankle_moment_nat = ref_ankle_moment_nat/mass
    ref_ankle_moment_exo = ref_ankle_moment_exo/mass
    ref_hip_moment_nat_l = ref_hip_moment_nat_l/mass
    ref_hip_moment_exo_l = ref_hip_moment_exo_l/mass
    ref_knee_moment_nat_l = ref_knee_moment_nat_l/mass
    ref_knee_moment_exo_l = ref_knee_moment_exo_l/mass
    ref_ankle_moment_nat_l = ref_ankle_moment_nat_l/mass
    ref_ankle_moment_exo_l = ref_ankle_moment_exo_l/mass
    ref_pelvis_tilt_moment_nat = ref_pelvis_tilt_moment_nat/mass
    ref_pelvis_tilt_moment_exo = ref_pelvis_tilt_moment_exo/mass
    ref_pelvis_list_moment_nat = ref_pelvis_list_moment_nat/mass
    ref_pelvis_list_moment_exo = ref_pelvis_list_moment_exo/mass
    ref_pelvis_rotation_moment_nat = ref_pelvis_rotation_moment_nat/mass
    ref_pelvis_rotation_moment_exo = ref_pelvis_rotation_moment_exo/mass
    ref_lumbar_extension_moment_nat = ref_lumbar_extension_moment_nat/mass
    ref_lumbar_extension_moment_exo = ref_lumbar_extension_moment_exo/mass
    ref_lumbar_bending_moment_nat = ref_lumbar_bending_moment_nat/mass
    ref_lumbar_bending_moment_exo = ref_lumbar_bending_moment_exo/mass
    ref_lumbar_rotation_moment_nat = ref_lumbar_rotation_moment_nat/mass
    ref_lumbar_rotation_moment_exo = ref_lumbar_rotation_moment_exo/mass
    ref_arm_flex_moment_nat = ref_arm_flex_moment_nat/mass
    ref_arm_flex_moment_exo = ref_arm_flex_moment_exo/mass
    ref_arm_flex_moment_nat_l = ref_arm_flex_moment_nat_l/mass
    ref_arm_flex_moment_exo_l = ref_arm_flex_moment_exo_l/mass
    ref_elbow_flex_moment_nat = ref_elbow_flex_moment_nat/mass
    ref_elbow_flex_moment_exo = ref_elbow_flex_moment_exo/mass
    ref_elbow_flex_moment_nat_l = ref_elbow_flex_moment_nat_l/mass
    ref_elbow_flex_moment_exo_l =ref_elbow_flex_moment_exo_l / mass

    #############################
    # Simple plot: kinematic measures (sim vs ref) across gait cycle.
    kin_plot_vars = [
        ('Hip Angle R', sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat, 'deg'),
        ('Hip Angle L', sim_hip_angle_nat_l * 180 / np.pi, ref_hip_angle_nat_l, 'deg'),
        ('Knee Angle R', sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat, 'deg'),
        ('Knee Angle L', sim_knee_angle_nat_l * 180 / np.pi, ref_knee_angle_nat_l, 'deg'),
        ('Ankle Angle R', sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat, 'deg'),
        ('Ankle Angle L', sim_ankle_angle_nat_l * 180 / np.pi, ref_ankle_angle_nat_l, 'deg'),
        ('Pelvis Vertical', sim_ty_nat, ref_ty_nat, 'm/height'),
        ('Pelvis Tilt', sim_pelvis_tilt_nat * 180 / np.pi, ref_pelvis_tilt_nat, 'deg'),
        ('Pelvis List', sim_pelvis_list_nat * 180 / np.pi, ref_pelvis_list_nat, 'deg'),
        ('Pelvis Rotation', sim_pelvis_rotation_nat * 180 / np.pi, ref_pelvis_rotation_nat, 'deg'),
        ('Lumbar Extension', sim_lumbar_extension_nat * 180 / np.pi, ref_lumbar_extension_nat, 'deg'),
        ('Lumbar Bending', sim_lumbar_bending_nat * 180 / np.pi, ref_lumbar_bending_nat, 'deg'),
        ('Lumbar Rotation', sim_lumbar_rotation_nat * 180 / np.pi, ref_lumbar_rotation_nat, 'deg'),
        ('Arm Flex R', sim_arm_flex_nat * 180 / np.pi, ref_arm_flex_nat, 'deg'),
        ('Arm Flex L', sim_arm_flex_nat_l * 180 / np.pi, ref_arm_flex_nat_l, 'deg'),
        ('Elbow Flex R', sim_elbow_flex_nat * 180 / np.pi, ref_elbow_flex_nat, 'deg'),
        ('Elbow Flex L', sim_elbow_flex_nat_l * 180 / np.pi, ref_elbow_flex_nat_l, 'deg'),
    ]

    kin_x = np.linspace(0, 100, kin_simlen_nat)
    fig_kin, ax_kin = plt.subplots(5, 5, figsize=(14, 12), dpi=300)
    ax_kin = ax_kin.flatten()
    for i, (label, sim_vals, ref_vals, units) in enumerate(kin_plot_vars):
        ax_kin[i].plot(kin_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_kin[i].plot(kin_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_kin[i].set_title(label, fontsize=10)
        ax_kin[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_kin[i].set_ylabel(units, fontsize=9)
        ax_kin[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_kin[i].legend(fontsize=8, loc='best')

    for j in range(len(kin_plot_vars), len(ax_kin)):
        ax_kin[j].axis('off')

    plt.tight_layout()
    plt.show()

    ############################
    # Simple plot: kinematic measures (sim vs ref) across gait cycle for exo
    kin_plot_vars = [
        ('Hip Angle R', sim_hip_angle_exo * 180 / np.pi, ref_hip_angle_exo, 'deg'),
        ('Hip Angle L', sim_hip_angle_exo_l * 180 / np.pi, ref_hip_angle_exo_l, 'deg'),
        ('Knee Angle R', sim_knee_angle_exo * 180 / np.pi, ref_knee_angle_exo, 'deg'),
        ('Knee Angle L', sim_knee_angle_exo_l * 180 / np.pi, ref_knee_angle_exo_l, 'deg'),
        ('Ankle Angle R', sim_ankle_angle_exo * 180 / np.pi, ref_ankle_angle_exo, 'deg'),
        ('Ankle Angle L', sim_ankle_angle_exo_l * 180 / np.pi, ref_ankle_angle_exo_l, 'deg'),
        ('Pelvis Vertical', sim_ty_exo, ref_ty_exo, 'm/height'),
        ('Pelvis Tilt', sim_pelvis_tilt_exo * 180 / np.pi, ref_pelvis_tilt_exo, 'deg'),
        ('Pelvis List', sim_pelvis_list_exo * 180 / np.pi, ref_pelvis_list_exo, 'deg'),
        ('Pelvis Rotation', sim_pelvis_rotation_exo * 180 / np.pi, ref_pelvis_rotation_exo, 'deg'),
        ('Lumbar Extension', sim_lumbar_extension_exo * 180 / np.pi, ref_lumbar_extension_exo, 'deg'),
        ('Lumbar Bending', sim_lumbar_bending_exo * 180 / np.pi, ref_lumbar_bending_exo, 'deg'),
        ('Lumbar Rotation', sim_lumbar_rotation_exo * 180 / np.pi, ref_lumbar_rotation_exo, 'deg'),
        ('Arm Flex R', sim_arm_flex_exo * 180 / np.pi, ref_arm_flex_exo, 'deg'),
        ('Arm Flex L', sim_arm_flex_exo_l * 180 / np.pi, ref_arm_flex_exo_l, 'deg'),
        ('Elbow Flex R', sim_elbow_flex_exo * 180 / np.pi, ref_elbow_flex_exo, 'deg'),
        ('Elbow Flex L', sim_elbow_flex_exo_l * 180 / np.pi, ref_elbow_flex_exo_l, 'deg'),
    ]

    kin_x = np.linspace(0, 100, kin_simlen_nat)
    fig_kin, ax_kin = plt.subplots(5, 5, figsize=(14, 12), dpi=300)
    ax_kin = ax_kin.flatten()
    for i, (label, sim_vals, ref_vals, units) in enumerate(kin_plot_vars):
        ax_kin[i].plot(kin_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_kin[i].plot(kin_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_kin[i].set_title(label, fontsize=10)
        ax_kin[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_kin[i].set_ylabel(units, fontsize=9)
        ax_kin[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_kin[i].legend(fontsize=8, loc='best')

    for j in range(len(kin_plot_vars), len(ax_kin)):
        ax_kin[j].axis('off')

    plt.tight_layout()
    plt.show()


    ############################
    # Simple plot: moment measures (sim vs ref) across gait cycle for nat
    kin_plot_vars = [
        ('Hip Moment R', sim_hip_moment_nat, ref_hip_moment_nat, 'deg'),
        ('Hip Moment L', sim_hip_moment_nat_l, ref_hip_moment_nat_l, 'deg'),
        ('Knee Moment R', sim_knee_moment_nat, ref_knee_moment_nat, 'deg'),
        ('Knee Moment L', sim_knee_moment_nat_l, ref_knee_moment_nat_l, 'deg'),
        ('Ankle Moment R', sim_ankle_moment_nat, ref_ankle_moment_nat, 'deg'),
        ('Ankle Moment L', sim_ankle_moment_nat_l, ref_ankle_moment_nat_l, 'deg'),
        # ('Pelvis Tilt', sim_pelvis_tilt_moment_nat, ref_pelvis_tilt_moment_nat, 'deg'),
        # ('Pelvis List', sim_pelvis_list_moment_nat, ref_pelvis_list_moment_nat, 'deg'),
        # ('Pelvis Rotation', sim_pelvis_rotation_moment_nat, ref_pelvis_rotation_moment_nat, 'deg'),
        ('Lumbar Extension', sim_lumbar_extension_moment_nat, ref_lumbar_extension_moment_nat, 'deg'),
        ('Lumbar Bending', sim_lumbar_bending_moment_nat, ref_lumbar_bending_moment_nat, 'deg'),
        ('Lumbar Rotation', sim_lumbar_rotation_moment_nat, ref_lumbar_rotation_moment_nat, 'deg'),
        ('Arm Flex R', sim_arm_flex_moment_nat, ref_arm_flex_moment_nat, 'deg'),
        ('Arm Flex L', sim_arm_flex_moment_nat_l, ref_arm_flex_moment_nat_l, 'deg'),
        ('Elbow Flex R', sim_elbow_flex_moment_nat, ref_elbow_flex_moment_nat, 'deg'),
        ('Elbow Flex L', sim_elbow_flex_moment_nat_l, ref_elbow_flex_moment_nat_l, 'deg'),
    ]

    kin_x = np.linspace(0, 100, kin_simlen_nat)
    fig_kin, ax_kin = plt.subplots(5, 5, figsize=(14, 12), dpi=300)
    ax_kin = ax_kin.flatten()
    for i, (label, sim_vals, ref_vals, units) in enumerate(kin_plot_vars):
        ax_kin[i].plot(kin_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_kin[i].plot(kin_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_kin[i].set_title(label, fontsize=10)
        ax_kin[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_kin[i].set_ylabel(units, fontsize=9)
        ax_kin[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_kin[i].legend(fontsize=8, loc='best')

    for j in range(len(kin_plot_vars), len(ax_kin)):
        ax_kin[j].axis('off')

    plt.tight_layout()
    plt.show()


    ############################
    # Simple plot: moment measures (sim vs ref) across gait cycle for exo
    kin_plot_vars = [
        ('Hip Moment R', sim_hip_moment_exo, ref_hip_moment_exo, 'deg'),
        ('Hip Moment L', sim_hip_moment_exo_l, ref_hip_moment_exo_l, 'deg'),
        ('Knee Moment R', sim_knee_moment_exo, ref_knee_moment_exo, 'deg'),
        ('Knee Moment L', sim_knee_moment_exo_l, ref_knee_moment_exo_l, 'deg'),
        ('Ankle Moment R', sim_ankle_moment_exo, ref_ankle_moment_exo, 'deg'),
        ('Ankle Moment L', sim_ankle_moment_exo_l, ref_ankle_moment_exo_l, 'deg'),
        # ('Pelvis Tilt', sim_pelvis_tilt_moment_exo, ref_pelvis_tilt_moment_exo, 'deg'),
        # ('Pelvis List', sim_pelvis_list_moment_exo, ref_pelvis_list_moment_exo, 'deg'),
        # ('Pelvis Rotation', sim_pelvis_rotation_moment_exo, ref_pelvis_rotation_moment_exo, 'deg'),
        ('Lumbar Extension', sim_lumbar_extension_moment_exo, ref_lumbar_extension_moment_exo, 'deg'),
        ('Lumbar Bending', sim_lumbar_bending_moment_exo, ref_lumbar_bending_moment_exo, 'deg'),
        ('Lumbar Rotation', sim_lumbar_rotation_moment_exo, ref_lumbar_rotation_moment_exo, 'deg'),
        ('Arm Flex R', sim_arm_flex_moment_exo, ref_arm_flex_moment_exo, 'deg'),
        ('Arm Flex L', sim_arm_flex_moment_exo_l, ref_arm_flex_moment_exo_l, 'deg'),
        ('Elbow Flex R', sim_elbow_flex_moment_exo, ref_elbow_flex_moment_exo, 'deg'),
        ('Elbow Flex L', sim_elbow_flex_moment_exo_l, ref_elbow_flex_moment_exo_l, 'deg'),
    ]

    kin_x = np.linspace(0, 100, kin_simlen_nat)
    fig_kin, ax_kin = plt.subplots(5, 5, figsize=(14, 12), dpi=300)
    ax_kin = ax_kin.flatten()
    for i, (label, sim_vals, ref_vals, units) in enumerate(kin_plot_vars):
        ax_kin[i].plot(kin_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_kin[i].plot(kin_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_kin[i].set_title(label, fontsize=10)
        ax_kin[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_kin[i].set_ylabel(units, fontsize=9)
        ax_kin[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_kin[i].legend(fontsize=8, loc='best')

    for j in range(len(kin_plot_vars), len(ax_kin)):
        ax_kin[j].axis('off')

    plt.tight_layout()
    plt.show()


    ###################################
    # compute error metrics: RMSE for each variable.

    # compute RMSE for kinematics
    rmse_hip_angle_nat = np.sqrt(np.mean((sim_hip_angle_nat * 180 / np.pi - ref_hip_angle_nat) ** 2))
    rmse_hip_angle_exo = np.sqrt(np.mean((sim_hip_angle_exo * 180 / np.pi - ref_hip_angle_exo) ** 2))
    rmse_knee_angle_nat = np.sqrt(np.mean((sim_knee_angle_nat * 180 / np.pi - ref_knee_angle_nat) ** 2))
    rmse_knee_angle_exo = np.sqrt(np.mean((sim_knee_angle_exo * 180 / np.pi - ref_knee_angle_exo) ** 2))
    rmse_ankle_angle_nat = np.sqrt(np.mean((sim_ankle_angle_nat * 180 / np.pi - ref_ankle_angle_nat) ** 2))
    rmse_ankle_angle_exo = np.sqrt(np.mean((sim_ankle_angle_exo * 180 / np.pi - ref_ankle_angle_exo) ** 2))
    rmse_hip_angle_nat_l = np.sqrt(np.mean((sim_hip_angle_nat_l * 180 / np.pi - ref_hip_angle_nat_l) ** 2))
    rmse_hip_angle_exo_l = np.sqrt(np.mean((sim_hip_angle_exo_l * 180 / np.pi - ref_hip_angle_exo_l) ** 2))
    rmse_knee_angle_nat_l = np.sqrt(np.mean((sim_knee_angle_nat_l * 180 / np.pi - ref_knee_angle_nat_l) ** 2))
    rmse_knee_angle_exo_l = np.sqrt(np.mean((sim_knee_angle_exo_l * 180 / np.pi - ref_knee_angle_exo_l) ** 2))
    rmse_ankle_angle_nat_l = np.sqrt(np.mean((sim_ankle_angle_nat_l * 180 / np.pi - ref_ankle_angle_nat_l) ** 2))
    rmse_ankle_angle_exo_l = np.sqrt(np.mean((sim_ankle_angle_exo_l * 180 / np.pi - ref_ankle_angle_exo_l) ** 2))
    rmse_ty_nat = np.sqrt(np.mean((sim_ty_nat - ref_ty_nat) ** 2))
    rmse_ty_exo = np.sqrt(np.mean((sim_ty_exo - ref_ty_exo) ** 2))
    rmse_pelvis_tilt_nat = np.sqrt(np.mean((sim_pelvis_tilt_nat * 180 / np.pi - ref_pelvis_tilt_nat) ** 2))
    rmse_pelvis_list_nat = np.sqrt(np.mean((sim_pelvis_list_nat * 180 / np.pi - ref_pelvis_list_nat) ** 2))
    rmse_pelvis_rotation_nat = np.sqrt(np.mean((sim_pelvis_rotation_nat * 180 / np.pi - ref_pelvis_rotation_nat) ** 2))
    rmse_pelvis_tilt_exo = np.sqrt(np.mean((sim_pelvis_tilt_exo * 180 / np.pi - ref_pelvis_tilt_exo) ** 2))
    rmse_pelvis_list_exo = np.sqrt(np.mean((sim_pelvis_list_exo * 180 / np.pi - ref_pelvis_list_exo) ** 2))
    rmse_pelvis_rotation_exo = np.sqrt(np.mean((sim_pelvis_rotation_exo * 180 / np.pi - ref_pelvis_rotation_exo) ** 2))
    rmse_lumbar_extension_nat = np.sqrt(np.mean((sim_lumbar_extension_nat * 180 / np.pi - ref_lumbar_extension_nat) ** 2))
    rmse_lumbar_bending_nat = np.sqrt(np.mean((sim_lumbar_bending_nat * 180 / np.pi - ref_lumbar_bending_nat) ** 2))
    rmse_lumbar_rotation_nat = np.sqrt(np.mean((sim_lumbar_rotation_nat * 180 / np.pi - ref_lumbar_rotation_nat) ** 2))
    rmse_lumbar_extension_exo = np.sqrt(np.mean((sim_lumbar_extension_exo * 180 / np.pi - ref_lumbar_extension_exo) ** 2))
    rmse_lumbar_bending_exo = np.sqrt(np.mean((sim_lumbar_bending_exo * 180 / np.pi - ref_lumbar_bending_exo) ** 2))
    rmse_lumbar_rotation_exo = np.sqrt(np.mean((sim_lumbar_rotation_exo * 180 / np.pi - ref_lumbar_rotation_exo) ** 2))
    rmse_arm_flex_nat = np.sqrt(np.mean((sim_arm_flex_nat * 180 / np.pi - ref_arm_flex_nat) ** 2))
    rmse_arm_flex_nat_l = np.sqrt(np.mean((sim_arm_flex_nat_l * 180 / np.pi - ref_arm_flex_nat_l) ** 2))
    rmse_elbow_flex_nat = np.sqrt(np.mean((sim_elbow_flex_nat * 180 / np.pi - ref_elbow_flex_nat) ** 2))
    rmse_elbow_flex_nat_l = np.sqrt(np.mean((sim_elbow_flex_nat_l * 180 / np.pi - ref_elbow_flex_nat_l) ** 2))
    rmse_arm_flex_exo = np.sqrt(np.mean((sim_arm_flex_exo * 180 / np.pi - ref_arm_flex_exo) ** 2))
    rmse_arm_flex_exo_l = np.sqrt(np.mean((sim_arm_flex_exo_l * 180 / np.pi - ref_arm_flex_exo_l) ** 2))
    rmse_elbow_flex_exo = np.sqrt(np.mean((sim_elbow_flex_exo * 180 / np.pi - ref_elbow_flex_exo) ** 2))
    rmse_elbow_flex_exo_l = np.sqrt(np.mean((sim_elbow_flex_exo_l * 180 / np.pi - ref_elbow_flex_exo_l) ** 2))
    # compute average RMSE for kinematic 
    kin_rmse_nat = [rmse_hip_angle_nat, rmse_knee_angle_nat, rmse_ankle_angle_nat,
                    rmse_hip_angle_nat_l, rmse_knee_angle_nat_l, rmse_ankle_angle_nat_l,
                    # rmse_ty_nat, 
                    rmse_pelvis_tilt_nat, rmse_pelvis_list_nat, rmse_pelvis_rotation_nat, 
                    rmse_lumbar_extension_nat, rmse_lumbar_bending_nat, rmse_lumbar_rotation_nat,
                    rmse_arm_flex_nat, rmse_arm_flex_nat_l, rmse_elbow_flex_nat, rmse_elbow_flex_nat_l]
    avg_kin_rmse_nat = sum(kin_rmse_nat) / len(kin_rmse_nat)
    std_kin_rmse_nat = np.std(kin_rmse_nat)
    kin_musc_rmse_nat = [rmse_hip_angle_nat, rmse_knee_angle_nat, rmse_ankle_angle_nat,
                        rmse_hip_angle_nat_l, rmse_knee_angle_nat_l, rmse_ankle_angle_nat_l,
                        # rmse_ty_nat, 
                        # rmse_pelvis_tilt_nat, rmse_pelvis_list_nat, rmse_pelvis_rotation_nat, 
                        # rmse_lumbar_extension_nat, rmse_lumbar_bending_nat, rmse_lumbar_rotation_nat,
                        # rmse_arm_flex_nat, rmse_arm_flex_nat_l, rmse_elbow_flex_nat, rmse_elbow_flex_nat_l
                        ]
    avg_musc_kin_rmse_nat = sum(kin_musc_rmse_nat) / len(kin_musc_rmse_nat)
    std_musc_kin_rmse_nat = np.std(kin_musc_rmse_nat)

    kin_rmse_exo = [rmse_hip_angle_exo, rmse_knee_angle_exo, rmse_ankle_angle_exo,
                    rmse_hip_angle_exo_l, rmse_knee_angle_exo_l, rmse_ankle_angle_exo_l,
                    # rmse_ty_exo,
                    rmse_pelvis_tilt_exo, rmse_pelvis_list_exo, rmse_pelvis_rotation_exo,
                    rmse_lumbar_extension_exo, rmse_lumbar_bending_exo, rmse_lumbar_rotation_exo,
                    rmse_arm_flex_exo, rmse_arm_flex_exo_l, rmse_elbow_flex_exo, rmse_elbow_flex_exo_l]
    avg_kin_rmse_exo = sum(kin_rmse_exo) / len(kin_rmse_exo)
    std_kin_rmse_exo = np.std(kin_rmse_exo)
    kin_musc_rmse_exo = [rmse_hip_angle_exo, rmse_knee_angle_exo, rmse_ankle_angle_exo,
                        rmse_hip_angle_exo_l, rmse_knee_angle_exo_l, rmse_ankle_angle_exo_l,
                        # rmse_ty_exo,
                        # rmse_pelvis_tilt_exo, rmse_pelvis_list_exo, rmse_pelvis_rotation_exo,
                        # rmse_lumbar_extension_exo, rmse_lumbar_bending_exo, rmse_lumbar_rotation_exo,
                        # rmse_arm_flex_exo, rmse_arm_flex_exo_l, rmse_elbow_flex_exo, rmse_elbow_flex_exo_l
                        ]
    avg_musc_kin_rmse_exo = sum(kin_musc_rmse_exo) / len(kin_musc_rmse_exo)
    std_musc_kin_rmse_exo = np.std(kin_musc_rmse_exo)

    # average total for nat and exo 
    avg_total_kin_rmse = (avg_kin_rmse_nat + avg_kin_rmse_exo) / 2
    std_total_kin_rmse = np.std(kin_rmse_nat + kin_rmse_exo)
    avg_total_musc_kin_rmse = (avg_musc_kin_rmse_nat + avg_musc_kin_rmse_exo) / 2
    std_total_musc_kin_rmse = np.std(kin_musc_rmse_nat + kin_musc_rmse_exo)

    ##################
    # now RMSE for GRF
    rmse_grf_y_nat = np.sqrt(np.mean((sim_grf_y_nat - ref_grf_y_nat) ** 2))
    rmse_grf_x_nat = np.sqrt(np.mean((sim_grf_x_nat - ref_grf_x_nat) ** 2))
    rmse_grf_y_exo = np.sqrt(np.mean((sim_grf_y_exo - ref_grf_y_exo) ** 2))
    rmse_grf_x_exo = np.sqrt(np.mean((sim_grf_x_exo - ref_grf_x_exo) ** 2))
    avg_grf_rmse_nat = (rmse_grf_y_nat + rmse_grf_x_nat) / 2
    std_grf_rmse_nat = np.std([rmse_grf_y_nat, rmse_grf_x_nat])
    avg_grf_rmse_exo = (rmse_grf_y_exo + rmse_grf_x_exo) / 2
    std_grf_rmse_exo = np.std([rmse_grf_y_exo, rmse_grf_x_exo])
    avg_total_grf_rmse = (avg_grf_rmse_nat + avg_grf_rmse_exo) / 2
    std_total_grf_rmse = np.std([rmse_grf_y_nat, rmse_grf_x_nat, rmse_grf_y_exo, rmse_grf_x_exo])
    
    ###################
    # now RMSE for moments. 
    rmse_hip_moment_nat = np.sqrt(np.mean((sim_hip_moment_nat - ref_hip_moment_nat) ** 2))
    rmse_hip_moment_exo = np.sqrt(np.mean((sim_hip_moment_exo - ref_hip_moment_exo) ** 2))
    rmse_knee_moment_nat = np.sqrt(np.mean((sim_knee_moment_nat - ref_knee_moment_nat) ** 2))
    rmse_knee_moment_exo = np.sqrt(np.mean((sim_knee_moment_exo - ref_knee_moment_exo) ** 2))
    rmse_ankle_moment_nat = np.sqrt(np.mean((sim_ankle_moment_nat - ref_ankle_moment_nat) ** 2))
    rmse_ankle_moment_exo = np.sqrt(np.mean((sim_ankle_moment_exo - ref_ankle_moment_exo) ** 2))
    rmse_hip_moment_nat_l = np.sqrt(np.mean((sim_hip_moment_nat_l - ref_hip_moment_nat_l) ** 2))
    rmse_hip_moment_exo_l = np.sqrt(np.mean((sim_hip_moment_exo_l - ref_hip_moment_exo_l) ** 2))
    rmse_knee_moment_nat_l = np.sqrt(np.mean((sim_knee_moment_nat_l - ref_knee_moment_nat_l) ** 2))
    rmse_knee_moment_exo_l = np.sqrt(np.mean((sim_knee_moment_exo_l - ref_knee_moment_exo_l) ** 2))
    rmse_ankle_moment_nat_l = np.sqrt(np.mean((sim_ankle_moment_nat_l - ref_ankle_moment_nat_l) ** 2))
    rmse_ankle_moment_exo_l = np.sqrt(np.mean((sim_ankle_moment_exo_l - ref_ankle_moment_exo_l) ** 2))
    rmse_pelvis_tilt_moment_nat = np.sqrt(np.mean((sim_pelvis_tilt_moment_nat - ref_pelvis_tilt_moment_nat) ** 2))
    rmse_pelvis_tilt_moment_exo = np.sqrt(np.mean((sim_pelvis_tilt_moment_exo - ref_pelvis_tilt_moment_exo) ** 2))
    rmse_pelvis_list_moment_nat = np.sqrt(np.mean((sim_pelvis_list_moment_nat - ref_pelvis_list_moment_nat) ** 2))
    rmse_pelvis_list_moment_exo = np.sqrt(np.mean((sim_pelvis_list_moment_exo - ref_pelvis_list_moment_exo) ** 2))
    rmse_pelvis_rotation_moment_nat = np.sqrt(np.mean((sim_pelvis_rotation_moment_nat - ref_pelvis_rotation_moment_nat) ** 2))
    rmse_pelvis_rotation_moment_exo = np.sqrt(np.mean((sim_pelvis_rotation_moment_exo - ref_pelvis_rotation_moment_exo) ** 2))
    rmse_lumbar_extension_moment_nat = np.sqrt(np.mean((sim_lumbar_extension_moment_nat - ref_lumbar_extension_moment_nat) ** 2))
    rmse_lumbar_extension_moment_exo = np.sqrt(np.mean((sim_lumbar_extension_moment_exo - ref_lumbar_extension_moment_exo) ** 2))
    rmse_lumbar_bending_moment_nat = np.sqrt(np.mean((sim_lumbar_bending_moment_nat - ref_lumbar_bending_moment_nat) ** 2))
    rmse_lumbar_bending_moment_exo = np.sqrt(np.mean((sim_lumbar_bending_moment_exo - ref_lumbar_bending_moment_exo) ** 2))
    rmse_lumbar_rotation_moment_nat = np.sqrt(np.mean((sim_lumbar_rotation_moment_nat - ref_lumbar_rotation_moment_nat) ** 2))
    rmse_lumbar_rotation_moment_exo = np.sqrt(np.mean((sim_lumbar_rotation_moment_exo - ref_lumbar_rotation_moment_exo) ** 2))
    rmse_arm_flex_moment_nat = np.sqrt(np.mean((sim_arm_flex_moment_nat - ref_arm_flex_moment_nat) ** 2))
    rmse_arm_flex_moment_exo = np.sqrt(np.mean((sim_arm_flex_moment_exo - ref_arm_flex_moment_exo) ** 2))
    rmse_arm_flex_moment_nat_l = np.sqrt(np.mean((sim_arm_flex_moment_nat_l - ref_arm_flex_moment_nat_l) ** 2))
    rmse_arm_flex_moment_exo_l = np.sqrt(np.mean((sim_arm_flex_moment_exo_l - ref_arm_flex_moment_exo_l) ** 2))
    rmse_elbow_flex_moment_nat = np.sqrt(np.mean((sim_elbow_flex_moment_nat - ref_elbow_flex_moment_nat) ** 2))
    rmse_elbow_flex_moment_exo = np.sqrt(np.mean((sim_elbow_flex_moment_exo - ref_elbow_flex_moment_exo) ** 2))
    rmse_elbow_flex_moment_nat_l = np.sqrt(np.mean((sim_elbow_flex_moment_nat_l - ref_elbow_flex_moment_nat_l) ** 2))
    rmse_elbow_flex_moment_exo_l = np.sqrt(np.mean((sim_elbow_flex_moment_exo_l - ref_elbow_flex_moment_exo_l) ** 2))
    moment_rmse_nat = [rmse_hip_moment_nat, rmse_knee_moment_nat, rmse_ankle_moment_nat,
                       rmse_hip_moment_nat_l, rmse_knee_moment_nat_l, rmse_ankle_moment_nat_l,
                    #    rmse_pelvis_tilt_moment_nat, rmse_pelvis_list_moment_nat, rmse_pelvis_rotation_moment_nat,
                       rmse_lumbar_extension_moment_nat, rmse_lumbar_bending_moment_nat, rmse_lumbar_rotation_moment_nat,
                       rmse_arm_flex_moment_nat, rmse_arm_flex_moment_nat_l, rmse_elbow_flex_moment_nat, rmse_elbow_flex_moment_nat_l]
    avg_moment_rmse_nat = sum(moment_rmse_nat) / len(moment_rmse_nat)
    std_moment_rmse_nat = np.std(moment_rmse_nat)
    avg_musc_moment_rmse_nat = sum(moment_rmse_nat[:6]) / 6
    std_musc_moment_rmse_nat = np.std(moment_rmse_nat[:6])

    moment_rmse_exo = [rmse_hip_moment_exo, rmse_knee_moment_exo, rmse_ankle_moment_exo,
                       rmse_hip_moment_exo_l, rmse_knee_moment_exo_l, rmse_ankle_moment_exo_l,
                    #    rmse_pelvis_tilt_moment_exo, rmse_pelvis_list_moment_exo, rmse_pelvis_rotation_moment_exo,
                       rmse_lumbar_extension_moment_exo, rmse_lumbar_bending_moment_exo, rmse_lumbar_rotation_moment_exo,
                       rmse_arm_flex_moment_exo, rmse_arm_flex_moment_exo_l, rmse_elbow_flex_moment_exo, rmse_elbow_flex_moment_exo_l]
    avg_moment_rmse_exo = sum(moment_rmse_exo) / len(moment_rmse_exo)
    std_moment_rmse_exo = np.std(moment_rmse_exo)
    avg_musc_moment_rmse_exo = sum(moment_rmse_exo[:6]) / 6
    std_musc_moment_rmse_exo = np.std(moment_rmse_exo[:6])
    
    average_total_moment_rmse = (avg_moment_rmse_nat + avg_moment_rmse_exo) / 2
    std_total_moment_rmse = np.std(moment_rmse_nat + moment_rmse_exo)
    average_total_musc_moment_rmse = (avg_musc_moment_rmse_nat + avg_musc_moment_rmse_exo) / 2
    std_total_musc_moment_rmse = np.std(moment_rmse_nat[:6] + moment_rmse_exo[:6])

    ###################
    # print out the rmse values
    print('\n' + '='*60)
    print('RMSE Values - Natural running 2.7 m/s')
    print('='*60)
    print(f'Average Kinematic RMSE: {avg_kin_rmse_nat:.2f} ± {std_kin_rmse_nat:.2f}')
    print(f'Average Muscular Kinematic RMSE: {avg_musc_kin_rmse_nat:.2f} ± {std_musc_kin_rmse_nat:.2f}')
    print(f'Average GRF RMSE: {avg_grf_rmse_nat:.2f} ± {std_grf_rmse_nat:.2f}')
    print(f'Average Moment RMSE: {avg_moment_rmse_nat:.2f} ± {std_moment_rmse_nat:.2f}')
    print(f'Average Muscular Moment RMSE: {avg_musc_moment_rmse_nat:.2f} ± {std_musc_moment_rmse_nat:.2f}')
    print('\n' + '='*60)
    print(f'Hip Angle RMSE: {rmse_hip_angle_nat:.2f} deg')
    print(f'Knee Angle RMSE: {rmse_knee_angle_nat:.2f} deg')
    print(f'Ankle Angle RMSE: {rmse_ankle_angle_nat:.2f} deg')
    print(f'Pelvis Vertical RMSE: {rmse_ty_nat:.4f} m/height')
    print(f'Pelvis Tilt RMSE: {rmse_pelvis_tilt_nat:.2f} deg')
    print(f'Pelvis List RMSE: {rmse_pelvis_list_nat:.2f} deg')
    print(f'Pelvis Rotation RMSE: {rmse_pelvis_rotation_nat:.2f} deg')
    print(f'Lumbar Extension RMSE: {rmse_lumbar_extension_nat:.2f} deg')
    print(f'Lumbar Bending RMSE: {rmse_lumbar_bending_nat:.2f} deg')
    print(f'Lumbar Rotation RMSE: {rmse_lumbar_rotation_nat:.2f} deg')
    print(f'Arm Flexion RMSE: {rmse_arm_flex_nat:.2f} deg')
    print(f'Arm Flexion L RMSE: {rmse_arm_flex_nat_l:.2f} deg')
    print(f'Elbow Flexion RMSE: {rmse_elbow_flex_nat:.2f} deg')
    print(f'Elbow Flexion L RMSE: {rmse_elbow_flex_nat_l:.2f} deg')
    print('\n' + '='*60)
    print(f'GRF Y RMSE: {rmse_grf_y_nat:.2f} N')
    print(f'GRF X RMSE: {rmse_grf_x_nat:.2f} N')
    print('\n' + '='*60)
    print(f'Hip Moment RMSE: {rmse_hip_moment_nat:.2f} Nm/kg')
    print(f'Knee Moment RMSE: {rmse_knee_moment_nat:.2f} Nm/kg')
    print(f'Ankle Moment RMSE: {rmse_ankle_moment_nat:.2f} Nm/kg')
    # print(f'Pelvis Tilt Moment RMSE: {rmse_pelvis_tilt_moment_nat:.2f} Nm/kg')
    # print(f'Pelvis List Moment RMSE: {rmse_pelvis_list_moment_nat:.2f} Nm/kg')
    # print(f'Pelvis Rotation Moment RMSE: {rmse_pelvis_rotation_moment_nat:.2f} Nm/kg')
    print(f'Lumbar Extension Moment RMSE: {rmse_lumbar_extension_moment_nat:.2f} Nm/kg')
    print(f'Lumbar Bending Moment RMSE: {rmse_lumbar_bending_moment_nat:.2f} Nm/kg')
    print(f'Lumbar Rotation Moment RMSE: {rmse_lumbar_rotation_moment_nat:.2f} Nm/kg')
    print(f'Arm Flexion Moment RMSE: {rmse_arm_flex_moment_nat:.2f} Nm/kg')
    print(f'Arm Flexion L Moment RMSE: {rmse_arm_flex_moment_nat_l:.2f} Nm/kg')
    print(f'Elbow Flexion Moment RMSE: {rmse_elbow_flex_moment_nat:.2f} Nm/kg')
    print(f'Elbow Flexion L Moment RMSE: {rmse_elbow_flex_moment_nat_l:.2f} Nm/kg')
    

    print('\n' + '='*60)
    print('RMSE Values - Exotendon running 2.7 m/s')
    print('='*60)
    print(f'Average Kinematic RMSE: {avg_kin_rmse_exo:.2f} ± {std_kin_rmse_exo:.2f}')
    print(f'Average Muscular Kinematic RMSE: {avg_musc_kin_rmse_exo:.2f} ± {std_musc_kin_rmse_exo:.2f}')
    print(f'Average GRF RMSE: {avg_grf_rmse_exo:.2f} ± {std_grf_rmse_exo:.2f}')
    print(f'Average Moment RMSE: {avg_moment_rmse_exo:.2f} ± {std_moment_rmse_exo:.2f}')
    print(f'Average Muscular Moment RMSE: {avg_musc_moment_rmse_exo:.2f} ± {std_musc_moment_rmse_exo:.2f}')
    print('\n' + '='*60)
    print(f'Hip Angle RMSE: {rmse_hip_angle_exo:.2f} deg')
    print(f'Knee Angle RMSE: {rmse_knee_angle_exo:.2f} deg')
    print(f'Ankle Angle RMSE: {rmse_ankle_angle_exo:.2f} deg')
    print(f'Pelvis Vertical RMSE: {rmse_ty_exo:.4f} m/height')
    print(f'Pelvis Tilt RMSE: {rmse_pelvis_tilt_exo:.2f} deg')
    print(f'Pelvis List RMSE: {rmse_pelvis_list_exo:.2f} deg')
    print(f'Pelvis Rotation RMSE: {rmse_pelvis_rotation_exo:.2f} deg')
    print(f'Lumbar Extension RMSE: {rmse_lumbar_extension_exo:.2f} deg')
    print(f'Lumbar Bending RMSE: {rmse_lumbar_bending_exo:.2f} deg')
    print(f'Lumbar Rotation RMSE: {rmse_lumbar_rotation_exo:.2f} deg')
    print(f'Arm Flexion RMSE: {rmse_arm_flex_exo:.2f} deg')
    print(f'Arm Flexion L RMSE: {rmse_arm_flex_exo_l:.2f} deg')
    print(f'Elbow Flexion RMSE: {rmse_elbow_flex_exo:.2f} deg')
    print(f'Elbow Flexion L RMSE: {rmse_elbow_flex_exo_l:.2f} deg')
    print('\n' + '='*60)
    print(f'GRF Y RMSE: {rmse_grf_y_exo:.2f} N')
    print(f'GRF X RMSE: {rmse_grf_x_exo:.2f} N')
    print('\n' + '='*60)
    print(f'Hip Moment RMSE: {rmse_hip_moment_exo:.2f} Nm/kg')
    print(f'Knee Moment RMSE: {rmse_knee_moment_exo:.2f} Nm/kg')
    print(f'Ankle Moment RMSE: {rmse_ankle_moment_exo:.2f} Nm/kg')
    # print(f'Pelvis Tilt Moment RMSE: {rmse_pelvis_tilt_moment_exo:.2f} Nm/kg')
    # print(f'Pelvis List Moment RMSE: {rmse_pelvis_list_moment_exo:.2f} Nm/kg')
    # print(f'Pelvis Rotation Moment RMSE: {rmse_pelvis_rotation_moment_exo:.2f} Nm/kg')
    print(f'Lumbar Extension Moment RMSE: {rmse_lumbar_extension_moment_exo:.2f} Nm/kg')
    print(f'Lumbar Bending Moment RMSE: {rmse_lumbar_bending_moment_exo:.2f} Nm/kg')
    print(f'Lumbar Rotation Moment RMSE: {rmse_lumbar_rotation_moment_exo:.2f} Nm/kg')
    print(f'Arm Flexion Moment RMSE: {rmse_arm_flex_moment_exo:.2f} Nm/kg')
    print(f'Arm Flexion L Moment RMSE: {rmse_arm_flex_moment_exo_l:.2f} Nm/kg')
    print(f'Elbow Flexion Moment RMSE: {rmse_elbow_flex_moment_exo:.2f} Nm/kg')
    print(f'Elbow Flexion L Moment RMSE: {rmse_elbow_flex_moment_exo_l:.2f} Nm/kg')
    
    print('\n' + '='*60)
    print('RMSE Values - Combined running 2.7 m/s')
    print('='*60)
    print(f'Average Kinematic RMSE: {avg_total_kin_rmse:.2f} ± {std_total_kin_rmse:.2f}')
    print(f'Average Muscular Kinematic RMSE: {avg_total_musc_kin_rmse:.2f} ± {std_total_musc_kin_rmse:.2f}')
    print(f'Average GRF RMSE: {avg_total_grf_rmse:.2f} ± {std_total_grf_rmse:.2f}')
    print(f'Average Moment RMSE: {average_total_moment_rmse:.2f} ± {std_total_moment_rmse:.2f}')
    print(f'Average Muscular Moment RMSE: {average_total_musc_moment_rmse:.2f} ± {std_total_musc_moment_rmse:.2f}')
    print('\n' + '='*60)


    #######################
    # compute pearson r for the data
    pearson_vars_nat = [
        ('Hip Angle', sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat),
        ('Knee Angle', sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat),
        ('Ankle Angle', sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat),
        ('Hip Angle L', sim_hip_angle_nat_l * 180 / np.pi, ref_hip_angle_nat_l),
        ('Knee Angle L', sim_knee_angle_nat_l * 180 / np.pi, ref_knee_angle_nat_l),
        ('Ankle Angle L', sim_ankle_angle_nat_l * 180 / np.pi, ref_ankle_angle_nat_l),
        ('Pelvis Vertical', sim_ty_nat, ref_ty_nat),
        ('Pelvis Tilt', sim_pelvis_tilt_nat * 180 / np.pi, ref_pelvis_tilt_nat),
        ('Pelvis List', sim_pelvis_list_nat * 180 / np.pi, ref_pelvis_list_nat),
        ('Pelvis Rotation', sim_pelvis_rotation_nat * 180 / np.pi, ref_pelvis_rotation_nat),
        ('Lumbar Extension', sim_lumbar_extension_nat * 180 / np.pi, ref_lumbar_extension_nat),
        ('Lumbar Bending', sim_lumbar_bending_nat * 180 / np.pi, ref_lumbar_bending_nat),
        ('Lumbar Rotation', sim_lumbar_rotation_nat * 180 / np.pi, ref_lumbar_rotation_nat),
        ('Arm Flexion', sim_arm_flex_nat * 180 / np.pi, ref_arm_flex_nat),
        ('Arm Flexion L', sim_arm_flex_nat_l * 180 / np.pi, ref_arm_flex_nat_l),
        ('Elbow Flexion', sim_elbow_flex_nat * 180 / np.pi, ref_elbow_flex_nat),
        ('Elbow Flexion L', sim_elbow_flex_nat_l * 180 / np.pi, ref_elbow_flex_nat_l),
        ('GRF Y', sim_grf_y_nat, ref_grf_y_nat),
        ('GRF X', sim_grf_x_nat, ref_grf_x_nat),
        ('Hip Moment', sim_hip_moment_nat, ref_hip_moment_nat),
        ('Knee Moment', sim_knee_moment_nat, ref_knee_moment_nat),
        ('Ankle Moment', sim_ankle_moment_nat, ref_ankle_moment_nat),
        # ('Pelvis Tilt Moment', sim_pelvis_tilt_moment_nat, ref_pelvis_tilt_moment_nat),
        # ('Pelvis List Moment', sim_pelvis_list_moment_nat, ref_pelvis_list_moment_nat),
        # ('Pelvis Rotation Moment', sim_pelvis_rotation_moment_nat, ref_pelvis_rotation_moment_nat),
        ('Lumbar Extension Moment', sim_lumbar_extension_moment_nat, ref_lumbar_extension_moment_nat),
        ('Lumbar Bending Moment', sim_lumbar_bending_moment_nat, ref_lumbar_bending_moment_nat),
        ('Lumbar Rotation Moment', sim_lumbar_rotation_moment_nat, ref_lumbar_rotation_moment_nat),
        ('Arm Flexion Moment', sim_arm_flex_moment_nat, ref_arm_flex_moment_nat),
        ('Arm Flexion L Moment', sim_arm_flex_moment_nat_l, ref_arm_flex_moment_nat_l),
        ('Elbow Flexion Moment', sim_elbow_flex_moment_nat, ref_elbow_flex_moment_nat),
        ('Elbow Flexion L Moment', sim_elbow_flex_moment_nat_l, ref_elbow_flex_moment_nat_l)
    ]
    pearson_vars_exo = [
        ('Hip Angle', sim_hip_angle_exo * 180 / np.pi, ref_hip_angle_exo),
        ('Knee Angle', sim_knee_angle_exo * 180 / np.pi, ref_knee_angle_exo),
        ('Ankle Angle', sim_ankle_angle_exo * 180 / np.pi, ref_ankle_angle_exo),
        ('Hip Angle L', sim_hip_angle_exo_l * 180 / np.pi, ref_hip_angle_exo_l),
        ('Knee Angle L', sim_knee_angle_exo_l * 180 / np.pi, ref_knee_angle_exo_l),
        ('Ankle Angle L', sim_ankle_angle_exo_l * 180 / np.pi, ref_ankle_angle_exo_l),
        ('Pelvis Vertical', sim_ty_exo, ref_ty_exo),
        ('Pelvis Tilt', sim_pelvis_tilt_exo * 180 / np.pi, ref_pelvis_tilt_exo),
        ('Pelvis List', sim_pelvis_list_exo * 180 / np.pi, ref_pelvis_list_exo),
        ('Pelvis Rotation', sim_pelvis_rotation_exo * 180 / np.pi, ref_pelvis_rotation_exo),
        ('Lumbar Extension', sim_lumbar_extension_exo * 180 / np.pi, ref_lumbar_extension_exo),
        ('Lumbar Bending', sim_lumbar_bending_exo * 180 / np.pi, ref_lumbar_bending_exo),
        ('Lumbar Rotation', sim_lumbar_rotation_exo * 180 / np.pi, ref_lumbar_rotation_exo),
        ('Arm Flexion', sim_arm_flex_exo * 180 / np.pi, ref_arm_flex_exo),
        ('Arm Flexion L', sim_arm_flex_exo_l * 180 / np.pi, ref_arm_flex_exo_l),
        ('Elbow Flexion', sim_elbow_flex_exo * 180 / np.pi, ref_elbow_flex_exo),
        ('Elbow Flexion L', sim_elbow_flex_exo_l * 180 / np.pi, ref_elbow_flex_exo_l),
        ('GRF Y', sim_grf_y_exo, ref_grf_y_exo),
        ('GRF X', sim_grf_x_exo, ref_grf_x_exo),
        ('Hip Moment', sim_hip_moment_exo, ref_hip_moment_exo),
        ('Knee Moment', sim_knee_moment_exo, ref_knee_moment_exo),
        ('Ankle Moment', sim_ankle_moment_exo, ref_ankle_moment_exo),
        ('Hip Moment L', sim_hip_moment_exo_l, ref_hip_moment_exo_l),
        ('Knee Moment L', sim_knee_moment_exo_l, ref_knee_moment_exo_l),
        ('Ankle Moment L', sim_ankle_moment_exo_l, ref_ankle_moment_exo_l),
        # ('Pelvis Tilt Moment', sim_pelvis_tilt_moment_exo, ref_pelvis_tilt_moment_exo),
        # ('Pelvis List Moment', sim_pelvis_list_moment_exo, ref_pelvis_list_moment_exo),
        # ('Pelvis Rotation Moment', sim_pelvis_rotation_moment_exo, ref_pelvis_rotation_moment_exo),
        ('Lumbar Extension Moment', sim_lumbar_extension_moment_exo, ref_lumbar_extension_moment_exo),
        ('Lumbar Bending Moment', sim_lumbar_bending_moment_exo, ref_lumbar_bending_moment_exo),
        ('Lumbar Rotation Moment', sim_lumbar_rotation_moment_exo, ref_lumbar_rotation_moment_exo),
        ('Arm Flexion Moment', sim_arm_flex_moment_exo, ref_arm_flex_moment_exo),
        ('Arm Flexion L Moment', sim_arm_flex_moment_exo_l, ref_arm_flex_moment_exo_l),
        ('Elbow Flexion Moment', sim_elbow_flex_moment_exo, ref_elbow_flex_moment_exo),
        ('Elbow Flexion L Moment', sim_elbow_flex_moment_exo_l, ref_elbow_flex_moment_exo_l)
    ]

    pearson_metrics_nat = []
    for name, sim_data, ref_data in pearson_vars_nat:
        sim_vals = np.asarray(sim_data).ravel()
        ref_vals = np.asarray(ref_data).ravel()
        n = min(len(sim_vals), len(ref_vals))
        if n < 2: 
            pearson = np.nan
        else:
            pearson = np.corrcoef(sim_vals[:n], ref_vals[:n])[0, 1]
        pearson_metrics_nat.append({'Variable': name, 'Pearson r': pearson})
    pearson_metrics_exo = []
    for name, sim_data, ref_data in pearson_vars_exo:
        sim_vals = np.asarray(sim_data).ravel()
        ref_vals = np.asarray(ref_data).ravel()
        n = min(len(sim_vals), len(ref_vals))
        if n < 2:
            pearson = np.nan
        else:
            pearson = np.corrcoef(sim_vals[:n], ref_vals[:n])[0, 1]
        pearson_metrics_exo.append({'Variable': name, 'Pearson r': pearson})
    pearson_df_nat = pd.DataFrame(pearson_metrics_nat)
    pearson_df_exo = pd.DataFrame(pearson_metrics_exo)
    print('\n' + '='*60)
    print('Pearson correlation coefficients for natural simulation:')
    print(pearson_df_nat.to_string(index=False))
    print('\n' + '='*60)
    print('Pearson correlation coefficients for exoskeleton simulation:')
    print(pearson_df_exo.to_string(index=False))
    pearson_df_nat.to_csv('pearson_metrics_natural_27.csv', index=False)
    pearson_df_exo.to_csv('pearson_metrics_exoskeleton_27.csv', index=False)
    #################
    # separate into diff groups. 
    kin_labels = {
        'Hip Angle', 'Knee Angle', 'Ankle Angle',
        'Hip Angle L', 'Knee Angle L', 'Ankle Angle L',
        'Pelvis Vertical', 'Pelvis Tilt', 'Pelvis List', 'Pelvis Rotation',
        'Lumbar Extension', 'Lumbar Bending', 'Lumbar Rotation',
        'Arm Flexion', 'Arm Flexion L', 'Elbow Flexion', 'Elbow Flexion L'
    }
    kin_labels_musc = {
        'Hip Angle', 'Knee Angle', 'Ankle Angle',
        'Hip Angle L', 'Knee Angle L', 'Ankle Angle L',
    }
    moment_labels_musc = {
        'Hip Moment', 'Knee Moment', 'Ankle Moment',
        'Hip Moment L', 'Knee Moment L', 'Ankle Moment L'
    }
    grf_labels = {'GRF Y', 'GRF X'}
    moment_labels = {
        'Hip Moment', 'Knee Moment', 'Ankle Moment',
        'Hip Moment L', 'Knee Moment L', 'Ankle Moment L',
        # 'Pelvis Tilt Moment', 'Pelvis List Moment', 'Pelvis Rotation Moment',
        'Lumbar Extension Moment', 'Lumbar Bending Moment', 'Lumbar Rotation Moment',
        'Arm Flexion Moment', 'Arm Flexion L Moment', 'Elbow Flexion Moment', 'Elbow Flexion L Moment'
    }
    # now compute
    kin_rs_nat = [
        item['Pearson r'] for item in pearson_metrics_nat if item['Variable'] in kin_labels
    ]
    moment_rs_nat = [
        item['Pearson r'] for item in pearson_metrics_nat if item['Variable'] in moment_labels
    ]
    grf_rs_nat = [
        item['Pearson r'] for item in pearson_metrics_nat if item['Variable'] in grf_labels
    ]
    kin_rs_exo = [
        item['Pearson r'] for item in pearson_metrics_exo if item['Variable'] in kin_labels
    ]
    moment_rs_exo = [
        item['Pearson r'] for item in pearson_metrics_exo if item['Variable'] in moment_labels
    ]
    grf_rs_exo = [
        item['Pearson r'] for item in pearson_metrics_exo if item['Variable'] in grf_labels
    ]
    kin_rs_musc_nat = [
        item['Pearson r'] for item in pearson_metrics_nat if item['Variable'] in kin_labels_musc
    ]
    moment_rs_musc_nat = [
        item['Pearson r'] for item in pearson_metrics_nat if item['Variable'] in moment_labels_musc
    ]

    kin_rs_musc_exo = [
        item['Pearson r'] for item in pearson_metrics_exo if item['Variable'] in kin_labels_musc
    ]
    moment_rs_musc_exo = [
        item['Pearson r'] for item in pearson_metrics_exo if item['Variable'] in moment_labels_musc
    ]

    kin_rs_musc_combined = kin_rs_musc_nat + kin_rs_musc_exo
    moment_rs_musc_combined = moment_rs_musc_nat + moment_rs_musc_exo

    kin_rs_combined = kin_rs_nat + kin_rs_exo
    moment_rs_combined = moment_rs_nat + moment_rs_exo
    grf_rs_combined = grf_rs_nat + grf_rs_exo
    
    
    # compute standard dev for each
    kin_rs_std_nat = np.std(kin_rs_nat)
    moment_rs_std_nat = np.std(moment_rs_nat)
    grf_rs_std_nat = np.std(grf_rs_nat)
    kin_rs_std_exo = np.std(kin_rs_exo)
    moment_rs_std_exo = np.std(moment_rs_exo)
    grf_rs_std_exo = np.std(grf_rs_exo)
    kin_rs_combined_std = np.std(kin_rs_combined)
    moment_rs_combined_std = np.std(moment_rs_combined)
    grf_rs_combined_std = np.std(grf_rs_combined)
    # muscle spanning
    kin_rs_musc_std_nat = np.std(kin_rs_musc_nat)
    moment_rs_musc_std_nat = np.std(moment_rs_musc_nat)
    kin_rs_musc_std_exo = np.std(kin_rs_musc_exo)
    moment_rs_musc_std_exo = np.std(moment_rs_musc_exo)
    kin_rs_musc_std_combined = np.std(kin_rs_musc_combined)
    moment_rs_musc_std_combined = np.std(moment_rs_musc_combined)

    print('\n' + '='*60)
    print('Average Pearson r for kinematics - Natural:', np.nanmean(kin_rs_nat))
    print('Standard Deviation of Pearson r for kinematics - Natural:', kin_rs_std_nat)
    print('Average Pearson r for kinematics - Exoskeleton:', np.nanmean(kin_rs_exo))
    print('Standard Deviation of Pearson r for kinematics - Exoskeleton:', kin_rs_std_exo)
    print('Average Pearson r for kinematics - Combined:', np.nanmean(kin_rs_combined))
    print('Standard Deviation of Pearson r for kinematics - Combined:', kin_rs_combined_std)
    print('\n' + '='*60)
    print('Average Pearson r for muscle spanning kinematics - Natural:', np.nanmean(kin_rs_musc_nat))
    print('Standard Deviation of Pearson r for muscle spanning kinematics - Natural:', kin_rs_musc_std_nat)
    print('Average Pearson r for muscle spanning kinematics - Exoskeleton:', np.nanmean(kin_rs_musc_exo))
    print('Standard Deviation of Pearson r for muscle spanning kinematics - Exoskeleton:', kin_rs_musc_std_exo)
    print('Average Pearson r for muscle spanning kinematics - Combined:', np.nanmean(kin_rs_musc_combined))
    print('Standard Deviation of Pearson r for muscle spanning kinematics - Combined:', kin_rs_musc_std_combined)
    print('\n' + '='*60)
    print('Average Pearson r for moments - Natural:', np.nanmean(moment_rs_nat))
    print('Standard Deviation of Pearson r for moments - Natural:', moment_rs_std_nat)
    print('Average Pearson r for moments - Exoskeleton:', np.nanmean(moment_rs_exo))
    print('Standard Deviation of Pearson r for moments - Exoskeleton:', moment_rs_std_exo)
    print('Average Pearson r for moments - Combined:', np.nanmean(moment_rs_combined))
    print('Standard Deviation of Pearson r for moments - Combined:', moment_rs_combined_std)
    print('\n' + '='*60)
    print('Average Pearson r for muscle spanning moments - Natural:', np.nanmean(moment_rs_musc_nat))
    print('Standard Deviation of Pearson r for muscle spanning moments - Natural:', moment_rs_musc_std_nat)
    print('Average Pearson r for muscle spanning moments - Exoskeleton:', np.nanmean(moment_rs_musc_exo))
    print('Standard Deviation of Pearson r for muscle spanning moments - Exoskeleton:', moment_rs_musc_std_exo)
    print('Average Pearson r for muscle spanning moments - Combined:', np.nanmean(moment_rs_musc_combined))
    print('Standard Deviation of Pearson r for muscle spanning moments - Combined:', moment_rs_musc_std_combined)
    print('\n' + '='*60)
    print('Average Pearson r for GRFs - Natural:', np.nanmean(grf_rs_nat))
    print('Standard Deviation of Pearson r for GRFs - Natural:', grf_rs_std_nat)
    print('Average Pearson r for GRFs - Exoskeleton:', np.nanmean(grf_rs_exo))
    print('Standard Deviation of Pearson r for GRFs - Exoskeleton:', grf_rs_std_exo)
    print('Average Pearson r for GRFs - Combined:', np.nanmean(grf_rs_combined))
    print('Standard Deviation of Pearson r for GRFs - Combined:', grf_rs_combined_std)

    ##########################
    print('\n' + '='*60)
    print('All RMSE and Pearson r values have been computed and printed above.')
    print('You can use the printed values to create your own figures or tables as needed.')
    ##########################
    ##########################################################################
    # first is the natural validation figure
    ##########################################################################
    ## now create the figure that we want. It should be a 3x3 grid of subplots. 
    # the first column should be hip knee and ankle angles
    # second column should be hip knee and ankle moments
    # third column should be the pelvis ty, GRF y, and GRF x
    # fig, ax = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
    # x = np.linspace(0, 100, kin_simlen_nat)
    
    # # Hip angle
    # # ax[0, 0].plot(x, ref_hip_angle_nat, label='Nat. Ref', color='orange')
    # # ax[0, 0].plot(x, ref_hip_angle_exo, label='Exo Ref', color='purple')
    # ax[0, 0].fill_between(x, ref_hip_angle_nat - 2*std_nat['hip_flexion_r'], ref_hip_angle_nat + 2*std_nat['hip_flexion_r'], color='orange', alpha=0.2)
    # # ax[0, 0].fill_between(x, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    # ax[0, 0].plot(x, sim_hip_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[0, 0].plot(x, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    # ax[0, 0].set_title('Hip Angle', fontsize=14)
    # ax[0, 0].tick_params(axis='y', labelsize=14)    
    # ax[0, 0].tick_params(axis='x', labelsize=14)
    # ax[0, 0].set_ylabel('Angle (deg)', fontsize=14)
    # ax[0, 0].legend(fontsize=14, loc='lower right')

    # # Knee angle
    # # ax[1, 0].plot(x, ref_knee_angle_nat, label='Nat. Ref', color='orange')
    # # ax[1, 0].plot(x, ref_knee_angle_exo, label='Exo Ref', color='purple')
    # ax[1, 0].fill_between(x, ref_knee_angle_nat - 2*std_nat['knee_angle_r'], ref_knee_angle_nat + 2*std_nat['knee_angle_r'], color='orange', alpha=0.2)
    # # ax[1, 0].fill_between(x, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    # ax[1, 0].plot(x, sim_knee_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[1, 0].plot(x, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    # ax[1, 0].set_title('Knee Angle', fontsize=14)
    # ax[1, 0].set_ylabel('Angle (deg)', fontsize=14)
    # ax[1, 0].tick_params(axis='y', labelsize=14)    
    # ax[1, 0].tick_params(axis='x', labelsize=14)
    # ax[1, 0].legend(fontsize=14, loc='upper left')

    # # Ankle angle
    # # ax[2, 0].plot(x, ref_ankle_angle_nat, label='Nat. Ref', color='orange')
    # # ax[2, 0].plot(x, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    # ax[2, 0].fill_between(x, ref_ankle_angle_nat - 2*std_nat['ankle_angle_r'], ref_ankle_angle_nat + 2*std_nat['ankle_angle_r'], color='orange', alpha=0.2)
    # # ax[2, 0].fill_between(x, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    # ax[2, 0].plot(x, sim_ankle_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[2, 0].plot(x, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    # ax[2, 0].set_title('Ankle Angle', fontsize=14)
    # ax[2, 0].set_ylabel('Angle (deg)', fontsize=14)
    # ax[2, 0].set_xlabel('Gait Cycle %', fontsize=14)
    # ax[2, 0].tick_params(axis='y', labelsize=14)    
    # ax[2, 0].tick_params(axis='x', labelsize=14)
    # ax[2, 0].legend(fontsize=14, loc='upper right')

    # # Hip moment
    # # ax[0, 1].plot(x, ref_hip_moment_nat, label='Nat. Ref', color='orange')
    # # ax[0, 1].plot(x, ref_hip_moment_exo, label='Exo Ref', color='purple')
    # ax[0, 1].fill_between(x, ref_hip_moment_nat - 2*stdmomnat['hip_flexion_r_moment'], ref_hip_moment_nat + 2*stdmomnat['hip_flexion_r_moment'], color='orange', alpha=0.2)
    # # ax[0, 1].fill_between(x, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2)
    # ax[0, 1].plot(x, sim_hip_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[0, 1].plot(x, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[0, 1].set_title('Hip Moment', fontsize=14)
    # ax[0, 1].set_ylabel('Moment (Nm)', fontsize=14)
    # ax[0, 1].tick_params(axis='y', labelsize=14)    
    # ax[0, 1].tick_params(axis='x', labelsize=14)
    # ax[0, 1].legend(fontsize=14, loc='upper right')

    # # Knee moment
    # # ax[1, 1].plot(x, ref_knee_moment_nat, label='Nat. Ref', color='orange')
    # # ax[1, 1].plot(x, ref_knee_moment_exo, label='Exo Ref', color='purple')
    # ax[1, 1].fill_between(x, ref_knee_moment_nat - 2*stdmomnat['knee_angle_r_moment'], ref_knee_moment_nat + 2*stdmomnat['knee_angle_r_moment'], color='orange', alpha=0.2)
    # # ax[1, 1].fill_between(x, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2)
    # ax[1, 1].plot(x, sim_knee_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[1, 1].plot(x, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[1, 1].set_title('Knee Moment', fontsize=14)
    # ax[1, 1].set_ylabel('Moment (Nm)', fontsize=14)
    # ax[1, 1].tick_params(axis='y', labelsize=14)    
    # ax[1, 1].tick_params(axis='x', labelsize=14)
    # ax[1, 1].legend(fontsize=14, loc='lower right')

    # # Ankle moment
    # # ax[2, 1].plot(x, ref_ankle_moment_nat, label='Nat. Ref', color='orange')
    # # ax[2, 1].plot(x, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    # ax[2, 1].fill_between(x, ref_ankle_moment_nat - 2*stdmomnat['ankle_angle_r_moment'], ref_ankle_moment_nat + 2*stdmomnat['ankle_angle_r_moment'], color='orange', alpha=0.2)
    # # ax[2, 1].fill_between(x, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2)
    # ax[2, 1].plot(x, sim_ankle_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[2, 1].plot(x, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[2, 1].set_title('Ankle Moment', fontsize=14)
    # ax[2, 1].set_ylabel('Moment (Nm)', fontsize=14)
    # ax[2, 1].set_xlabel('Gait Cycle %', fontsize=14)
    # ax[2, 1].tick_params(axis='y', labelsize=14)    
    # ax[2, 1].tick_params(axis='x', labelsize=14)
    # ax[2, 1].legend(fontsize=14, loc='lower right')

    # # Pelvis ty
    # # ax[0, 2].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # # ax[0, 2].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    # ax[0, 2].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2)
    # # ax[0, 2].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2)
    # ax[0, 2].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[0, 2].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[0, 2].set_title('Pelvis Vertical Translation', fontsize=14)
    # ax[0, 2].set_ylabel('Translation (m/height)', fontsize=14)
    # ax[0, 2].tick_params(axis='y', labelsize=14)    
    # ax[0, 2].tick_params(axis='x', labelsize=14)
    # ax[0, 2].legend(fontsize=14, loc='upper right')

    # # GRF y
    # # Fill between for stdgrfnat and stdgrfexo for y GRF
    # ax[1, 2].fill_between(x, ref_grf_y_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fy'], ref_grf_y_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fy'], color='orange', alpha=0.2)
    # # ax[1, 2].fill_between(x, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2)
    # # ax[1, 2].plot(x, ref_grf_y_nat, label='Nat. Ref', color='orange')
    # # ax[1, 2].plot(x, ref_grf_y_exo, label='Exo Ref', color='purple')
    # ax[1, 2].plot(x, sim_grf_y_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[1, 2].plot(x, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[1, 2].set_title('Vertical GRF', fontsize=14)
    # ax[1, 2].set_ylabel('Force (BW)', fontsize=14)
    # ax[1, 2].tick_params(axis='y', labelsize=14)    
    # ax[1, 2].tick_params(axis='x', labelsize=14)
    # ax[1, 2].legend(fontsize=14, loc='upper right')

    # # GRF x
    # ax[2, 2].fill_between(x, ref_grf_x_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fx'], ref_grf_x_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fx'], color='orange', alpha=0.2)
    # # ax[2, 2].fill_between(x, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2)
    # # ax[2, 2].plot(x, ref_grf_x_nat, label='Nat. Ref', color='orange')
    # # ax[2, 2].plot(x, ref_grf_x_exo, label='Exo Ref', color='purple')
    # ax[2, 2].plot(x, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[2, 2].plot(x, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[2, 2].set_title('Horizontal GRF', fontsize=14)
    # ax[2, 2].set_ylabel('Force (BW)', fontsize=14)
    # ax[2, 2].set_xlabel('Gait Cycle %', fontsize=14)
    # ax[2, 2].tick_params(axis='y', labelsize=14)    
    # ax[2, 2].tick_params(axis='x', labelsize=14)
    # ax[2, 2].legend(fontsize=14, loc='upper right')

    # plt.tight_layout()
    # figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    # plt.savefig(figpath + 'figure_saggitalvalidation_27.png')
    # plt.show()


    # possibly inputing other plots here to illustrate the exotendon results....
    fig, ax = plt.subplots(5, 4, figsize=(13, 15), dpi=500)
    x = np.linspace(0, 100, kin_simlen_nat)
    
    # Hip angle
    # ax[1,0].plot(x, ref_hip_angle_nat, label='Nat. Ref', color='orange')
    # ax[1,0].plot(x, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[1,0].fill_between(x, ref_hip_angle_nat - 2*std_nat['hip_flexion_r'], ref_hip_angle_nat + 2*std_nat['hip_flexion_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[1,0].fill_between(x, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[1,0].plot(x, sim_hip_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1,0].plot(x, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[1,0].set_title('Hip Angle', fontsize=14)
    ax[1,0].tick_params(axis='y', labelsize=14)    
    ax[1,0].tick_params(axis='x', labelsize=14)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=14)
    # ax[1,0].legend(fontsize=14, loc='lower right')

    # Knee angle
    # ax[2,0].plot(x, ref_knee_angle_nat, label='Nat. Ref', color='orange')
    # ax[2,0].plot(x, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[2,0].fill_between(x, ref_knee_angle_nat - 2*std_nat['knee_angle_r'], ref_knee_angle_nat + 2*std_nat['knee_angle_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[2,0].fill_between(x, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[2,0].plot(x, sim_knee_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2,0].plot(x, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[2,0].set_title('Knee Angle', fontsize=14)
    ax[2,0].set_ylabel('Angle (deg)', fontsize=14)
    ax[2,0].tick_params(axis='y', labelsize=14)    
    ax[2,0].tick_params(axis='x', labelsize=14)
    # ax[2,0].legend(fontsize=14, loc='upper left')

    # Ankle angle
    # ax[3,0].plot(x, ref_ankle_angle_nat, label='Nat. Ref', color='orange')
    # ax[3,0].plot(x, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[3,0].fill_between(x, ref_ankle_angle_nat - 2*std_nat['ankle_angle_r'], ref_ankle_angle_nat + 2*std_nat['ankle_angle_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[3,0].fill_between(x, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[3,0].plot(x, sim_ankle_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[3,0].plot(x, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[3,0].set_title('Ankle Angle', fontsize=14)
    ax[3,0].set_ylabel('Angle (deg)', fontsize=14)
    ax[3,0].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,0].tick_params(axis='y', labelsize=14)    
    ax[3,0].tick_params(axis='x', labelsize=14)
    # ax[3,0].legend(fontsize=14, loc='upper right')

    # Hip moment
    # ax[1,1].plot(x, ref_hip_moment_nat, label='Nat. Ref', color='orange')
    # ax[1,1].plot(x, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[1,1].fill_between(x, ref_hip_moment_nat - 2*stdmomnat['hip_flexion_r_moment'], ref_hip_moment_nat + 2*stdmomnat['hip_flexion_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[1,1].fill_between(x, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2)
    ax[1,1].plot(x, sim_hip_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1,1].plot(x, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1,1].set_title('Hip Moment', fontsize=14)
    ax[1,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[1,1].tick_params(axis='y', labelsize=14)    
    ax[1,1].tick_params(axis='x', labelsize=14)
    # ax[1,1].legend(fontsize=14, loc='upper right')

    # Knee moment
    # ax[2,1].plot(x, ref_knee_moment_nat, label='Nat. Ref', color='orange')
    # ax[2,1].plot(x, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[2,1].fill_between(x, ref_knee_moment_nat - 2*stdmomnat['knee_angle_r_moment'], ref_knee_moment_nat + 2*stdmomnat['knee_angle_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[2,1].fill_between(x, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2)
    ax[2,1].plot(x, sim_knee_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2,1].plot(x, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2,1].set_title('Knee Moment', fontsize=14)
    ax[2,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[2,1].tick_params(axis='y', labelsize=14)    
    ax[2,1].tick_params(axis='x', labelsize=14)
    # ax[2,1].legend(fontsize=14, loc='lower right')

    # Ankle moment
    # ax[3,1].plot(x, ref_ankle_moment_nat, label='Nat. Ref', color='orange')
    # ax[3,1].plot(x, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[3,1].fill_between(x, ref_ankle_moment_nat - 2*stdmomnat['ankle_angle_r_moment'], ref_ankle_moment_nat + 2*stdmomnat['ankle_angle_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[3,1].fill_between(x, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2)
    ax[3,1].plot(x, sim_ankle_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[3,1].plot(x, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[3,1].set_title('Ankle Moment', fontsize=14)
    ax[3,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[3,1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,1].tick_params(axis='y', labelsize=14)    
    ax[3,1].tick_params(axis='x', labelsize=14)
    # ax[3,1].legend(fontsize=14, loc='lower right')

    # Pelvis ty
    # ax[4,0].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # ax[4,0].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    ax[4,0].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[4,0].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2)
    ax[4,0].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[4,0].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[4,0].set_title('Pelvis Vertical Translation', fontsize=14)
    ax[4,0].set_ylabel('Translation (m/height)', fontsize=14)
    ax[4,0].tick_params(axis='y', labelsize=14)    
    ax[4,0].tick_params(axis='x', labelsize=14)
    # ax[4,0].legend(fontsize=14, loc='upper right')

    # GRF y
    # Fill between for stdgrfnat and stdgrfexo for y GRF
    ax[0,0].fill_between(x, ref_grf_y_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fy'], ref_grf_y_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fy'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[0,0].fill_between(x, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2)
    # ax[0,0].plot(x, ref_grf_y_nat, label='Nat. Ref', color='orange')
    # ax[0,0].plot(x, ref_grf_y_exo, label='Exo Ref', color='purple')
    ax[0,0].plot(x, sim_grf_y_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0,0].plot(x, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('Vertical GRF', fontsize=14)
    ax[0,0].set_ylabel('Force (BW)', fontsize=14)
    ax[0,0].tick_params(axis='y', labelsize=14)    
    ax[0,0].tick_params(axis='x', labelsize=14)
    # ax[0,0].legend(fontsize=14, loc='upper right')

    # GRF x
    ax[0,1].fill_between(x, ref_grf_x_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fx'], ref_grf_x_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fx'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[0,1].fill_between(x, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2)
    # ax[0,1].plot(x, ref_grf_x_nat, label='Nat. Ref', color='orange')
    # ax[0,1].plot(x, ref_grf_x_exo, label='Exo Ref', color='purple')
    ax[0,1].plot(x, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0,1].plot(x, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('Horizontal GRF', fontsize=14)
    ax[0,1].set_ylabel('Force (BW)', fontsize=14)
    ax[0,1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[0,1].tick_params(axis='y', labelsize=14)    
    ax[0,1].tick_params(axis='x', labelsize=14)
    # ax[0,1].legend(fontsize=14, loc='upper right')
    ###########################################################
    ##
    ###########################################################
    # Hip angle
    # ax[1,2].plot(x, ref_hip_angle_nat, label='Nat. Ref', color='orange')
    # ax[1,2].plot(x, ref_hip_angle_exo, label='Exo Ref', color='purple')
    # ax[1,2].fill_between(x, ref_hip_angle_nat - 2*std_nat['hip_flexion_r'], ref_hip_angle_nat + 2*std_nat['hip_flexion_r'], color='orange', alpha=0.2)
    ax[1,2].fill_between(x, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[1,2].plot(x, sim_hip_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    ax[1,2].plot(x, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[1,2].set_title('Hip Angle', fontsize=14)
    ax[1,2].tick_params(axis='y', labelsize=14)    
    ax[1,2].tick_params(axis='x', labelsize=14)
    ax[1,2].set_ylabel('Angle (deg)', fontsize=14)
    # ax[1,2].legend(fontsize=14, loc='lower right')

    # Knee angle
    # ax[2,2].plot(x, ref_knee_angle_nat, label='Nat. Ref', color='orange')
    # ax[2,2].plot(x, ref_knee_angle_exo, label='Exo Ref', color='purple')
    # ax[2,2].fill_between(x, ref_knee_angle_nat - 2*std_nat['knee_angle_r'], ref_knee_angle_nat + 2*std_nat['knee_angle_r'], color='orange', alpha=0.2)
    ax[2,2].fill_between(x, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[2,2].plot(x, sim_knee_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    ax[2,2].plot(x, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[2,2].set_title('Knee Angle', fontsize=14)
    ax[2,2].set_ylabel('Angle (deg)', fontsize=14)
    ax[2,2].tick_params(axis='y', labelsize=14)    
    ax[2,2].tick_params(axis='x', labelsize=14)
    # ax[2,2].legend(fontsize=14, loc='upper left')

    # Ankle angle
    # ax[3,2].plot(x, ref_ankle_angle_nat, label='Nat. Ref', color='orange')
    # ax[3,2].plot(x, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    # ax[3,2].fill_between(x, ref_ankle_angle_nat - 2*std_nat['ankle_angle_r'], ref_ankle_angle_nat + 2*std_nat['ankle_angle_r'], color='orange', alpha=0.2)
    ax[3,2].fill_between(x, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[3,2].plot(x, sim_ankle_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    ax[3,2].plot(x, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[3,2].set_title('Ankle Angle', fontsize=14)
    ax[3,2].set_ylabel('Angle (deg)', fontsize=14)
    ax[3,2].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,2].tick_params(axis='y', labelsize=14)    
    ax[3,2].tick_params(axis='x', labelsize=14)
    # ax[3,2].legend(fontsize=14, loc='upper right')

    # Hip moment
    # ax[1,3].plot(x, ref_hip_moment_nat, label='Nat. Ref', color='orange')
    # ax[1,3].plot(x, ref_hip_moment_exo, label='Exo Ref', color='purple')
    # ax[1,3].fill_between(x, ref_hip_moment_nat - 2*stdmomnat['hip_flexion_r_moment'], ref_hip_moment_nat + 2*stdmomnat['hip_flexion_r_moment'], color='orange', alpha=0.2)
    ax[1,3].fill_between(x, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[1,3].plot(x, sim_hip_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[1,3].plot(x, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1,3].set_title('Hip Moment', fontsize=14)
    ax[1,3].set_ylabel('Moment (Nm)', fontsize=14)
    ax[1,3].tick_params(axis='y', labelsize=14)    
    ax[1,3].tick_params(axis='x', labelsize=14)
    # ax[1,3].legend(fontsize=14, loc='upper right')

    # Knee moment
    # ax[2,3].plot(x, ref_knee_moment_nat, label='Nat. Ref', color='orange')
    # ax[2,3].plot(x, ref_knee_moment_exo, label='Exo Ref', color='purple')
    # ax[2,3].fill_between(x, ref_knee_moment_nat - 2*stdmomnat['knee_angle_r_moment'], ref_knee_moment_nat + 2*stdmomnat['knee_angle_r_moment'], color='orange', alpha=0.2)
    ax[2,3].fill_between(x, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[2,3].plot(x, sim_knee_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[2,3].plot(x, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2,3].set_title('Knee Moment', fontsize=14)
    ax[2,3].set_ylabel('Moment (Nm)', fontsize=14)
    ax[2,3].tick_params(axis='y', labelsize=14)    
    ax[2,3].tick_params(axis='x', labelsize=14)
    # ax[2,3].legend(fontsize=14, loc='lower right')

    # Ankle moment
    # ax[3,3].plot(x, ref_ankle_moment_nat, label='Nat. Ref', color='orange')
    # ax[3,3].plot(x, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    # ax[3,3].fill_between(x, ref_ankle_moment_nat - 2*stdmomnat['ankle_angle_r_moment'], ref_ankle_moment_nat + 2*stdmomnat['ankle_angle_r_moment'], color='orange', alpha=0.2)
    ax[3,3].fill_between(x, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[3,3].plot(x, sim_ankle_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[3,3].plot(x, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[3,3].set_title('Ankle Moment', fontsize=14)
    ax[3,3].set_ylabel('Moment (Nm)', fontsize=14)
    ax[3,3].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,3].tick_params(axis='y', labelsize=14)    
    ax[3,3].tick_params(axis='x', labelsize=14)
    # ax[3,3].legend(fontsize=14, loc='lower right')

    # Pelvis ty
    # ax[4,2].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # ax[4,2].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    # ax[4,2].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2)
    ax[4,2].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[4,2].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[4,2].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[4,2].set_title('Pelvis Vertical Translation', fontsize=14)
    ax[4,2].set_ylabel('Translation (m/height)', fontsize=14)
    ax[4,2].tick_params(axis='y', labelsize=14)    
    ax[4,2].tick_params(axis='x', labelsize=14)
    # ax[4,2].legend(fontsize=14, loc='upper right')

    # GRF y
    # Fill between for stdgrfnat and stdgrfexo for y GRF
    # ax[0,2].fill_between(x, ref_grf_y_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fy'], ref_grf_y_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fy'], color='orange', alpha=0.2)
    ax[0,2].fill_between(x, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[0,2].plot(x, ref_grf_y_nat, label='Nat. Ref', color='orange')
    # ax[0,2].plot(x, ref_grf_y_exo, label='Exo Ref', color='purple')
    # ax[0,2].plot(x, sim_grf_y_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[0,2].plot(x, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,2].set_title('Vertical GRF', fontsize=14)
    ax[0,2].set_ylabel('Force (BW)', fontsize=14)
    ax[0,2].tick_params(axis='y', labelsize=14)    
    ax[0,2].tick_params(axis='x', labelsize=14)
    # ax[0,2].legend(fontsize=14, loc='upper right')

    # GRF x
    # ax[0,3].fill_between(x, ref_grf_x_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fx'], ref_grf_x_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fx'], color='orange', alpha=0.2)
    ax[0,3].fill_between(x, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2, label='Exo Ref ±2SD')
    # ax[0,3].plot(x, ref_grf_x_nat, label='Nat. Ref', color='orange')
    # ax[0,3].plot(x, ref_grf_x_exo, label='Exo Ref', color='purple')
    # ax[0,3].plot(x, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    ax[0,3].plot(x, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,3].set_title('Horizontal GRF', fontsize=14)
    ax[0,3].set_ylabel('Force (BW)', fontsize=14)
    ax[0,3].set_xlabel('Gait Cycle %', fontsize=14)
    ax[0,3].tick_params(axis='y', labelsize=14)    
    ax[0,3].tick_params(axis='x', labelsize=14)
    # ax[0,3].legend(fontsize=14, loc='upper right')
    

    # Turn off unused subplots (axes) in the 6x4 grid
    ax[4, 1].axis('off')
    ax[4, 3].axis('off')

    # Hide the last subplot and use it to display the legend   

    # get the legend labels from the previous subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[4, 1].legend(handles, labels, loc='center left', fontsize=14)
    handles_e, labels_e = ax[0, 2].get_legend_handles_labels()
    ax[4, 3].legend(handles_e, labels_e, loc='center left', fontsize=14)

    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)

    fig.text(0.01, 0.99, 'a)', fontsize=18, fontweight='bold', ha='left', va='top')
    fig.text(0.52, 0.99, 'b)', fontsize=18, fontweight='bold', ha='center', va='top')
# Add panel labels at the top of the figure

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_27.png')
    plt.show()
    return

# create a function that does full gait cycle and all the natural stuff in one for validating 4.0
def saggitalValidationFigure40(simNat, iknat2D, labels2D, coordinates_sim_clean, mean_nat, std_nat, GRFsimnat, GRFrefnat, meangrfnat, stdgrfnat, natmomentfile, idnat, meanmomnat, stdmomnat, modelfile):
    # load the model and get the mass
    model = osim.Model(modelfile)
    mass = model.getTotalMass(model.initSystem())
    height = 1.78
    ## starting with the kinematics
    # get the length of the simulation data
    kin_simlen_nat = len(simNat.getIndependentColumn())
    # kin_simlen_exo = len(simExo.getIndependentColumn())
    # if kin_simlen_nat != kin_simlen_exo:
    #     print('Simulation kinematic data lengths do not match. Exiting.')
    #     return
    # get the sim kinematics
    sim_hip_angle_nat = simNat.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_hip_angle_nat_l = simNat.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_nat = simNat.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_knee_angle_nat_l = simNat.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_nat = simNat.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_ankle_angle_nat_l = simNat.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ty_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    sim_pelvistilt_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvislist_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvisrotation_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbarextension_nat = simNat.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()
    sim_lumbarbending_nat = simNat.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbarrotation_nat = simNat.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_armflex_r_nat = simNat.getDependentColumn('/jointset/acromial_r/arm_flex_r/value').to_numpy()
    sim_armflex_l_nat = simNat.getDependentColumn('/jointset/acromial_l/arm_flex_l/value').to_numpy()
    sim_elbowflex_r_nat = simNat.getDependentColumn('/jointset/elbow_r/elbow_flex_r/value').to_numpy()
    sim_elbowflex_l_nat = simNat.getDependentColumn('/jointset/elbow_l/elbow_flex_l/value').to_numpy()


    # get the reference data
    ref_hip_angle_nat = iknat2D.getDependentColumn('hip_flexion_r').to_numpy()
    ref_hip_angle_nat_l = iknat2D.getDependentColumn('hip_flexion_l').to_numpy()
    ref_knee_angle_nat = iknat2D.getDependentColumn('knee_angle_r').to_numpy()
    ref_knee_angle_nat_l = iknat2D.getDependentColumn('knee_angle_l').to_numpy()
    ref_ankle_angle_nat = iknat2D.getDependentColumn('ankle_angle_r').to_numpy()
    ref_ankle_angle_nat_l = iknat2D.getDependentColumn('ankle_angle_l').to_numpy()
    ref_ty_nat = iknat2D.getDependentColumn('pelvis_ty').to_numpy()
    ref_pelvistilt_nat = iknat2D.getDependentColumn('pelvis_tilt').to_numpy()
    ref_pelvislist_nat = iknat2D.getDependentColumn('pelvis_list').to_numpy()
    ref_pelvisrotation_nat = iknat2D.getDependentColumn('pelvis_rotation').to_numpy()
    ref_lumbarextension_nat = iknat2D.getDependentColumn('lumbar_extension').to_numpy()
    ref_lumbarbending_nat = iknat2D.getDependentColumn('lumbar_bending').to_numpy()
    ref_lumbarrotation_nat = iknat2D.getDependentColumn('lumbar_rotation').to_numpy()
    ref_armflex_r_nat = iknat2D.getDependentColumn('arm_flex_r').to_numpy()
    ref_armflex_l_nat = iknat2D.getDependentColumn('arm_flex_l').to_numpy()
    ref_elbowflex_r_nat = iknat2D.getDependentColumn('elbow_flex_r').to_numpy()
    ref_elbowflex_l_nat = iknat2D.getDependentColumn('elbow_flex_l').to_numpy()


    # get the length of the reference data
    kin_reflen_nat = len(ref_hip_angle_nat)
    # kin_reflen_exo = len(ref_hip_angle_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation. 
    kin_xsim = np.linspace(0,100,kin_simlen_nat)
    kin_xref_nat = np.linspace(0,100,kin_reflen_nat)
    # kin_xref_exo = np.linspace(0,100,kin_reflen_exo)


    # interpolate the reference data to the simulation data length
    ref_hip_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_nat)
    ref_hip_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_nat_l)
    ref_knee_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_nat)
    ref_knee_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_nat_l)
    ref_ankle_angle_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_nat)
    ref_ankle_angle_nat_l = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_nat_l)
    ref_ty_nat = np.interp(kin_xsim, kin_xref_nat, ref_ty_nat)
    ref_pelvistilt_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvistilt_nat)
    ref_pelvislist_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvislist_nat)
    ref_pelvisrotation_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvisrotation_nat)
    ref_lumbarextension_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarextension_nat)
    ref_lumbarbending_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarbending_nat)
    ref_lumbarrotation_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarrotation_nat)
    ref_armflex_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_armflex_r_nat)
    ref_armflex_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_armflex_l_nat)
    ref_elbowflex_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_elbowflex_r_nat)
    ref_elbowflex_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_elbowflex_l_nat)


    # normalize the ty values to the height of the subject
    ref_ty_nat = ref_ty_nat / height
    # ref_ty_exo = ref_ty_exo / height
    sim_ty_nat = sim_ty_nat / height
    # sim_ty_exo = sim_ty_exo / height

    ## get the GRF data
    grfsimnat = osim.TimeSeriesTable(GRFsimnat)
    # grfsimexo = osim.TimeSeriesTable(GRFsimexo)
    grfrefnat = osim.TimeSeriesTable(GRFrefnat)
    # grfrefexo = osim.TimeSeriesTable(GRFrefexo)
    # get the length of the simulation data
    grf_simlen_nat = len(grfsimnat.getIndependentColumn())
    # grf_simlen_exo = len(grfsimexo.getIndependentColumn())
    # if grf_simlen_nat != grf_simlen_exo:
    #     print('Simulation GRF data lengths do not match. Exiting.')
    #     return
    # get the sim GRF data
    sim_grf_y_nat = grfsimnat.getDependentColumn('ground_force_r_vy').to_numpy()
    # sim_grf_y_exo = grfsimexo.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_x_nat = grfsimnat.getDependentColumn('ground_force_r_vx').to_numpy()
    # sim_grf_x_exo = grfsimexo.getDependentColumn('ground_force_r_vx').to_numpy()
    # get the reference GRF data
    ref_grf_y_nat = grfrefnat.getDependentColumn('R_ground_force_vy').to_numpy()
    # ref_grf_y_exo = grfrefexo.getDependentColumn('rF_y').to_numpy()
    ref_grf_x_nat = grfrefnat.getDependentColumn('R_ground_force_vx').to_numpy()
    # ref_grf_x_exo = grfrefexo.getDependentColumn('rF_x').to_numpy()
    # ref_grf_y_nat = meangrfnat['calcn_r_Right_GRF_Fy']
    # ref_grf_y_exo = meangrfexo['calcn_r_Right_GRF_Fy']
    # ref_grf_x_nat = meangrfnat['calcn_r_Right_GRF_Fx']
    # ref_grf_x_exo = meangrfexo['calcn_r_Right_GRF_Fx']
    # get the length of the reference data
    grf_reflen_nat = len(ref_grf_y_nat)
    # grf_reflen_exo = len(ref_grf_y_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    grf_xsim = np.linspace(0,100,grf_simlen_nat)
    grf_xref_nat = np.linspace(0,100,grf_reflen_nat)
    # grf_xref_exo = np.linspace(0,100,grf_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_grf_y_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_y_nat)
    # ref_grf_y_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_y_exo)
    ref_grf_x_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_x_nat)
    # ref_grf_x_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_x_exo)
    # divide all of the GRF data based on the mass of the model
    sim_grf_y_nat = sim_grf_y_nat/(mass*9.81)
    # sim_grf_y_exo = sim_grf_y_exo/(mass*9.81)
    sim_grf_x_nat = sim_grf_x_nat/(mass*9.81)
    # sim_grf_x_exo = sim_grf_x_exo/(mass*9.81)
    ref_grf_y_nat = ref_grf_y_nat/(mass*9.81)
    # ref_grf_y_exo = ref_grf_y_exo/(mass*9.81)
    ref_grf_x_nat = ref_grf_x_nat/(mass*9.81)
    # ref_grf_x_exo = ref_grf_x_exo/(mass*9.81)
    
    ## get the moment data
    natmoment = osim.TimeSeriesTable(natmomentfile)
    # exomoment = osim.TimeSeriesTable(exomomentfile)
    natrefmoment = osim.TimeSeriesTable(idnat)
    # exorefmoment = osim.TimeSeriesTable(idexo)
    # get the length of the simulation data
    moment_simlen_nat = len(natmoment.getIndependentColumn())
    # moment_simlen_exo = len(exomoment.getIndependentColumn())
    # if moment_simlen_nat != moment_simlen_exo:
    #     print('Simulation moment data lengths do not match. Exiting.')
    #     return
    
    # get the sim moment data
    sim_hip_moment_nat = natmoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_hip_moment_nat_l = natmoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_knee_moment_nat = natmoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_knee_moment_nat_l = natmoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_ankle_moment_nat = natmoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_ankle_moment_nat_l = natmoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_pelvistilt_moment_nat = natmoment.getDependentColumn('pelvis_tilt_moment').to_numpy()
    sim_pelvislist_moment_nat = natmoment.getDependentColumn('pelvis_list_moment').to_numpy()
    sim_pelvisrotation_moment_nat = natmoment.getDependentColumn('pelvis_rotation_moment').to_numpy()
    sim_lumbarextension_moment_nat = natmoment.getDependentColumn('lumbar_extension_moment').to_numpy()
    sim_lumbarbending_moment_nat = natmoment.getDependentColumn('lumbar_bending_moment').to_numpy()
    sim_lumbarrotation_moment_nat = natmoment.getDependentColumn('lumbar_rotation_moment').to_numpy()
    sim_armflex_r_moment_nat = natmoment.getDependentColumn('arm_flex_r_moment').to_numpy()
    sim_armflex_l_moment_nat = natmoment.getDependentColumn('arm_flex_l_moment').to_numpy()
    sim_elbowflex_r_moment_nat = natmoment.getDependentColumn('elbow_flex_r_moment').to_numpy()
    sim_elbowflex_l_moment_nat = natmoment.getDependentColumn('elbow_flex_l_moment').to_numpy()


    # get the reference moment data
    ref_hip_moment_nat = idnat.getDependentColumn('hip_flexion_r_moment').to_numpy()
    ref_hip_moment_nat_l = idnat.getDependentColumn('hip_flexion_l_moment').to_numpy()
    ref_knee_moment_nat = idnat.getDependentColumn('knee_angle_r_moment').to_numpy()
    ref_knee_moment_nat_l = idnat.getDependentColumn('knee_angle_l_moment').to_numpy()
    ref_ankle_moment_nat = idnat.getDependentColumn('ankle_angle_r_moment').to_numpy()
    ref_ankle_moment_nat_l = idnat.getDependentColumn('ankle_angle_l_moment').to_numpy()
    ref_pelvistilt_moment_nat = idnat.getDependentColumn('pelvis_tilt_moment').to_numpy()
    ref_pelvislist_moment_nat = idnat.getDependentColumn('pelvis_list_moment').to_numpy()
    ref_pelvisrotation_moment_nat = idnat.getDependentColumn('pelvis_rotation_moment').to_numpy()
    ref_lumbarextension_moment_nat = idnat.getDependentColumn('lumbar_extension_moment').to_numpy()
    ref_lumbarbending_moment_nat = idnat.getDependentColumn('lumbar_bending_moment').to_numpy()
    ref_lumbarrotation_moment_nat = idnat.getDependentColumn('lumbar_rotation_moment').to_numpy()
    ref_armflex_r_moment_nat = idnat.getDependentColumn('arm_flex_r_moment').to_numpy()
    ref_armflex_l_moment_nat = idnat.getDependentColumn('arm_flex_l_moment').to_numpy()
    ref_elbowflex_r_moment_nat = idnat.getDependentColumn('elbow_flex_r_moment').to_numpy()
    ref_elbowflex_l_moment_nat = idnat.getDependentColumn('elbow_flex_l_moment').to_numpy()


    # get the length of the reference data
    moment_reflen_nat = len(ref_hip_moment_nat)
    # moment_reflen_exo = len(ref_hip_moment_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    moment_xsim = np.linspace(0,100,moment_simlen_nat)
    moment_xref_nat = np.linspace(0,100,moment_reflen_nat)
    # moment_xref_exo = np.linspace(0,100,moment_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_hip_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_nat)
    ref_hip_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_nat_l)
    ref_knee_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_nat)
    ref_knee_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_nat_l)
    ref_ankle_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_nat)
    ref_ankle_moment_nat_l = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_nat_l)
    ref_pelvistilt_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvistilt_moment_nat)
    ref_pelvislist_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvislist_moment_nat)
    ref_pelvisrotation_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_pelvisrotation_moment_nat)
    ref_lumbarextension_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbarextension_moment_nat)
    ref_lumbarbending_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbarbending_moment_nat)
    ref_lumbarrotation_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_lumbarrotation_moment_nat)
    ref_armflex_r_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_armflex_r_moment_nat)
    ref_armflex_l_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_armflex_l_moment_nat)
    ref_elbowflex_r_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_elbowflex_r_moment_nat)
    ref_elbowflex_l_moment_nat = np.interp(moment_xsim, moment_xref_nat, ref_elbowflex_l_moment_nat)
    
    # normalize all of it to body mass
    sim_hip_moment_nat = sim_hip_moment_nat/mass    
    sim_hip_moment_nat_l = sim_hip_moment_nat_l/mass
    sim_knee_moment_nat = sim_knee_moment_nat/mass
    sim_knee_moment_nat_l = sim_knee_moment_nat_l/mass
    sim_ankle_moment_nat = sim_ankle_moment_nat/mass
    sim_ankle_moment_nat_l = sim_ankle_moment_nat_l/mass
    sim_pelvistilt_moment_nat = sim_pelvistilt_moment_nat/mass
    sim_pelvislist_moment_nat = sim_pelvislist_moment_nat/mass
    sim_pelvisrotation_moment_nat = sim_pelvisrotation_moment_nat/mass
    sim_lumbarextension_moment_nat = sim_lumbarextension_moment_nat/mass
    sim_lumbarbending_moment_nat = sim_lumbarbending_moment_nat/mass
    sim_lumbarrotation_moment_nat = sim_lumbarrotation_moment_nat/mass
    sim_armflex_r_moment_nat = sim_armflex_r_moment_nat/mass
    sim_armflex_l_moment_nat = sim_armflex_l_moment_nat/mass
    sim_elbowflex_r_moment_nat = sim_elbowflex_r_moment_nat/mass
    sim_elbowflex_l_moment_nat = sim_elbowflex_l_moment_nat/mass

    ref_hip_moment_nat = ref_hip_moment_nat/mass
    ref_hip_moment_nat_l = ref_hip_moment_nat_l/mass
    ref_knee_moment_nat = -ref_knee_moment_nat/mass
    ref_knee_moment_nat_l = -ref_knee_moment_nat_l/mass
    ref_ankle_moment_nat = ref_ankle_moment_nat/mass
    ref_ankle_moment_nat_l = ref_ankle_moment_nat_l/mass
    ref_pelvistilt_moment_nat = ref_pelvistilt_moment_nat/mass
    ref_pelvislist_moment_nat = ref_pelvislist_moment_nat/mass
    ref_pelvisrotation_moment_nat = ref_pelvisrotation_moment_nat/mass
    ref_lumbarextension_moment_nat = ref_lumbarextension_moment_nat/mass
    ref_lumbarbending_moment_nat = ref_lumbarbending_moment_nat/mass
    ref_lumbarrotation_moment_nat = ref_lumbarrotation_moment_nat/mass
    ref_armflex_r_moment_nat = ref_armflex_r_moment_nat/mass
    ref_armflex_l_moment_nat = ref_armflex_l_moment_nat/mass
    ref_elbowflex_r_moment_nat = ref_elbowflex_r_moment_nat/mass
    ref_elbowflex_l_moment_nat = ref_elbowflex_l_moment_nat/mass


    ############################

    # Simple plot: kinematic measures (sim vs ref) across gait cycle.
    kin_plot_vars = [
        ('Hip Angle R', sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat, 'deg'),
        ('Hip Angle L', sim_hip_angle_nat_l * 180 / np.pi, ref_hip_angle_nat_l, 'deg'),
        ('Knee Angle R', sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat, 'deg'),
        ('Knee Angle L', sim_knee_angle_nat_l * 180 / np.pi, ref_knee_angle_nat_l, 'deg'),
        ('Ankle Angle R', sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat, 'deg'),
        ('Ankle Angle L', sim_ankle_angle_nat_l * 180 / np.pi, ref_ankle_angle_nat_l, 'deg'),
        ('Pelvis Vertical', sim_ty_nat, ref_ty_nat, 'm/height'),
        ('Pelvis Tilt', sim_pelvistilt_nat * 180 / np.pi, ref_pelvistilt_nat, 'deg'),
        ('Pelvis List', sim_pelvislist_nat * 180 / np.pi, ref_pelvislist_nat, 'deg'),
        ('Pelvis Rotation', sim_pelvisrotation_nat * 180 / np.pi, ref_pelvisrotation_nat, 'deg'),
        ('Lumbar Extension', sim_lumbarextension_nat * 180 / np.pi, ref_lumbarextension_nat, 'deg'),
        ('Lumbar Bending', sim_lumbarbending_nat * 180 / np.pi, ref_lumbarbending_nat, 'deg'),
        ('Lumbar Rotation', sim_lumbarrotation_nat * 180 / np.pi, ref_lumbarrotation_nat, 'deg'),
        ('Arm Flex R', sim_armflex_r_nat * 180 / np.pi, ref_armflex_r_nat, 'deg'),
        ('Arm Flex L', sim_armflex_l_nat * 180 / np.pi, ref_armflex_l_nat, 'deg'),
        ('Elbow Flex R', sim_elbowflex_r_nat * 180 / np.pi, ref_elbowflex_r_nat, 'deg'),
        ('Elbow Flex L', sim_elbowflex_l_nat * 180 / np.pi, ref_elbowflex_l_nat, 'deg'),
    ]

    kin_x = np.linspace(0, 100, len(sim_hip_angle_nat))
    fig_kin, ax_kin = plt.subplots(5, 5, figsize=(14, 12), dpi=300)
    ax_kin = ax_kin.flatten()
    for i, (label, sim_vals, ref_vals, units) in enumerate(kin_plot_vars):
        ax_kin[i].plot(kin_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_kin[i].plot(kin_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_kin[i].set_title(label, fontsize=10)
        ax_kin[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_kin[i].set_ylabel(units, fontsize=9)
        ax_kin[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_kin[i].legend(fontsize=8, loc='best')

    for j in range(len(kin_plot_vars), len(ax_kin)):
        ax_kin[j].axis('off')

    plt.tight_layout()
    plt.show()

    ###########################
    # Simple plot: moment measures (sim vs ref) across gait cycle.
    moment_plot_vars = [
        ('Hip Moment R', sim_hip_moment_nat, ref_hip_moment_nat),
        ('Hip Moment L', sim_hip_moment_nat_l, ref_hip_moment_nat_l),
        ('Knee Moment R', sim_knee_moment_nat, ref_knee_moment_nat),
        ('Knee Moment L', sim_knee_moment_nat_l, ref_knee_moment_nat_l),
        ('Ankle Moment R', sim_ankle_moment_nat, ref_ankle_moment_nat),
        ('Ankle Moment L', sim_ankle_moment_nat_l, ref_ankle_moment_nat_l),
        ('Pelvis Tilt Moment', sim_pelvistilt_moment_nat, ref_pelvistilt_moment_nat),
        ('Pelvis List Moment', sim_pelvislist_moment_nat, ref_pelvislist_moment_nat),
        ('Pelvis Rotation Moment', sim_pelvisrotation_moment_nat, ref_pelvisrotation_moment_nat),
        ('Lumbar Extension Moment', sim_lumbarextension_moment_nat, ref_lumbarextension_moment_nat),
        ('Lumbar Bending Moment', sim_lumbarbending_moment_nat, ref_lumbarbending_moment_nat),
        ('Lumbar Rotation Moment', sim_lumbarrotation_moment_nat, ref_lumbarrotation_moment_nat),
        ('Arm Flex R Moment', sim_armflex_r_moment_nat, ref_armflex_r_moment_nat),
        ('Arm Flex L Moment', sim_armflex_l_moment_nat, ref_armflex_l_moment_nat),
        ('Elbow Flex R Moment', sim_elbowflex_r_moment_nat, ref_elbowflex_r_moment_nat),
        ('Elbow Flex L Moment', sim_elbowflex_l_moment_nat, ref_elbowflex_l_moment_nat),
    ]

    moment_x = np.linspace(0, 100, len(sim_hip_moment_nat))
    fig_mom, ax_mom = plt.subplots(4, 4, figsize=(14, 12), dpi=300)
    ax_mom = ax_mom.flatten()
    for i, (label, sim_vals, ref_vals) in enumerate(moment_plot_vars):
        ax_mom[i].plot(moment_x, ref_vals, color='black', linewidth=1, label='Ref')
        ax_mom[i].plot(moment_x, sim_vals, color='orange', linestyle='--', linewidth=1, label='Sim')
        ax_mom[i].set_title(label, fontsize=10)
        ax_mom[i].set_xlabel('Gait Cycle %', fontsize=9)
        ax_mom[i].set_ylabel('Moment (Nm/kg)', fontsize=9)
        ax_mom[i].tick_params(axis='both', labelsize=8)
        if i == 0:
            ax_mom[i].legend(fontsize=8, loc='best')

    for j in range(len(moment_plot_vars), len(ax_mom)):
        ax_mom[j].axis('off')

    plt.tight_layout()
    plt.show()

    ############################
    # compute error metrics: RMSE and R^2 for each variable
    
    # Compute RMSE for kinematics
    rmse_hip_r_nat = np.sqrt(np.mean((sim_hip_angle_nat * 180 / np.pi - ref_hip_angle_nat)**2))
    rmse_knee_r_nat = np.sqrt(np.mean((sim_knee_angle_nat * 180 / np.pi - ref_knee_angle_nat)**2))
    rmse_ankle_r_nat = np.sqrt(np.mean((sim_ankle_angle_nat * 180 / np.pi - ref_ankle_angle_nat)**2))
    rmse_ty_nat = np.sqrt(np.mean((sim_ty_nat - ref_ty_nat)**2))
    rmse_hip_l_nat = np.sqrt(np.mean((sim_hip_angle_nat_l * 180 / np.pi - ref_hip_angle_nat_l)**2))
    rmse_knee_l_nat = np.sqrt(np.mean((sim_knee_angle_nat_l * 180 / np.pi - ref_knee_angle_nat_l)**2))
    rmse_ankle_l_nat = np.sqrt(np.mean((sim_ankle_angle_nat_l * 180 / np.pi - ref_ankle_angle_nat_l)**2))
    rmse_pelvistilt_nat = np.sqrt(np.mean((sim_pelvistilt_nat * 180 / np.pi - ref_pelvistilt_nat)**2))
    rmse_pelvislist_nat = np.sqrt(np.mean((sim_pelvislist_nat * 180 / np.pi - ref_pelvislist_nat)**2))
    rmse_pelvisrotation_nat = np.sqrt(np.mean((sim_pelvisrotation_nat * 180 / np.pi - ref_pelvisrotation_nat)**2))
    rmse_lumbarextension_nat = np.sqrt(np.mean((sim_lumbarextension_nat * 180 / np.pi - ref_lumbarextension_nat)**2))
    rmse_lumbarbending_nat = np.sqrt(np.mean((sim_lumbarbending_nat * 180 / np.pi - ref_lumbarbending_nat)**2))
    rmse_lumbarrotation_nat = np.sqrt(np.mean((sim_lumbarrotation_nat * 180 / np.pi - ref_lumbarrotation_nat)**2))
    rmse_armflex_r_nat = np.sqrt(np.mean((sim_armflex_r_nat * 180 / np.pi - ref_armflex_r_nat)**2))
    rmse_armflex_l_nat = np.sqrt(np.mean((sim_armflex_l_nat * 180 / np.pi - ref_armflex_l_nat)**2))
    rmse_elbowflex_r_nat = np.sqrt(np.mean((sim_elbowflex_r_nat * 180 / np.pi - ref_elbowflex_r_nat)**2))
    rmse_elbowflex_l_nat = np.sqrt(np.mean((sim_elbowflex_l_nat * 180 / np.pi - ref_elbowflex_l_nat)**2))
    
    
    # compute R^2 for kinematics
    corr_hip_r_nat = r2_score(ref_hip_angle_nat, sim_hip_angle_nat * 180 / np.pi)
    corr_knee_r_nat = r2_score(ref_knee_angle_nat, sim_knee_angle_nat * 180 / np.pi)
    corr_ankle_r_nat = r2_score(ref_ankle_angle_nat, sim_ankle_angle_nat * 180 / np.pi)
    corr_ty_nat = r2_score(ref_ty_nat, sim_ty_nat)
    corr_hip_l_nat = r2_score(ref_hip_angle_nat_l, sim_hip_angle_nat_l * 180 / np.pi)
    corr_knee_l_nat = r2_score(ref_knee_angle_nat_l, sim_knee_angle_nat_l * 180 / np.pi)
    corr_ankle_l_nat = r2_score(ref_ankle_angle_nat_l, sim_ankle_angle_nat_l * 180 / np.pi)
    corr_pelvistilt_nat = r2_score(ref_pelvistilt_nat, sim_pelvistilt_nat * 180 / np.pi)
    corr_pelvislist_nat = r2_score(ref_pelvislist_nat, sim_pelvislist_nat * 180 / np.pi)
    corr_pelvisrotation_nat = r2_score(ref_pelvisrotation_nat, sim_pelvisrotation_nat * 180 / np.pi)
    corr_lumbarextension_nat = r2_score(ref_lumbarextension_nat, sim_lumbarextension_nat * 180 / np.pi)
    corr_lumbarbending_nat = r2_score(ref_lumbarbending_nat, sim_lumbarbending_nat * 180 / np.pi)
    corr_lumbarrotation_nat = r2_score(ref_lumbarrotation_nat, sim_lumbarrotation_nat * 180 / np.pi)
    corr_armflex_r_nat = r2_score(ref_armflex_r_nat, sim_armflex_r_nat * 180 / np.pi)
    corr_armflex_l_nat = r2_score(ref_armflex_l_nat, sim_armflex_l_nat * 180 / np.pi)
    corr_elbowflex_r_nat = r2_score(ref_elbowflex_r_nat, sim_elbowflex_r_nat * 180 / np.pi)
    corr_elbowflex_l_nat = r2_score(ref_elbowflex_l_nat, sim_elbowflex_l_nat * 180 / np.pi)
    # compute the average RMSE across all kinematics 
    kin_rmse = [rmse_hip_r_nat, rmse_knee_r_nat, rmse_ankle_r_nat, 
                rmse_ty_nat, rmse_hip_l_nat, rmse_knee_l_nat, rmse_ankle_l_nat, 
                rmse_pelvistilt_nat, rmse_pelvislist_nat, rmse_pelvisrotation_nat, 
                rmse_lumbarextension_nat, rmse_lumbarbending_nat, rmse_lumbarrotation_nat, 
                rmse_armflex_r_nat, rmse_armflex_l_nat, rmse_elbowflex_r_nat, rmse_elbowflex_l_nat]
    avg_kin_rmse = sum(kin_rmse) / len(kin_rmse)
    std_kin_rmse = np.std(kin_rmse)
    # compute average RMSE for lower limb kinematics
    kin_lower_rmse = [rmse_hip_r_nat, rmse_knee_r_nat, rmse_ankle_r_nat, rmse_hip_l_nat, rmse_knee_l_nat, rmse_ankle_l_nat]
    avg_kin_lower_rmse = sum(kin_lower_rmse) / len(kin_lower_rmse)
    std_kin_lower_rmse = np.std(kin_lower_rmse)
    
    # Compute RMSE for GRF
    rmse_grf_y_nat = np.sqrt(np.mean((sim_grf_y_nat - ref_grf_y_nat)**2))
    rmse_grf_x_nat = np.sqrt(np.mean((sim_grf_x_nat - ref_grf_x_nat)**2))
    # compute R^2 for GRF
    corr_grf_y_nat = r2_score(ref_grf_y_nat, sim_grf_y_nat)
    corr_grf_x_nat = r2_score(ref_grf_x_nat, sim_grf_x_nat)
    # average GRF RMSE
    avg_grf_rmse = (rmse_grf_y_nat + rmse_grf_x_nat) / 2
    std_grf_rmse = np.std([rmse_grf_y_nat, rmse_grf_x_nat])
    
    # Compute RMSE for moments
    rmse_hip_moment_nat = np.sqrt(np.mean((sim_hip_moment_nat - ref_hip_moment_nat)**2))
    rmse_knee_moment_nat = np.sqrt(np.mean((sim_knee_moment_nat - ref_knee_moment_nat)**2))
    rmse_ankle_moment_nat = np.sqrt(np.mean((sim_ankle_moment_nat - ref_ankle_moment_nat)**2))
    rmse_hip_moment_nat_l = np.sqrt(np.mean((sim_hip_moment_nat_l - ref_hip_moment_nat_l)**2))
    rmse_knee_moment_nat_l = np.sqrt(np.mean((sim_knee_moment_nat_l - ref_knee_moment_nat_l)**2))
    rmse_ankle_moment_nat_l = np.sqrt(np.mean((sim_ankle_moment_nat_l - ref_ankle_moment_nat_l)**2))
    # rmse_pelvistilt_moment_nat = np.sqrt(np.mean((sim_pelvistilt_moment_nat - ref_pelvistilt_moment_nat)**2))
    # rmse_pelvislist_moment_nat = np.sqrt(np.mean((sim_pelvislist_moment_nat - ref_pelvislist_moment_nat)**2))
    # rmse_pelvisrotation_moment_nat = np.sqrt(np.mean((sim_pelvisrotation_moment_nat - ref_pelvisrotation_moment_nat)**2))
    rmse_lumbarextension_moment_nat = np.sqrt(np.mean((sim_lumbarextension_moment_nat - ref_lumbarextension_moment_nat)**2))
    rmse_lumbarbending_moment_nat = np.sqrt(np.mean((sim_lumbarbending_moment_nat - ref_lumbarbending_moment_nat)**2))
    rmse_lumbarrotation_moment_nat = np.sqrt(np.mean((sim_lumbarrotation_moment_nat - ref_lumbarrotation_moment_nat)**2))
    rmse_armflex_r_moment_nat = np.sqrt(np.mean((sim_armflex_r_moment_nat - ref_armflex_r_moment_nat)**2))
    rmse_armflex_l_moment_nat = np.sqrt(np.mean((sim_armflex_l_moment_nat - ref_armflex_l_moment_nat)**2))
    rmse_elbowflex_r_moment_nat = np.sqrt(np.mean((sim_elbowflex_r_moment_nat - ref_elbowflex_r_moment_nat)**2))
    rmse_elbowflex_l_moment_nat = np.sqrt(np.mean((sim_elbowflex_l_moment_nat - ref_elbowflex_l_moment_nat)**2))
    # average moment RMSE
    moment_rmse = [rmse_hip_moment_nat, rmse_knee_moment_nat, rmse_ankle_moment_nat,
                   rmse_hip_moment_nat_l, rmse_knee_moment_nat_l, rmse_ankle_moment_nat_l,
                   # rmse_pelvistilt_moment_nat, rmse_pelvislist_moment_nat, rmse_pelvisrotation_moment_nat,
                   rmse_lumbarextension_moment_nat, rmse_lumbarbending_moment_nat, rmse_lumbarrotation_moment_nat,
                   rmse_armflex_r_moment_nat, rmse_armflex_l_moment_nat, rmse_elbowflex_r_moment_nat, rmse_elbowflex_l_moment_nat]
    avg_moment_rmse = sum(moment_rmse) / len(moment_rmse)
    std_moment_rmse = np.std(moment_rmse)
    # compute average moment RMSE for lower limb 
    moment_lower_rmse = [rmse_hip_moment_nat, rmse_knee_moment_nat, rmse_ankle_moment_nat,
                         rmse_hip_moment_nat_l, rmse_knee_moment_nat_l, rmse_ankle_moment_nat_l]
    avg_moment_lower_rmse = sum(moment_lower_rmse) / len(moment_lower_rmse)
    std_moment_lower_rmse = np.std(moment_lower_rmse)

    # compute R^2 for moments
    corr_hip_moment_nat = r2_score(ref_hip_moment_nat, sim_hip_moment_nat)
    corr_knee_moment_nat = r2_score(ref_knee_moment_nat, sim_knee_moment_nat)
    corr_ankle_moment_nat = r2_score(ref_ankle_moment_nat, sim_ankle_moment_nat)
    corr_hip_moment_l_nat = r2_score(ref_hip_moment_nat_l, sim_hip_moment_nat_l)
    corr_knee_moment_l_nat = r2_score(ref_knee_moment_nat_l, sim_knee_moment_nat_l)
    corr_ankle_moment_l_nat = r2_score(ref_ankle_moment_nat_l, sim_ankle_moment_nat_l)
    # corr_pelvistilt_moment_nat = r2_score(ref_pelvistilt_moment_nat, sim_pelvistilt_moment_nat)
    # corr_pelvislist_moment_nat = r2_score(ref_pelvislist_moment_nat, sim_pelvislist_moment_nat)
    # corr_pelvisrotation_moment_nat = r2_score(ref_pelvisrotation_moment_nat, sim_pelvisrotation_moment_nat)
    corr_lumbarextension_moment_nat = r2_score(ref_lumbarextension_moment_nat, sim_lumbarextension_moment_nat)
    corr_lumbarbending_moment_nat = r2_score(ref_lumbarbending_moment_nat, sim_lumbarbending_moment_nat)
    corr_lumbarrotation_moment_nat = r2_score(ref_lumbarrotation_moment_nat, sim_lumbarrotation_moment_nat)
    corr_armflex_r_moment_nat = r2_score(ref_armflex_r_moment_nat, sim_armflex_r_moment_nat)
    corr_armflex_l_moment_nat = r2_score(ref_armflex_l_moment_nat, sim_armflex_l_moment_nat)
    corr_elbowflex_r_moment_nat = r2_score(ref_elbowflex_r_moment_nat, sim_elbowflex_r_moment_nat)
    corr_elbowflex_l_moment_nat = r2_score(ref_elbowflex_l_moment_nat, sim_elbowflex_l_moment_nat)


    # Print RMSE values
    print('\n' + '='*60)
    print('RMSE Values - Natural Walking')
    print('='*60)
    print(f'Hip Angle RMSE:        {rmse_hip_r_nat:.4f} degrees')
    print(f'Knee Angle RMSE:       {rmse_knee_r_nat:.4f} degrees')
    print(f'Ankle Angle RMSE:      {rmse_ankle_r_nat:.4f} degrees')
    print(f'Pelvis Vertical RMSE:  {rmse_ty_nat:.4f} (normalized)')
    print(f'Hip Angle Left RMSE:   {rmse_hip_l_nat:.4f} degrees')
    print(f'Knee Angle Left RMSE:  {rmse_knee_l_nat:.4f} degrees')
    print(f'Ankle Angle Left RMSE: {rmse_ankle_l_nat:.4f} degrees')
    print(f'Pelvis Tilt RMSE:     {rmse_pelvistilt_nat:.4f} degrees')
    print(f'Pelvis List RMSE:     {rmse_pelvislist_nat:.4f} degrees')
    print(f'Pelvis Rotation RMSE: {rmse_pelvisrotation_nat:.4f} degrees')
    print(f'Lumbar Extension RMSE: {rmse_lumbarextension_nat:.4f} degrees')
    print(f'Lumbar Bending RMSE:  {rmse_lumbarbending_nat:.4f} degrees')
    print(f'Lumbar Rotation RMSE: {rmse_lumbarrotation_nat:.4f} degrees')
    print(f'Arm Flexion R RMSE:   {rmse_armflex_r_nat:.4f} degrees')
    print(f'Arm Flexion L RMSE:   {rmse_armflex_l_nat:.4f} degrees')
    print(f'Elbow Flexion R RMSE: {rmse_elbowflex_r_nat:.4f} degrees')
    print(f'Elbow Flexion L RMSE: {rmse_elbowflex_l_nat:.4f} degrees')
    print(f'Average RMSE across all kinematics: {avg_kin_rmse:.4f} degrees')
    print(f'Standard Deviation of RMSE across all kinematics: {std_kin_rmse:.4f} degrees')
    print(f'Average RMSE across lower limb kinematics: {avg_kin_lower_rmse:.4f} degrees')
    print(f'Standard Deviation of RMSE across lower limb kinematics: {std_kin_lower_rmse:.4f} degrees')
    print('\n')
    print(f'GRF Vertical RMSE:     {rmse_grf_y_nat:.4f} (BW)')
    print(f'GRF Horizontal RMSE:   {rmse_grf_x_nat:.4f} (BW)')
    print(f'Average GRF RMSE:     {avg_grf_rmse:.4f} (BW)')
    print(f'Standard Deviation of GRF RMSE: {std_grf_rmse:.4f} (BW)')
    print('\n')
    print(f'Hip Moment RMSE:       {rmse_hip_moment_nat:.4f} (Nm/kg)')
    print(f'Knee Moment RMSE:      {rmse_knee_moment_nat:.4f} (Nm/kg)')
    print(f'Ankle Moment RMSE:     {rmse_ankle_moment_nat:.4f} (Nm/kg)')
    print(f'Hip Moment Left RMSE:  {rmse_hip_moment_nat_l:.4f} (Nm/kg)')
    print(f'Knee Moment Left RMSE: {rmse_knee_moment_nat_l:.4f} (Nm/kg)')
    print(f'Ankle Moment Left RMSE: {rmse_ankle_moment_nat_l:.4f} (Nm/kg)')
    # print(f'Pelvis Tilt Moment RMSE: {rmse_pelvistilt_moment_nat:.4f} (Nm/kg)')
    # print(f'Pelvis List Moment RMSE: {rmse_pelvislist_moment_nat:.4f} (Nm/kg)')
    # print(f'Pelvis Rotation Moment RMSE: {rmse_pelvisrotation_moment_nat:.4f} (Nm/kg)')
    print(f'Lumbar Extension Moment RMSE: {rmse_lumbarextension_moment_nat:.4f} (Nm/kg)')
    print(f'Lumbar Bending Moment RMSE: {rmse_lumbarbending_moment_nat:.4f} (Nm/kg)')
    print(f'Lumbar Rotation Moment RMSE: {rmse_lumbarrotation_moment_nat:.4f} (Nm/kg)')
    print(f'Arm Flexion R Moment RMSE: {rmse_armflex_r_moment_nat:.4f} (Nm/kg)')
    print(f'Arm Flexion L Moment RMSE: {rmse_armflex_l_moment_nat:.4f} (Nm/kg)')
    print(f'Elbow Flexion R Moment RMSE: {rmse_elbowflex_r_moment_nat:.4f} (Nm/kg)')
    print(f'Elbow Flexion L Moment RMSE: {rmse_elbowflex_l_moment_nat:.4f} (Nm/kg)')
    print(f'Average Moment RMSE across all moments: {avg_moment_rmse:.4f} (Nm/kg)')
    print(f'Standard Deviation of Moment RMSE across all moments: {std_moment_rmse:.4f} (Nm/kg)')
    print(f'Average Moment RMSE across lower limb moments: {avg_moment_lower_rmse:.4f} (Nm/kg)')
    print(f'Standard Deviation of Moment RMSE across lower limb moments: {std_moment_lower_rmse:.4f} (Nm/kg)')
    print('='*60 + '\n')

    # # Print R^2 values
    # print('\n' + '='*60)
    # print('R^2 Values - Natural Walking')
    # print('='*60)
    # print(f'Hip Angle R^2:        {corr_hip_r_nat:.4f}')
    # print(f'Knee Angle R^2:       {corr_knee_r_nat:.4f}')
    # print(f'Ankle Angle R^2:      {corr_ankle_r_nat:.4f}')
    # print(f'Pelvis Vertical R^2:  {corr_ty_nat:.4f}')
    # print(f'Hip Angle Left R^2:   {corr_hip_l_nat:.4f}')
    # print(f'Knee Angle Left R^2:  {corr_knee_l_nat:.4f}')
    # print(f'Ankle Angle Left R^2: {corr_ankle_l_nat:.4f}')
    # print(f'Pelvis Tilt R^2:     {corr_pelvistilt_nat:.4f}')
    # print(f'Pelvis List R^2:     {corr_pelvislist_nat:.4f}')
    # print(f'Pelvis Rotation R^2: {corr_pelvisrotation_nat:.4f}')
    # print(f'Lumbar Extension R^2: {corr_lumbarextension_nat:.4f}')
    # print(f'Lumbar Bending R^2:  {corr_lumbarbending_nat:.4f}')
    # print(f'Lumbar Rotation R^2: {corr_lumbarrotation_nat:.4f}')
    # print(f'Arm Flexion R R^2:   {corr_armflex_r_nat:.4f}')
    # print(f'Arm Flexion L R^2:   {corr_armflex_l_nat:.4f}')
    # print(f'Elbow Flexion R R^2: {corr_elbowflex_r_nat:.4f}')
    # print(f'Elbow Flexion L R^2: {corr_elbowflex_l_nat:.4f}')
    # print('\n')
    # print(f'GRF Vertical R^2:     {corr_grf_y_nat:.4f}')
    # print(f'GRF Horizontal R^2:   {corr_grf_x_nat:.4f}')
    # print('\n')
    # print(f'Hip Moment R^2:       {corr_hip_moment_nat:.4f}')
    # print(f'Knee Moment R^2:      {corr_knee_moment_nat:.4f}')
    # print(f'Ankle Moment R^2:     {corr_ankle_moment_nat:.4f}')
    # print(f'Hip Moment Left R^2:  {corr_hip_moment_l_nat:.4f}')
    # print(f'Knee Moment Left R^2: {corr_knee_moment_l_nat:.4f}')
    # print(f'Ankle Moment Left R^2: {corr_ankle_moment_l_nat:.4f}')
    # # print(f'Pelvis Tilt Moment R^2: {corr_pelvistilt_moment_nat:.4f}')
    # # print(f'Pelvis List Moment R^2: {corr_pelvislist_moment_nat:.4f}')
    # # print(f'Pelvis Rotation Moment R^2: {corr_pelvisrotation_moment_nat:.4f}')
    # print(f'Lumbar Extension Moment R^2: {corr_lumbarextension_moment_nat:.4f}')
    # print(f'Lumbar Bending Moment R^2: {corr_lumbarbending_moment_nat:.4f}')
    # print(f'Lumbar Rotation Moment R^2: {corr_lumbarrotation_moment_nat:.4f}')
    # print(f'Arm Flexion R Moment R^2: {corr_armflex_r_moment_nat:.4f}')
    # print(f'Arm Flexion L Moment R^2: {corr_armflex_l_moment_nat:.4f}')
    # print(f'Elbow Flexion R Moment R^2: {corr_elbowflex_r_moment_nat:.4f}')
    # print(f'Elbow Flexion L Moment R^2: {corr_elbowflex_l_moment_nat:.4f}')
    # print('='*60 + '\n')
    
    # Store RMSE values in a dictionary
    rmse_dict = {
        'Variable': ['Hip Angle', 'Knee Angle', 'Ankle Angle', 'Pelvis Vertical',
                     'Hip Angle Left', 'Knee Angle Left', 'Ankle Angle Left', 
                     'Pelvis Tilt', 'Pelvis List', 'Pelvis Rotation',
                     'Lumbar Extension', 'Lumbar Bending', 'Lumbar Rotation',
                     'Arm Flexion R', 'Arm Flexion L', 'Elbow Flexion R', 'Elbow Flexion L',
                     'GRF Vertical', 'GRF Horizontal', 
                     'Hip Moment', 'Knee Moment', 'Ankle Moment',
                     'Hip Moment Left', 'Knee Moment Left', 'Ankle Moment Left',
                    #  'Pelvis Tilt Moment', 'Pelvis List Moment', 'Pelvis Rotation Moment', 
                     'Lumbar Extension Moment', 'Lumbar Bending Moment', 'Lumbar Rotation Moment', 
                     'Arm Flexion R Moment', 'Arm Flexion L Moment', 'Elbow Flexion R Moment', 'Elbow Flexion L Moment'],
        'RMSE': [rmse_hip_r_nat, rmse_knee_r_nat, rmse_ankle_r_nat, rmse_ty_nat, 
                 rmse_hip_l_nat, rmse_knee_l_nat, rmse_ankle_l_nat, 
                 rmse_pelvistilt_nat, rmse_pelvislist_nat, rmse_pelvisrotation_nat, 
                 rmse_lumbarextension_nat, rmse_lumbarbending_nat, rmse_lumbarrotation_nat,
                 rmse_armflex_r_nat, rmse_armflex_l_nat, rmse_elbowflex_r_nat, rmse_elbowflex_l_nat,
                 rmse_grf_y_nat, rmse_grf_x_nat, 
                 rmse_hip_moment_nat, rmse_knee_moment_nat, rmse_ankle_moment_nat,
                 rmse_hip_moment_nat_l, rmse_knee_moment_nat_l, rmse_ankle_moment_nat_l,
                #  rmse_pelvistilt_moment_nat, rmse_pelvislist_moment_nat, rmse_pelvisrotation_moment_nat, 
                 rmse_lumbarextension_moment_nat, rmse_lumbarbending_moment_nat, rmse_lumbarrotation_moment_nat,
                 rmse_armflex_r_moment_nat, rmse_armflex_l_moment_nat, rmse_elbowflex_r_moment_nat, rmse_elbowflex_l_moment_nat],
        # 'Units': ['deg', 'deg', 'deg', 'normalized', 'BW', 'BW', 'Nm/kg', 'Nm/kg', 'Nm/kg']
    }
    # store R^2 values in a dictionary
    r2_dict = {
        'Variable': ['Hip Angle', 'Knee Angle', 'Ankle Angle', 'Pelvis Vertical',
                     'Hip Angle Left', 'Knee Angle Left', 'Ankle Angle Left', 
                     'Pelvis Tilt', 'Pelvis List', 'Pelvis Rotation',
                     'Lumbar Extension', 'Lumbar Bending', 'Lumbar Rotation',
                     'Arm Flexion R', 'Arm Flexion L', 'Elbow Flexion R', 'Elbow Flexion L',
                     'GRF Vertical', 'GRF Horizontal', 
                     'Hip Moment', 'Knee Moment', 'Ankle Moment',
                     'Hip Moment Left', 'Knee Moment Left', 'Ankle Moment Left',
                    #  'Pelvis Tilt Moment', 'Pelvis List Moment', 'Pelvis Rotation Moment',
                     'Lumbar Extension Moment', 'Lumbar Bending Moment', 'Lumbar Rotation Moment',
                     'Arm Flexion R Moment', 'Arm Flexion L Moment', 'Elbow Flexion R Moment', 'Elbow Flexion L Moment'],
        'R^2': [corr_hip_r_nat, corr_knee_r_nat, corr_ankle_r_nat, corr_ty_nat,
                corr_hip_l_nat, corr_knee_l_nat, corr_ankle_l_nat,
                corr_pelvistilt_nat, corr_pelvislist_nat, corr_pelvisrotation_nat,
                corr_lumbarextension_nat, corr_lumbarbending_nat, corr_lumbarrotation_nat,
                corr_armflex_r_nat, corr_armflex_l_nat, corr_elbowflex_r_nat, corr_elbowflex_l_nat,
                corr_grf_y_nat, corr_grf_x_nat, 
                corr_hip_moment_nat, corr_knee_moment_nat, corr_ankle_moment_nat,
                corr_hip_moment_l_nat, corr_knee_moment_l_nat, corr_ankle_moment_l_nat,
                # corr_pelvistilt_moment_nat, corr_pelvislist_moment_nat, corr_pelvisrotation_moment_nat,
                corr_lumbarextension_moment_nat, corr_lumbarbending_moment_nat, corr_lumbarrotation_moment_nat,
                corr_armflex_r_moment_nat, corr_armflex_l_moment_nat, corr_elbowflex_r_moment_nat, corr_elbowflex_l_moment_nat],
    }


    rmse_df = pd.DataFrame(rmse_dict)
    # print(rmse_df.to_string())
    corr_df = pd.DataFrame(r2_dict)
    # print(corr_df.to_string())

    # Save to CSV
    figpath = os.getcwd() + '\\..\\..\\analysis\\'
    rmse_df.to_csv(figpath + 'validation_RMSE_40.csv', index=False)
    corr_df.to_csv(figpath + 'validation_R2_40.csv', index=False)

    ###########################
    # compute a pearson correlation on the data
    pearson_vars = [
        ('Hip Angle', sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat),
        ('Knee Angle', sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat),
        ('Ankle Angle', sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat),
        ('Pelvis Vertical', sim_ty_nat, ref_ty_nat),
        ('Hip Angle Left', sim_hip_angle_nat_l * 180 / np.pi, ref_hip_angle_nat_l),
        ('Knee Angle Left', sim_knee_angle_nat_l * 180 / np.pi, ref_knee_angle_nat_l),
        ('Ankle Angle Left', sim_ankle_angle_nat_l * 180 / np.pi, ref_ankle_angle_nat_l),
        ('Pelvis Tilt', sim_pelvistilt_nat * 180 / np.pi, ref_pelvistilt_nat),
        ('Pelvis List', sim_pelvislist_nat * 180 / np.pi, ref_pelvislist_nat),
        ('Pelvis Rotation', sim_pelvisrotation_nat * 180 / np.pi, ref_pelvisrotation_nat),
        ('Lumbar Extension', sim_lumbarextension_nat * 180 / np.pi, ref_lumbarextension_nat),
        ('Lumbar Bending', sim_lumbarbending_nat * 180 / np.pi, ref_lumbarbending_nat),
        ('Lumbar Rotation', sim_lumbarrotation_nat * 180 / np.pi, ref_lumbarrotation_nat),
        ('Arm Flexion R', sim_armflex_r_nat * 180 / np.pi, ref_armflex_r_nat),
        ('Arm Flexion L', sim_armflex_l_nat * 180 / np.pi, ref_armflex_l_nat),
        ('Elbow Flexion R', sim_elbowflex_r_nat * 180 / np.pi, ref_elbowflex_r_nat),
        ('Elbow Flexion L', sim_elbowflex_l_nat * 180 / np.pi, ref_elbowflex_l_nat),
        ('GRF Vertical', sim_grf_y_nat, ref_grf_y_nat),
        ('GRF Horizontal', sim_grf_x_nat, ref_grf_x_nat),
        ('Hip Moment', sim_hip_moment_nat, ref_hip_moment_nat),
        ('Knee Moment', sim_knee_moment_nat, ref_knee_moment_nat),
        ('Ankle Moment', sim_ankle_moment_nat, ref_ankle_moment_nat),
        ('Hip Moment Left', sim_hip_moment_nat_l, ref_hip_moment_nat_l),
        ('Knee Moment Left', sim_knee_moment_nat_l, ref_knee_moment_nat_l),
        ('Ankle Moment Left', sim_ankle_moment_nat_l, ref_ankle_moment_nat_l),
        # ('Pelvis Tilt Moment', sim_pelvistilt_moment_nat, ref_pelvistilt_moment_nat),
        # ('Pelvis List Moment', sim_pelvislist_moment_nat, ref_pelvislist_moment_nat),
        # ('Pelvis Rotation Moment', sim_pelvisrotation_moment_nat, ref_pelvisrotation_moment_nat),
        ('Lumbar Extension Moment', sim_lumbarextension_moment_nat, ref_lumbarextension_moment_nat),
        ('Lumbar Bending Moment', sim_lumbarbending_moment_nat, ref_lumbarbending_moment_nat),
        ('Lumbar Rotation Moment', sim_lumbarrotation_moment_nat, ref_lumbarrotation_moment_nat),
        ('Arm Flexion R Moment', sim_armflex_r_moment_nat, ref_armflex_r_moment_nat),
        ('Arm Flexion L Moment', sim_armflex_l_moment_nat, ref_armflex_l_moment_nat),
        ('Elbow Flexion R Moment', sim_elbowflex_r_moment_nat, ref_elbowflex_r_moment_nat),
        ('Elbow Flexion L Moment', sim_elbowflex_l_moment_nat, ref_elbowflex_l_moment_nat)
    ]

    pearson_metrics = []
    for label, sim_vals, ref_vals in pearson_vars:
        sim_vals = np.asarray(sim_vals).ravel()
        ref_vals = np.asarray(ref_vals).ravel()
        n = min(len(sim_vals), len(ref_vals))
        if n < 2:
            pearson = np.nan
        else:
            pearson = np.corrcoef(sim_vals[:n], ref_vals[:n])[0, 1]
        pearson_metrics.append({'Variable': label, 'Pearson r': pearson})

    pearson_df = pd.DataFrame(pearson_metrics)
    print('\n' + '='*60)
    print(pearson_df.to_string())
    pearson_df.to_csv(figpath + 'validation_Pearson_40.csv', index=False)

    kinematic_labels = {
        'Hip Angle', 'Knee Angle', 'Ankle Angle', 'Pelvis Vertical',
        'Hip Angle Left', 'Knee Angle Left', 'Ankle Angle Left',
        'Pelvis Tilt', 'Pelvis List', 'Pelvis Rotation',
        'Lumbar Extension', 'Lumbar Bending', 'Lumbar Rotation',
        'Arm Flexion R', 'Arm Flexion L', 'Elbow Flexion R', 'Elbow Flexion L'
    }
    lower_kinematic_labels = {
        'Hip Angle', 'Knee Angle', 'Ankle Angle', 
        'Hip Angle Left', 'Knee Angle Left', 'Ankle Angle Left'
        }
    moment_labels = {
        'Hip Moment', 'Knee Moment', 'Ankle Moment',
        'Hip Moment Left', 'Knee Moment Left', 'Ankle Moment Left',
        # 'Pelvis Tilt Moment', 'Pelvis List Moment', 'Pelvis Rotation Moment',
        'Lumbar Extension Moment', 'Lumbar Bending Moment', 'Lumbar Rotation Moment',
        'Arm Flexion R Moment', 'Arm Flexion L Moment',
        'Elbow Flexion R Moment', 'Elbow Flexion L Moment'
    }
    lower_moment_labels = {
        'Hip Moment', 'Knee Moment', 'Ankle Moment',
        'Hip Moment Left', 'Knee Moment Left', 'Ankle Moment Left'
        }
    grf_labels = {'GRF Vertical', 'GRF Horizontal'}
    kinematic_rs = [
        item['Pearson r'] for item in pearson_metrics
        if item['Variable'] in kinematic_labels
    ]
    moment_rs = [
        item['Pearson r'] for item in pearson_metrics
        if item['Variable'] in moment_labels
    ]
    grf_rs = [
        item['Pearson r'] for item in pearson_metrics
        if item['Variable'] in grf_labels
    ]
    lower_kinematic_rs = [
        item['Pearson r'] for item in pearson_metrics
        if item['Variable'] in lower_kinematic_labels
    ]
    lower_moment_rs = [
        item['Pearson r'] for item in pearson_metrics
        if item['Variable'] in lower_moment_labels
    ]
    avg_pearson_kin = np.nan if len(kinematic_rs) == 0 else float(np.nanmean(kinematic_rs))
    std_pearson_kin = np.nan if len(kinematic_rs) == 0 else float(np.nanstd(kinematic_rs))
    avg_pearson_mom = np.nan if len(moment_rs) == 0 else float(np.nanmean(moment_rs))
    std_pearson_mom = np.nan if len(moment_rs) == 0 else float(np.nanstd(moment_rs))
    avg_pearson_grf = np.nan if len(grf_rs) == 0 else float(np.nanmean(grf_rs))
    std_pearson_grf = np.nan if len(grf_rs) == 0 else float(np.nanstd(grf_rs))
    avg_pearson_lower_kin = np.nan if len(lower_kinematic_rs) == 0 else float(np.nanmean(lower_kinematic_rs))
    std_pearson_lower_kin = np.nan if len(lower_kinematic_rs) == 0 else float(np.nanstd(lower_kinematic_rs))
    avg_pearson_lower_mom = np.nan if len(lower_moment_rs) == 0 else float(np.nanmean(lower_moment_rs))
    std_pearson_lower_mom = np.nan if len(lower_moment_rs) == 0 else float(np.nanstd(lower_moment_rs))
    print(f'Average Pearson r (kinematics): {avg_pearson_kin:.4f}')
    print(f'Standard Deviation of Pearson r (kinematics): {std_pearson_kin:.4f}')
    print(f'Average Pearson r (moments): {avg_pearson_mom:.4f}')
    print(f'Standard Deviation of Pearson r (moments): {std_pearson_mom:.4f}')
    print(f'Average Pearson r (GRF): {avg_pearson_grf:.4f}')
    print(f'Standard Deviation of Pearson r (GRF): {std_pearson_grf:.4f}')
    print(f'Average Pearson r (lower limb kinematics): {avg_pearson_lower_kin:.4f}')
    print(f'Standard Deviation of Pearson r (lower limb kinematics): {std_pearson_lower_kin:.4f}')
    print(f'Average Pearson r (lower limb moments): {avg_pearson_lower_mom:.4f}')
    print(f'Standard Deviation of Pearson r (lower limb moments): {std_pearson_lower_mom:.4f}')
    
    print('\n' + '='*60)
    print('\n' + '='*60)
    print('above are likely the ones that we will use and report')
    print('\n' + '='*60)
    print('\n' + '='*60)
    

    ###########################
    # lag metrics: compute RMSE and correlation at the best temporal lag to account for timing differences between sim and ref.

    def best_lag_rmse_corr(sim_vals, ref_vals, max_shift):
        # Minimize RMSE over small temporal shifts to reduce timing sensitivity.
        sim_vals = np.asarray(sim_vals)
        ref_vals = np.asarray(ref_vals)
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        best_rmse = np.inf
        best_corr = np.nan
        best_shift = 0
        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                sim_seg = sim_vals[:n + shift]
                ref_seg = ref_vals[-shift:]
            elif shift > 0:
                sim_seg = sim_vals[shift:]
                ref_seg = ref_vals[:n - shift]
            else:
                sim_seg = sim_vals
                ref_seg = ref_vals

            if len(sim_seg) < 2:
                continue

            rmse = np.sqrt(np.mean((sim_seg - ref_seg) ** 2))
            corr = np.corrcoef(sim_seg, ref_seg)[0, 1]
            if rmse < best_rmse:
                best_rmse = rmse
                best_corr = corr
                best_shift = shift

        return best_rmse, best_corr, best_shift

    # Use a small shift window (2% of gait cycle) to account for timing offsets.
    x = np.linspace(0, 100, kin_simlen_nat)
    max_shift = max(1, int(0.05 * len(x)))
    lag_metrics = []

    lag_metrics.append({'Variable': 'Hip Angle',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Knee Angle',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Ankle Angle',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Pelvis Vertical',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_ty_nat, ref_ty_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_ty_nat, ref_ty_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_ty_nat, ref_ty_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'GRF Vertical',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_grf_y_nat, ref_grf_y_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_grf_y_nat, ref_grf_y_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_grf_y_nat, ref_grf_y_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'GRF Horizontal',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_grf_x_nat, ref_grf_x_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_grf_x_nat, ref_grf_x_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_grf_x_nat, ref_grf_x_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Hip Moment',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_hip_moment_nat, ref_hip_moment_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_hip_moment_nat, ref_hip_moment_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_hip_moment_nat, ref_hip_moment_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Knee Moment',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_knee_moment_nat, ref_knee_moment_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_knee_moment_nat, ref_knee_moment_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_knee_moment_nat, ref_knee_moment_nat, max_shift)[1]})
    lag_metrics.append({'Variable': 'Ankle Moment',
                        'Best Lag (samples)': best_lag_rmse_corr(sim_ankle_moment_nat, ref_ankle_moment_nat, max_shift)[2],
                        'RMSE': best_lag_rmse_corr(sim_ankle_moment_nat, ref_ankle_moment_nat, max_shift)[0],
                        'Correlation': best_lag_rmse_corr(sim_ankle_moment_nat, ref_ankle_moment_nat, max_shift)[1]})

    lag_df = pd.DataFrame(lag_metrics)
    # print(lag_df.to_string())
    lag_df.to_csv(figpath + 'validation_bestlag_RMSE_corr_40.csv', index=False)

    def rmse_by_lag(sim_vals, ref_vals, max_shift):
        sim_vals = np.asarray(sim_vals)
        ref_vals = np.asarray(ref_vals)
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        lags = np.arange(-max_shift, max_shift + 1)
        rmses = np.zeros(len(lags))
        for i, shift in enumerate(lags):
            if shift < 0:
                sim_seg = sim_vals[:n + shift]
                ref_seg = ref_vals[-shift:]
            elif shift > 0:
                sim_seg = sim_vals[shift:]
                ref_seg = ref_vals[:n - shift]
            else:
                sim_seg = sim_vals
                ref_seg = ref_vals
            rmses[i] = np.sqrt(np.mean((sim_seg - ref_seg) ** 2))
        return lags, rmses

    # Visualize RMSE vs lag so timing sensitivity is easy to see.
    lag_plot_vars = [
        ('Hip Angle', sim_hip_angle_nat * 180 / np.pi, ref_hip_angle_nat),
        ('Knee Angle', sim_knee_angle_nat * 180 / np.pi, ref_knee_angle_nat),
        ('Ankle Angle', sim_ankle_angle_nat * 180 / np.pi, ref_ankle_angle_nat),
        ('Pelvis Vertical', sim_ty_nat, ref_ty_nat),
        ('GRF Vertical', sim_grf_y_nat, ref_grf_y_nat),
        ('GRF Horizontal', sim_grf_x_nat, ref_grf_x_nat),
        ('Hip Moment', sim_hip_moment_nat, ref_hip_moment_nat),
        ('Knee Moment', sim_knee_moment_nat, ref_knee_moment_nat),
        ('Ankle Moment', sim_ankle_moment_nat, ref_ankle_moment_nat),
    ]

    # fig_lag, ax_lag = plt.subplots(3, 3, figsize=(12, 9), dpi=300)
    # ax_lag = ax_lag.flatten()
    # for i, (label, sim_vals, ref_vals) in enumerate(lag_plot_vars):
    #     lags, rmses = rmse_by_lag(sim_vals, ref_vals, max_shift)
    #     ax_lag[i].plot(lags, rmses, color='black')
    #     ax_lag[i].axvline(0, color='gray', linestyle='--', linewidth=1)
    #     ax_lag[i].set_title(label, fontsize=10)
    #     ax_lag[i].set_xlabel('Lag (samples)')
    #     ax_lag[i].set_ylabel('RMSE')

    # plt.tight_layout()
    # plt.savefig(figpath + 'validation_RMSE_vs_lag_40.png')


    ###########################
    # DTW metrics: compute DTW distance and normalized DTW distance to account for timing differences between sim and ref.

    def compute_dtw_metrics(sim_vals, ref_vals):
        sim_vals = np.asarray(sim_vals).ravel()
        ref_vals = np.asarray(ref_vals).ravel()
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        dtw_dist, _ = fastdtw(sim_vals, ref_vals, dist=lambda u, v: abs(u - v))
        dtw_norm = dtw_dist / max(1, n)
        return dtw_dist, dtw_norm

    dtw_metrics = []
    for label, sim_vals, ref_vals in lag_plot_vars:
        dtw_dist, dtw_norm = compute_dtw_metrics(sim_vals, ref_vals)
        dtw_metrics.append({'Variable': label, 'DTW Distance': dtw_dist, 'DTW Distance (norm)': dtw_norm})

    dtw_df = pd.DataFrame(dtw_metrics)
    # print(dtw_df.to_string())
    dtw_df.to_csv(figpath + 'validation_DTW_40.csv', index=False)

    ###########################
    # normalized RMSE: compute RMSE normalized by the range of the reference data to account for differences in variable magnitude.

    def normalize_std_to_length(std_vals, n):
        std_vals = np.asarray(std_vals).ravel()
        if len(std_vals) == n:
            return std_vals
        if len(std_vals) < 2:
            return np.full(n, np.nan)
        x_std = np.linspace(0, 100, len(std_vals))
        x_target = np.linspace(0, 100, n)
        return np.interp(x_target, x_std, std_vals)

    def normalized_rmse_by_std(sim_vals, ref_vals, std_vals):
        sim_vals = np.asarray(sim_vals).ravel()
        ref_vals = np.asarray(ref_vals).ravel()
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        std_vals = normalize_std_to_length(std_vals, n)
        valid = np.isfinite(std_vals) & (std_vals > 0)
        if not np.any(valid):
            return np.nan
        return np.sqrt(np.mean(((sim_vals[valid] - ref_vals[valid]) / std_vals[valid]) ** 2))

    std_map = {
        'Hip Angle': std_nat['hip_flexion_r'],
        'Knee Angle': std_nat['knee_angle_r'],
        'Ankle Angle': std_nat['ankle_angle_r'],
        'Pelvis Vertical': std_nat['pelvis_ty'],
        'GRF Vertical': stdgrfnat['calcn_r_Right_GRF_Fy'],
        'GRF Horizontal': stdgrfnat['calcn_r_Right_GRF_Fx'],
        'Hip Moment': stdmomnat['hip_flexion_r_moment'],
        'Knee Moment': stdmomnat['knee_angle_r_moment'],
        'Ankle Moment': stdmomnat['ankle_angle_r_moment'],
    }

    nrmse_metrics = []
    for label, sim_vals, ref_vals in lag_plot_vars:
        nrmse_metrics.append({
            'Variable': label,
            'NRMSE (std)': normalized_rmse_by_std(sim_vals, ref_vals, std_map[label])
        })

    nrmse_df = pd.DataFrame(nrmse_metrics)
    # print(nrmse_df.to_string())
    nrmse_df.to_csv(figpath + 'validation_NRMSE_40.csv', index=False)

    ###########################
    # normalized RMSE with lag: compute normalized RMSE at the best temporal lag to account for timing differences and variable magnitude.

    def best_lag_nrmse_by_std(sim_vals, ref_vals, std_vals, max_shift):
        sim_vals = np.asarray(sim_vals).ravel()
        ref_vals = np.asarray(ref_vals).ravel()
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        best_nrmse = np.inf
        best_shift = 0

        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                sim_seg = sim_vals[:n + shift]
                ref_seg = ref_vals[-shift:]
            elif shift > 0:
                sim_seg = sim_vals[shift:]
                ref_seg = ref_vals[:n - shift]
            else:
                sim_seg = sim_vals
                ref_seg = ref_vals

            if len(sim_seg) < 2:
                continue

            std_seg = normalize_std_to_length(std_vals, len(sim_seg))
            valid = np.isfinite(std_seg) & (std_seg > 0)
            if not np.any(valid):
                continue

            nrmse = np.sqrt(np.mean(((sim_seg[valid] - ref_seg[valid]) / std_seg[valid]) ** 2))
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_shift = shift

        return best_nrmse, best_shift

    nrmse_lag_metrics = []
    for label, sim_vals, ref_vals in lag_plot_vars:
        best_nrmse, best_shift = best_lag_nrmse_by_std(sim_vals, ref_vals, std_map[label], max_shift)
        nrmse_lag_metrics.append({
            'Variable': label,
            'Best Lag (samples)': best_shift,
            'NRMSE (std, best lag)': best_nrmse
        })

    nrmse_lag_df = pd.DataFrame(nrmse_lag_metrics)
    # print(nrmse_lag_df.to_string())
    nrmse_lag_df.to_csv(figpath + 'validation_NRMSE_bestlag_40.csv', index=False)

    ###########################
    # CMC: compute the coefficient of multiple correlation between the sim and ref curves to assess overall similarity in shape.

    def cmc_two_waveforms(sim_vals, ref_vals):
        sim_vals = np.asarray(sim_vals)
        ref_vals = np.asarray(ref_vals)
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        if n < 2:
            return np.nan
        data = np.vstack([sim_vals, ref_vals])
        mean_time = np.mean(data, axis=0)
        grand_mean = np.mean(data)
        sse = np.sum((data - mean_time) ** 2)
        sst = np.sum((data - grand_mean) ** 2)
        if sst == 0:
            return np.nan
        cmc_val = 1 - (sse / sst)
        return np.sqrt(max(0.0, cmc_val))

    cmc_metrics = []
    for label, sim_vals, ref_vals in lag_plot_vars:
        cmc_metrics.append({'Variable': label, 'CMC': cmc_two_waveforms(sim_vals, ref_vals)})

    cmc_df = pd.DataFrame(cmc_metrics)
    # print(cmc_df.to_string())
    cmc_df.to_csv(figpath + 'validation_CMC_40.csv', index=False)

    ###########################
    # ICC: compute the intraclass correlation coefficient between the sim and ref curves to assess agreement in both shape and magnitude.

    def icc_2_1(sim_vals, ref_vals):
        sim_vals = np.asarray(sim_vals)
        ref_vals = np.asarray(ref_vals)
        n = min(len(sim_vals), len(ref_vals))
        sim_vals = sim_vals[:n]
        ref_vals = ref_vals[:n]
        if n < 2:
            return np.nan
        data = np.vstack([sim_vals, ref_vals]).T
        n_targets, k_raters = data.shape
        mean_target = np.mean(data, axis=1)
        mean_rater = np.mean(data, axis=0)
        grand_mean = np.mean(data)

        ss_target = k_raters * np.sum((mean_target - grand_mean) ** 2)
        ss_rater = n_targets * np.sum((mean_rater - grand_mean) ** 2)
        ss_error = np.sum((data - mean_target[:, None] - mean_rater + grand_mean) ** 2)

        df_target = n_targets - 1
        df_rater = k_raters - 1
        df_error = df_target * df_rater

        if df_target == 0 or df_error == 0:
            return np.nan

        ms_target = ss_target / df_target
        ms_rater = ss_rater / df_rater if df_rater > 0 else 0.0
        ms_error = ss_error / df_error

        denom = ms_target + (k_raters - 1) * ms_error + (k_raters * (ms_rater - ms_error) / n_targets)
        if denom == 0:
            return np.nan
        return (ms_target - ms_error) / denom

    icc_metrics = []
    for label, sim_vals, ref_vals in lag_plot_vars:
        icc_metrics.append({'Variable': label, 'ICC(2,1)': icc_2_1(sim_vals, ref_vals)})

    icc_df = pd.DataFrame(icc_metrics)
    # print(icc_df.to_string())
    icc_df.to_csv(figpath + 'validation_ICC_40.csv', index=False)
    








    #################################
    # plotting 
    # possibly inputing other plots here to illustrate the exotendon results....
    fig, ax = plt.subplots(5, 4, figsize=(13, 15), dpi=500)
    x = np.linspace(0, 100, kin_simlen_nat)
    
    # Hip angle
    # ax[1,0].plot(x, ref_hip_angle_nat, label='Nat. Ref', color='orange')
    # ax[1,0].plot(x, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[1,0].fill_between(x, ref_hip_angle_nat - 2*std_nat['hip_flexion_r'], ref_hip_angle_nat + 2*std_nat['hip_flexion_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[1,0].fill_between(x, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[1,0].plot(x, sim_hip_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1,0].plot(x, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[1,0].set_title('Hip Angle', fontsize=14)
    ax[1,0].tick_params(axis='y', labelsize=14)    
    ax[1,0].tick_params(axis='x', labelsize=14)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=14)
    # ax[1,0].legend(fontsize=14, loc='lower right')

    # Knee angle
    # ax[2,0].plot(x, ref_knee_angle_nat, label='Nat. Ref', color='orange')
    # ax[2,0].plot(x, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[2,0].fill_between(x, ref_knee_angle_nat - 2*std_nat['knee_angle_r'], ref_knee_angle_nat + 2*std_nat['knee_angle_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[2,0].fill_between(x, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[2,0].plot(x, sim_knee_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2,0].plot(x, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[2,0].set_title('Knee Angle', fontsize=14)
    ax[2,0].set_ylabel('Angle (deg)', fontsize=14)
    ax[2,0].tick_params(axis='y', labelsize=14)    
    ax[2,0].tick_params(axis='x', labelsize=14)
    # ax[2,0].legend(fontsize=14, loc='upper left')

    # Ankle angle
    # ax[3,0].plot(x, ref_ankle_angle_nat, label='Nat. Ref', color='orange')
    # ax[3,0].plot(x, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[3,0].fill_between(x, ref_ankle_angle_nat - 2*std_nat['ankle_angle_r'], ref_ankle_angle_nat + 2*std_nat['ankle_angle_r'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[3,0].fill_between(x, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[3,0].plot(x, sim_ankle_angle_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[3,0].plot(x, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[3,0].set_title('Ankle Angle', fontsize=14)
    ax[3,0].set_ylabel('Angle (deg)', fontsize=14)
    ax[3,0].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,0].tick_params(axis='y', labelsize=14)    
    ax[3,0].tick_params(axis='x', labelsize=14)
    # ax[3,0].legend(fontsize=14, loc='upper right')

    # Hip moment
    # ax[1,1].plot(x, ref_hip_moment_nat, label='Nat. Ref', color='orange')
    # ax[1,1].plot(x, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[1,1].fill_between(x, ref_hip_moment_nat - 2*stdmomnat['hip_flexion_r_moment'], ref_hip_moment_nat + 2*stdmomnat['hip_flexion_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[1,1].fill_between(x, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2)
    ax[1,1].plot(x, sim_hip_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1,1].plot(x, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1,1].set_title('Hip Moment', fontsize=14)
    ax[1,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[1,1].tick_params(axis='y', labelsize=14)    
    ax[1,1].tick_params(axis='x', labelsize=14)
    # ax[1,1].legend(fontsize=14, loc='upper right')

    # Knee moment
    # ax[2,1].plot(x, ref_knee_moment_nat, label='Nat. Ref', color='orange')
    # ax[2,1].plot(x, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[2,1].fill_between(x, ref_knee_moment_nat - 2*stdmomnat['knee_angle_r_moment'], ref_knee_moment_nat + 2*stdmomnat['knee_angle_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[2,1].fill_between(x, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2)
    ax[2,1].plot(x, sim_knee_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2,1].plot(x, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2,1].set_title('Knee Moment', fontsize=14)
    ax[2,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[2,1].tick_params(axis='y', labelsize=14)    
    ax[2,1].tick_params(axis='x', labelsize=14)
    # ax[2,1].legend(fontsize=14, loc='lower right')

    # Ankle moment
    # ax[3,1].plot(x, ref_ankle_moment_nat, label='Nat. Ref', color='orange')
    # ax[3,1].plot(x, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[3,1].fill_between(x, ref_ankle_moment_nat - 2*stdmomnat['ankle_angle_r_moment'], ref_ankle_moment_nat + 2*stdmomnat['ankle_angle_r_moment'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[3,1].fill_between(x, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2)
    ax[3,1].plot(x, sim_ankle_moment_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[3,1].plot(x, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[3,1].set_title('Ankle Moment', fontsize=14)
    ax[3,1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[3,1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[3,1].tick_params(axis='y', labelsize=14)    
    ax[3,1].tick_params(axis='x', labelsize=14)
    # ax[3,1].legend(fontsize=14, loc='lower right')

    # Pelvis ty
    # ax[4,0].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # ax[4,0].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    ax[4,0].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[4,0].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2)
    ax[4,0].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[4,0].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[4,0].set_title('Pelvis Vertical Translation', fontsize=14)
    ax[4,0].set_ylabel('Translation (m/height)', fontsize=14)
    ax[4,0].tick_params(axis='y', labelsize=14)    
    ax[4,0].tick_params(axis='x', labelsize=14)
    # ax[4,0].legend(fontsize=14, loc='upper right')

    # GRF y
    # Fill between for stdgrfnat and stdgrfexo for y GRF
    ax[0,0].fill_between(x, ref_grf_y_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fy'], ref_grf_y_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fy'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[0,0].fill_between(x, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2)
    # ax[0,0].plot(x, ref_grf_y_nat, label='Nat. Ref', color='orange')
    # ax[0,0].plot(x, ref_grf_y_exo, label='Exo Ref', color='purple')
    ax[0,0].plot(x, sim_grf_y_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0,0].plot(x, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('Vertical GRF', fontsize=14)
    ax[0,0].set_ylabel('Force (BW)', fontsize=14)
    ax[0,0].tick_params(axis='y', labelsize=14)    
    ax[0,0].tick_params(axis='x', labelsize=14)
    # ax[0,0].legend(fontsize=14, loc='upper right')

    # GRF x
    ax[0,1].fill_between(x, ref_grf_x_nat - 2*stdgrfnat['calcn_r_Right_GRF_Fx'], ref_grf_x_nat + 2*stdgrfnat['calcn_r_Right_GRF_Fx'], color='orange', alpha=0.2, label='Nat. Ref ±2SD')
    # ax[0,1].fill_between(x, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2)
    # ax[0,1].plot(x, ref_grf_x_nat, label='Nat. Ref', color='orange')
    # ax[0,1].plot(x, ref_grf_x_exo, label='Exo Ref', color='purple')
    ax[0,1].plot(x, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0,1].plot(x, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('Horizontal GRF', fontsize=14)
    ax[0,1].set_ylabel('Force (BW)', fontsize=14)
    ax[0,1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[0,1].tick_params(axis='y', labelsize=14)    
    ax[0,1].tick_params(axis='x', labelsize=14)
    # ax[0,1].legend(fontsize=14, loc='upper right')
    

    # Turn off unused subplots (axes) in the 6x4 grid
    ax[4, 1].axis('off')
    ax[4, 3].axis('off')

    # Hide the last subplot and use it to display the legend   

    # get the legend labels from the previous subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[4, 1].legend(handles, labels, loc='center left', fontsize=14)
    handles_e, labels_e = ax[0, 2].get_legend_handles_labels()
    ax[4, 3].legend(handles_e, labels_e, loc='center left', fontsize=14)

    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_40.png')
    plt.show()
    return


# create a function for plotting a nice figure for use as a paper validation figure. Focus on saggital metrics. 
def saggitalValidationHalf27(simNat, simExo, iknat2D, ikexo2D, labels2D, coordinates_sim_clean, mean_nat, std_nat, mean_exo, std_exo, GRFsimnat, GRFsimexo, GRFrefnat, GRFrefexo, meangrfnat, stdgrfnat, meangrfexo, stdgrfexo, natmomentfile, exomomentfile, idnat, idexo, meanmomnat, stdmomnat, meanmomexo, stdmomexo, modelfile):
    # load the model and get the mass
    model = osim.Model(modelfile)
    mass = model.getTotalMass(model.initSystem())
    height = 1.78
    ## starting with the kinematics
    # get the length of the simulation data
    kin_simlen_nat = len(simNat.getIndependentColumn())
    kin_simlen_exo = len(simExo.getIndependentColumn())
    if kin_simlen_nat != kin_simlen_exo:
        print('Simulation kinematic data lengths do not match. Exiting.')
        return
    # get the sim kinematics
    sim_hip_angle_r_nat = simNat.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_hip_angle_r_exo = simExo.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_r_nat = simNat.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_knee_angle_r_exo = simExo.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_r_nat = simNat.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_ankle_angle_r_exo = simExo.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    
    # sim_ty_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    # sim_ty_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    
    sim_hip_angle_l_nat = simNat.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_hip_angle_l_exo = simExo.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_l_nat = simNat.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_knee_angle_l_exo = simExo.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_l_nat = simNat.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ankle_angle_l_exo = simExo.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()

    # # get the reference data
    # ref_hip_angle_r_nat = mean_nat['hip_flexion_r']
    # ref_hip_angle_r_exo = mean_exo['hip_flexion_r']
    # ref_knee_angle_r_nat = mean_nat['knee_angle_r']
    # ref_knee_angle_r_exo = mean_exo['knee_angle_r']
    # ref_ankle_angle_r_nat = mean_nat['ankle_angle_r']
    # ref_ankle_angle_r_exo = mean_exo['ankle_angle_r']
    # # ref_ty_nat = mean_nat['pelvis_ty']
    # # ref_ty_exo = mean_exo['pelvis_ty']
    # ref_hip_angle_l_nat = mean_nat['hip_flexion_l']
    # ref_hip_angle_l_exo = mean_exo['hip_flexion_l']
    # ref_knee_angle_l_nat = mean_nat['knee_angle_l']
    # ref_knee_angle_l_exo = mean_exo['knee_angle_l']
    # ref_ankle_angle_l_nat = mean_nat['ankle_angle_l']
    # ref_ankle_angle_l_exo = mean_exo['ankle_angle_l']
    
    ref_hip_angle_r_nat = iknat2D.getDependentColumn('hip_flexion_r').to_numpy()
    ref_hip_angle_r_exo = ikexo2D.getDependentColumn('hip_flexion_r').to_numpy()
    ref_knee_angle_r_nat = iknat2D.getDependentColumn('knee_angle_r').to_numpy()
    ref_knee_angle_r_exo = ikexo2D.getDependentColumn('knee_angle_r').to_numpy()
    ref_ankle_angle_r_nat = iknat2D.getDependentColumn('ankle_angle_r').to_numpy()
    ref_ankle_angle_r_exo = ikexo2D.getDependentColumn('ankle_angle_r').to_numpy()
    # ref_ty_nat = iknat2D.getDependentColumn('pelvis_ty').to_numpy()
    # ref_ty_exo = ikexo2D.getDependentColumn('pelvis_ty').to_numpy()
    ref_hip_angle_l_nat = iknat2D.getDependentColumn('hip_flexion_l').to_numpy()
    ref_hip_angle_l_exo = ikexo2D.getDependentColumn('hip_flexion_l').to_numpy()
    ref_knee_angle_l_nat = iknat2D.getDependentColumn('knee_angle_l').to_numpy()
    ref_knee_angle_l_exo = ikexo2D.getDependentColumn('knee_angle_l').to_numpy()
    ref_ankle_angle_l_nat = iknat2D.getDependentColumn('ankle_angle_l').to_numpy()
    ref_ankle_angle_l_exo = ikexo2D.getDependentColumn('ankle_angle_l').to_numpy()

    # ref_hip_angle_nat = mean_nat['hip_flexion_r']
    # ref_hip_angle_exo = mean_exo['hip_flexion_r']
    # ref_knee_angle_nat = mean_nat['knee_angle_r']
    # ref_knee_angle_exo = mean_exo['knee_angle_r']
    # ref_ankle_angle_nat = mean_nat['ankle_angle_r']
    # ref_ankle_angle_exo = mean_exo['ankle_angle_r']
    # ref_ty_nat = mean_nat['pelvis_ty']
    # ref_ty_exo = mean_exo['pelvis_ty']

    # get the length of the reference data
    kin_reflen_nat = len(ref_hip_angle_r_nat)
    kin_reflen_exo = len(ref_hip_angle_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation. 
    kin_xsim = np.linspace(0,100,kin_simlen_nat)
    kin_xref_nat = np.linspace(0,100,kin_reflen_nat)
    kin_xref_exo = np.linspace(0,100,kin_reflen_exo)
    
    # interpolate the reference data to the simulation data length
    ref_hip_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_r_nat)
    ref_hip_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_r_exo)
    ref_knee_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_r_nat)
    ref_knee_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_r_exo)
    ref_ankle_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_r_nat)
    ref_ankle_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_r_exo)
    # ref_ty_nat = np.interp(kin_xsim, kin_xref_nat, ref_ty_nat)
    # ref_ty_exo = np.interp(kin_xsim, kin_xref_exo, ref_ty_exo)
    ref_hip_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_l_nat)
    ref_hip_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_l_exo)
    ref_knee_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_l_nat)
    ref_knee_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_l_exo)
    ref_ankle_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_l_nat)
    ref_ankle_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_l_exo)
    # normalize the ty values to the height of the subject
    # ref_ty_nat = ref_ty_nat / height
    # ref_ty_exo = ref_ty_exo / height
    # sim_ty_nat = sim_ty_nat / height
    # sim_ty_exo = sim_ty_exo / height
    
    # now shorten to be one step, rather than the full gait cycle. 
    ref_hip_angle_r_nat = ref_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_hip_angle_r_exo = ref_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_knee_angle_r_nat = ref_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_r_exo = ref_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_r_nat = ref_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_r_exo = ref_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_hip_angle_l_nat = ref_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_hip_angle_l_exo = ref_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    ref_knee_angle_l_nat = ref_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_l_exo = ref_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_l_nat = ref_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_l_exo = ref_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]

    sim_hip_angle_r_nat = sim_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_hip_angle_r_exo = sim_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_nat = sim_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_exo = sim_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_nat = sim_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_exo = sim_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_nat = sim_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_exo = sim_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_nat = sim_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_exo = sim_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_nat = sim_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_exo = sim_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]

    # now get the ref std for plus and minus
    std_hip_angle_r_nat = 2*std_nat['hip_flexion_r']
    std_hip_angle_r_exo = 2*std_exo['hip_flexion_r']
    std_knee_angle_r_nat = 2*std_nat['knee_angle_r']
    std_knee_angle_r_exo = 2*std_exo['knee_angle_r']
    std_ankle_angle_r_nat = 2*std_nat['ankle_angle_r']
    std_ankle_angle_r_exo = 2*std_exo['ankle_angle_r']
    std_hip_angle_l_nat = 2*std_nat['hip_flexion_l']
    std_hip_angle_l_exo = 2*std_exo['hip_flexion_l']
    std_knee_angle_l_nat = 2*std_nat['knee_angle_l']
    std_knee_angle_l_exo = 2*std_exo['knee_angle_l']
    std_ankle_angle_l_nat = 2*std_nat['ankle_angle_l']
    std_ankle_angle_l_exo = 2*std_exo['ankle_angle_l']
    # now resample
    std_hip_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_nat)), std_hip_angle_r_nat)
    std_hip_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_exo)), std_hip_angle_r_exo)
    std_knee_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_nat)), std_knee_angle_r_nat)
    std_knee_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_exo)), std_knee_angle_r_exo)
    std_ankle_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_nat)), std_ankle_angle_r_nat)
    std_ankle_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_exo)), std_ankle_angle_r_exo)
    std_hip_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_hip_angle_l_nat)
    std_hip_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_l_exo)), std_hip_angle_l_exo)
    std_knee_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_nat)), std_knee_angle_l_nat)
    std_knee_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_exo)), std_knee_angle_l_exo)
    std_ankle_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_ankle_angle_l_nat)
    std_ankle_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_exo)), std_ankle_angle_l_exo)
    # and cut down to a step rather than the full gait cycle
    std_hip_angle_r_nat = std_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_hip_angle_r_exo = std_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_knee_angle_r_nat = std_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_r_exo = std_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_ankle_angle_r_nat = std_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_r_exo = std_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_hip_angle_l_nat = std_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_hip_angle_l_exo = std_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    std_knee_angle_l_nat = std_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_l_exo = std_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    std_ankle_angle_l_nat = std_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_l_exo = std_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]


    ## get the GRF data
    grfsimnat = osim.TimeSeriesTable(GRFsimnat)
    grfsimexo = osim.TimeSeriesTable(GRFsimexo)
    grfrefnat = osim.TimeSeriesTable(GRFrefnat)
    grfrefexo = osim.TimeSeriesTable(GRFrefexo)
    # get the length of the simulation data
    grf_simlen_nat = len(grfsimnat.getIndependentColumn())
    grf_simlen_exo = len(grfsimexo.getIndependentColumn())
    if grf_simlen_nat != grf_simlen_exo:
        print('Simulation GRF data lengths do not match. Exiting.')
        return
    # get the sim GRF data
    sim_grf_y_nat = grfsimnat.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_y_exo = grfsimexo.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_x_nat = grfsimnat.getDependentColumn('ground_force_r_vx').to_numpy()
    sim_grf_x_exo = grfsimexo.getDependentColumn('ground_force_r_vx').to_numpy()
    # get the reference GRF data
    ref_grf_y_nat = grfrefnat.getDependentColumn('rF_y').to_numpy()
    ref_grf_y_exo = grfrefexo.getDependentColumn('rF_y').to_numpy()
    ref_grf_x_nat = grfrefnat.getDependentColumn('rF_x').to_numpy()
    ref_grf_x_exo = grfrefexo.getDependentColumn('rF_x').to_numpy()
    # ref_grf_y_nat = meangrfnat['calcn_r_Right_GRF_Fy']
    # ref_grf_y_exo = meangrfexo['calcn_r_Right_GRF_Fy']
    # ref_grf_x_nat = meangrfnat['calcn_r_Right_GRF_Fx']
    # ref_grf_x_exo = meangrfexo['calcn_r_Right_GRF_Fx']
    # get the length of the reference data
    grf_reflen_nat = len(ref_grf_y_nat)
    grf_reflen_exo = len(ref_grf_y_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    grf_xsim = np.linspace(0,100,grf_simlen_nat)
    grf_xref_nat = np.linspace(0,100,grf_reflen_nat)
    grf_xref_exo = np.linspace(0,100,grf_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_grf_y_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_y_nat)
    ref_grf_y_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_y_exo)
    ref_grf_x_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_x_nat)
    ref_grf_x_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_x_exo)
    # divide all of the GRF data based on the mass of the model
    sim_grf_y_nat = sim_grf_y_nat/(mass*9.81)
    sim_grf_y_exo = sim_grf_y_exo/(mass*9.81)
    sim_grf_x_nat = sim_grf_x_nat/(mass*9.81)
    sim_grf_x_exo = sim_grf_x_exo/(mass*9.81)
    ref_grf_y_nat = ref_grf_y_nat/(mass*9.81)
    ref_grf_y_exo = ref_grf_y_exo/(mass*9.81)
    ref_grf_x_nat = ref_grf_x_nat/(mass*9.81)
    ref_grf_x_exo = ref_grf_x_exo/(mass*9.81)

    # now shorten to be one step rather than the full gait cycle. 
    ref_grf_y_nat = ref_grf_y_nat[0:int(grf_simlen_nat/2)]
    ref_grf_y_exo = ref_grf_y_exo[0:int(grf_simlen_nat/2)]
    ref_grf_x_nat = ref_grf_x_nat[0:int(grf_simlen_nat/2)]
    ref_grf_x_exo = ref_grf_x_exo[0:int(grf_simlen_nat/2)]
    
    sim_grf_y_nat = sim_grf_y_nat[0:int(grf_simlen_nat/2)]
    sim_grf_y_exo = sim_grf_y_exo[0:int(grf_simlen_nat/2)]
    sim_grf_x_nat = sim_grf_x_nat[0:int(grf_simlen_nat/2)]
    sim_grf_x_exo = sim_grf_x_exo[0:int(grf_simlen_nat/2)]

    # get the std data for plus and minus
    std_grf_y_nat = 2*stdgrfnat['calcn_r_Right_GRF_Fy']
    std_grf_y_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fy']
    std_grf_x_nat = 2*stdgrfnat['calcn_r_Right_GRF_Fx']
    std_grf_x_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fx']
    # now resample
    std_grf_y_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_nat)), std_grf_y_nat)
    std_grf_y_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_exo)), std_grf_y_exo)
    std_grf_x_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_nat)), std_grf_x_nat)
    std_grf_x_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_exo)), std_grf_x_exo)
    # and cut down to a step rather than the full gait cycle
    std_grf_y_nat = std_grf_y_nat[0:int(grf_simlen_nat/2)]
    std_grf_y_exo = std_grf_y_exo[0:int(grf_simlen_nat/2)]
    std_grf_x_nat = std_grf_x_nat[0:int(grf_simlen_nat/2)]
    std_grf_x_exo = std_grf_x_exo[0:int(grf_simlen_nat/2)]

    
    ## get the moment data
    natmoment = osim.TimeSeriesTable(natmomentfile)
    exomoment = osim.TimeSeriesTable(exomomentfile)
    natrefmoment = osim.TimeSeriesTable(idnat)
    exorefmoment = osim.TimeSeriesTable(idexo)
    # get the length of the simulation data
    moment_simlen_nat = len(natmoment.getIndependentColumn())
    moment_simlen_exo = len(exomoment.getIndependentColumn())
    if moment_simlen_nat != moment_simlen_exo:
        print('Simulation moment data lengths do not match. Exiting.')
        return
    
    # get the sim moment data
    sim_hip_moment_r_nat = natmoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_hip_moment_r_exo = exomoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_knee_moment_r_nat = natmoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_knee_moment_r_exo = exomoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_ankle_moment_r_nat = natmoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_ankle_moment_r_exo = exomoment.getDependentColumn('ankle_angle_r_moment').to_numpy()

    sim_hip_moment_l_nat = natmoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_hip_moment_l_exo = exomoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_knee_moment_l_nat = natmoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_knee_moment_l_exo = exomoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_ankle_moment_l_nat = natmoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_ankle_moment_l_exo = exomoment.getDependentColumn('ankle_angle_l_moment').to_numpy()

    # get the reference moment data
    # ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    # ref_hip_moment_r_exo = meanmomexo['hip_flexion_r_moment']
    # ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    # ref_knee_moment_r_exo = meanmomexo['knee_angle_r_moment']
    # ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    # ref_ankle_moment_r_exo = meanmomexo['ankle_angle_r_moment']
    # ref_hip_moment_l_nat = meanmomnat['hip_flexion_l_moment']
    # ref_hip_moment_l_exo = meanmomexo['hip_flexion_l_moment']
    # ref_knee_moment_l_nat = meanmomnat['knee_angle_l_moment']
    # ref_knee_moment_l_exo = meanmomexo['knee_angle_l_moment']
    # ref_ankle_moment_l_nat = meanmomnat['ankle_angle_l_moment']
    # ref_ankle_moment_l_exo = meanmomexo['ankle_angle_l_moment']
    
    ref_hip_moment_r_nat = idnat.getDependentColumn('hip_flexion_r_moment').to_numpy()
    ref_hip_moment_r_exo = idexo.getDependentColumn('hip_flexion_r_moment').to_numpy()
    ref_knee_moment_r_nat = idnat.getDependentColumn('knee_angle_r_moment').to_numpy()
    ref_knee_moment_r_exo = idexo.getDependentColumn('knee_angle_r_moment').to_numpy()
    ref_ankle_moment_r_nat = idnat.getDependentColumn('ankle_angle_r_moment').to_numpy()
    ref_ankle_moment_r_exo = idexo.getDependentColumn('ankle_angle_r_moment').to_numpy()
    ref_hip_moment_l_nat = idnat.getDependentColumn('hip_flexion_l_moment').to_numpy()
    ref_hip_moment_l_exo = idexo.getDependentColumn('hip_flexion_l_moment').to_numpy()
    ref_knee_moment_l_nat = idnat.getDependentColumn('knee_angle_l_moment').to_numpy()
    ref_knee_moment_l_exo = idexo.getDependentColumn('knee_angle_l_moment').to_numpy()
    ref_ankle_moment_l_nat = idnat.getDependentColumn('ankle_angle_l_moment').to_numpy()
    ref_ankle_moment_l_exo = idexo.getDependentColumn('ankle_angle_l_moment').to_numpy()
    
    # ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    # ref_hip_moment_r_exo = meanmomexo['hip_flexion_r_moment']
    # ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    # ref_knee_moment_r_exo = meanmomexo['knee_angle_r_moment']
    # ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    # ref_ankle_moment_r_exo = meanmomexo['ankle_angle_r_moment']

    # get the length of the reference data
    moment_reflen_nat = len(ref_hip_moment_r_nat)
    moment_reflen_exo = len(ref_hip_moment_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    moment_xsim = np.linspace(0,100,moment_simlen_nat)
    moment_xref_nat = np.linspace(0,100,moment_reflen_nat)
    moment_xref_exo = np.linspace(0,100,moment_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_hip_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_r_nat)
    ref_hip_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_r_exo)
    ref_knee_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_r_nat)
    ref_knee_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_r_exo)
    ref_ankle_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_r_nat)
    ref_ankle_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_r_exo)

    ref_hip_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_l_nat)
    ref_hip_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_l_exo)
    ref_knee_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_l_nat)
    ref_knee_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_l_exo)
    ref_ankle_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_l_nat)
    ref_ankle_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_l_exo)
    
    # normalize all of it to body mass
    sim_hip_moment_r_nat = sim_hip_moment_r_nat/mass    
    sim_hip_moment_r_exo = sim_hip_moment_r_exo/mass
    sim_knee_moment_r_nat = sim_knee_moment_r_nat/mass
    sim_knee_moment_r_exo = sim_knee_moment_r_exo/mass
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat/mass
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo/mass
    sim_hip_moment_l_nat = sim_hip_moment_l_nat/mass
    sim_hip_moment_l_exo = sim_hip_moment_l_exo/mass
    sim_knee_moment_l_nat = sim_knee_moment_l_nat/mass
    sim_knee_moment_l_exo = sim_knee_moment_l_exo/mass
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat/mass
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo/mass

    ref_hip_moment_r_nat = ref_hip_moment_r_nat/mass
    ref_hip_moment_r_exo = ref_hip_moment_r_exo/mass
    ref_knee_moment_r_nat = ref_knee_moment_r_nat/mass
    ref_knee_moment_r_exo = ref_knee_moment_r_exo/mass
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat/mass
    ref_ankle_moment_r_exo = ref_ankle_moment_r_exo/mass
    ref_hip_moment_l_nat = ref_hip_moment_l_nat/mass
    ref_hip_moment_l_exo = ref_hip_moment_l_exo/mass
    ref_knee_moment_l_nat = ref_knee_moment_l_nat/mass
    ref_knee_moment_l_exo = ref_knee_moment_l_exo/mass
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat/mass
    ref_ankle_moment_l_exo = ref_ankle_moment_l_exo/mass
    
    # and now shorten to the single step, rather than gait cycle. 
    ref_hip_moment_r_nat = ref_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_hip_moment_r_exo = ref_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_knee_moment_r_nat = ref_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_r_exo = ref_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_r_exo = ref_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_hip_moment_l_nat = ref_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_hip_moment_l_exo = ref_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    ref_knee_moment_l_nat = ref_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_l_exo = ref_knee_moment_l_exo[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_l_exo = ref_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]

    sim_hip_moment_r_nat = sim_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_r_exo = sim_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_nat = sim_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_exo = sim_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_nat = sim_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_exo = sim_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_nat = sim_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_exo = sim_knee_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    
    # now get the std data for plus and minus
    std_hip_moment_r_nat = 2*stdmomnat['hip_flexion_r_moment']
    std_hip_moment_r_exo = 2*stdmomexo['hip_flexion_r_moment']
    std_knee_moment_r_nat = 2*stdmomnat['knee_angle_r_moment']
    std_knee_moment_r_exo = 2*stdmomexo['knee_angle_r_moment']
    std_ankle_moment_r_nat = 2*stdmomnat['ankle_angle_r_moment']
    std_ankle_moment_r_exo = 2*stdmomexo['ankle_angle_r_moment']
    std_hip_moment_l_nat = 2*stdmomnat['hip_flexion_l_moment']
    std_hip_moment_l_exo = 2*stdmomexo['hip_flexion_l_moment']
    std_knee_moment_l_nat = 2*stdmomnat['knee_angle_l_moment']
    std_knee_moment_l_exo = 2*stdmomexo['knee_angle_l_moment']
    std_ankle_moment_l_nat = 2*stdmomnat['ankle_angle_l_moment']
    std_ankle_moment_l_exo = 2*stdmomexo['ankle_angle_l_moment']
    # now resample
    std_hip_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100,len(std_hip_moment_r_nat)), std_hip_moment_r_nat)
    std_hip_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_r_exo)), std_hip_moment_r_exo)
    std_knee_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_nat)), std_knee_moment_r_nat)
    std_knee_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_exo)), std_knee_moment_r_exo)
    std_ankle_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_nat)), std_ankle_moment_r_nat)
    std_ankle_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_exo)), std_ankle_moment_r_exo)
    std_hip_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_nat)), std_hip_moment_l_nat)
    std_hip_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_exo)), std_hip_moment_l_exo)
    std_knee_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_nat)), std_knee_moment_l_nat)
    std_knee_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_exo)), std_knee_moment_l_exo)
    std_ankle_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_nat)), std_ankle_moment_l_nat)
    std_ankle_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_exo)), std_ankle_moment_l_exo)
    # and cut down to a step rather than the full gait cycle
    std_hip_moment_r_nat = std_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_hip_moment_r_exo = std_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_r_nat = std_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_knee_moment_r_exo = std_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_ankle_moment_r_nat = std_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_ankle_moment_r_exo = std_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_hip_moment_l_nat = std_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_hip_moment_l_exo = std_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_l_nat = std_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_knee_moment_l_exo = std_knee_moment_l_exo[0:int(moment_simlen_nat/2)]   
    std_ankle_moment_l_nat = std_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_ankle_moment_l_exo = std_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    

    ##########################################################################
    # first is the natural validation figure in 3x3
    ##########################################################################
    ## now create the figure that we want. It should be a 3x3 grid of subplots. 
    # the first column should be hip knee and ankle angles
    # second column should be hip knee and ankle moments
    # third column should be the pelvis ty, GRF y, and GRF x
    fig, ax = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_mom = np.linspace(0, 100, moment_simlen_nat)[0:int(moment_simlen_nat/2)]
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    # Hip angle
    ax[0, 0].plot(x_kin, ref_hip_angle_r_nat, label='Nat. Ref', color='orange')
    # ax[0, 0].plot(x_kin, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[0, 0].fill_between(x_kin, ref_hip_angle_r_nat - std_hip_angle_r_nat, ref_hip_angle_r_nat + std_hip_angle_r_nat, color='orange', alpha=0.2)
    # ax[0, 0].fill_between(x_kin, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0, 0].plot(x_kin, sim_hip_angle_r_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0, 0].plot(x_kin, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0, 0].set_title('Hip Angle', fontsize=14)
    ax[0, 0].tick_params(axis='y', labelsize=14)    
    ax[0, 0].tick_params(axis='x', labelsize=14)
    ax[0, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[0, 0].legend(fontsize=14) #, loc='lower right')

    # Knee angle
    ax[1, 0].plot(x_kin, ref_knee_angle_r_nat, label='Nat. Ref', color='orange')
    # ax[1, 0].plot(x_kin, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[1, 0].fill_between(x_kin, ref_knee_angle_r_nat - std_knee_angle_r_nat, ref_knee_angle_r_nat + std_knee_angle_r_nat, color='orange', alpha=0.2)
    # ax[1, 0].fill_between(x_kin, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[1, 0].plot(x_kin, sim_knee_angle_r_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1, 0].plot(x_kin, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 0].set_title('Knee Angle', fontsize=14)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[1, 0].tick_params(axis='y', labelsize=14)    
    ax[1, 0].tick_params(axis='x', labelsize=14)
    ax[1, 0].legend(fontsize=14) #, loc='upper left')

    # Ankle angle
    ax[2, 0].plot(x_kin, ref_ankle_angle_r_nat, label='Nat. Ref', color='orange')
    # ax[2, 0].plot(x_kin, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[2, 0].fill_between(x_kin, ref_ankle_angle_r_nat - std_ankle_angle_r_nat, ref_ankle_angle_r_nat + std_ankle_angle_r_nat, color='orange', alpha=0.2)
    # ax[2, 0].fill_between(x_kin, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[2, 0].plot(x_kin, sim_ankle_angle_r_nat * 180 / np.pi, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2, 0].plot(x_kin, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 0].set_title('Ankle Angle', fontsize=14)
    ax[2, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[2, 0].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 0].tick_params(axis='y', labelsize=14)    
    ax[2, 0].tick_params(axis='x', labelsize=14)
    ax[2, 0].legend(fontsize=14) #, loc='upper right')

    # Hip moment
    ax[0, 1].plot(x_mom, ref_hip_moment_r_nat, label='Nat. Ref', color='orange')
    # ax[0, 1].plot(x_mom, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[0, 1].fill_between(x_mom, ref_hip_moment_r_nat - std_hip_moment_r_nat, ref_hip_moment_r_nat + std_hip_moment_r_nat, color='orange', alpha=0.2)
    # ax[0, 1].fill_between(x_mom, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2)
    ax[0, 1].plot(x_mom, sim_hip_moment_r_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[0, 1].plot(x_mom, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0, 1].set_title('Hip Moment', fontsize=14)
    ax[0, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[0, 1].tick_params(axis='y', labelsize=14)    
    ax[0, 1].tick_params(axis='x', labelsize=14)
    ax[0, 1].legend(fontsize=14) #, loc='upper right')

    # Knee moment
    ax[1, 1].plot(x_mom, ref_knee_moment_r_nat, label='Nat. Ref', color='orange')
    # ax[1, 1].plot(x_mom, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[1, 1].fill_between(x_mom, ref_knee_moment_r_nat - std_knee_moment_r_nat, ref_knee_moment_r_nat + std_knee_moment_r_nat, color='orange', alpha=0.2)
    # ax[1, 1].fill_between(x_mom, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2)
    ax[1, 1].plot(x_mom, sim_knee_moment_r_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1, 1].plot(x_mom, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 1].set_title('Knee Moment', fontsize=14)
    ax[1, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[1, 1].tick_params(axis='y', labelsize=14)    
    ax[1, 1].tick_params(axis='x', labelsize=14)
    ax[1, 1].legend(fontsize=14) #, loc='lower right')

    # Ankle moment
    ax[2, 1].plot(x_mom, ref_ankle_moment_r_nat, label='Nat. Ref', color='orange')
    # ax[2, 1].plot(x_mom, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[2, 1].fill_between(x_mom, ref_ankle_moment_r_nat - std_ankle_moment_r_nat, ref_ankle_moment_r_nat + std_ankle_moment_r_nat, color='orange', alpha=0.2)
    # ax[2, 1].fill_between(x_mom, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2)
    ax[2, 1].plot(x_mom, sim_ankle_moment_r_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2, 1].plot(x_mom, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 1].set_title('Ankle Moment', fontsize=14)
    ax[2, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[2, 1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 1].tick_params(axis='y', labelsize=14)    
    ax[2, 1].tick_params(axis='x', labelsize=14)
    ax[2, 1].legend(fontsize=14) #, loc='lower right')

    # # Pelvis ty
    # # ax[0, 2].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # # ax[0, 2].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    # ax[0, 2].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2)
    # # ax[0, 2].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2)
    # ax[0, 2].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[0, 2].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[0, 2].set_title('Pelvis Vertical Translation', fontsize=14)
    # ax[0, 2].set_ylabel('Translation (m/height)', fontsize=14)
    # ax[0, 2].tick_params(axis='y', labelsize=14)    
    # ax[0, 2].tick_params(axis='x', labelsize=14)
    # ax[0, 2].legend(fontsize=14) #, loc='upper right')

    # GRF y
    # Fill between for stdgrfnat and stdgrfexo for y GRF
    ax[1, 2].plot(x_grf, ref_grf_y_nat, label='Nat. Ref', color='orange')
    ax[1, 2].fill_between(x_grf, ref_grf_y_nat - std_grf_y_nat, ref_grf_y_nat + std_grf_y_nat, color='orange', alpha=0.2)
    # ax[1, 2].fill_between(x_grf, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2)
    # ax[1, 2].plot(x_grf, ref_grf_y_exo, label='Exo Ref', color='purple')
    ax[1, 2].plot(x_grf, sim_grf_y_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[1, 2].plot(x_grf, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 2].set_title('Vertical GRF', fontsize=14)
    ax[1, 2].set_ylabel('Force (BW)', fontsize=14)
    ax[1, 2].tick_params(axis='y', labelsize=14)    
    ax[1, 2].tick_params(axis='x', labelsize=14)
    ax[1, 2].legend(fontsize=14) #, loc='upper right')

    # GRF x
    ax[2, 2].plot(x_grf, ref_grf_x_nat, label='Nat. Ref', color='orange')
    ax[2, 2].fill_between(x_grf, ref_grf_x_nat - std_grf_x_nat, ref_grf_x_nat + std_grf_x_nat, color='orange', alpha=0.2)
    # ax[2, 2].fill_between(x_grf, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2)
    # ax[2, 2].plot(x_grf, ref_grf_x_exo, label='Exo Ref', color='purple')
    ax[2, 2].plot(x_grf, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2, 2].plot(x_grf, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 2].set_title('Horizontal GRF', fontsize=14)
    ax[2, 2].set_ylabel('Force (BW)', fontsize=14)
    ax[2, 2].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 2].tick_params(axis='y', labelsize=14)    
    ax[2, 2].tick_params(axis='x', labelsize=14)
    ax[2, 2].legend(fontsize=14) #, loc='upper right')

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_27.png')
    # plt.show()


    ##########################################################################
    # second is the natural validation figure in 3x3, but with both legs
    ##########################################################################
    ## now create the figure that we want. It should be a 3x3 grid of subplots. 
    # the first column should be hip knee and ankle angles
    # second column should be hip knee and ankle moments
    # third column should be the pelvis ty, GRF y, and GRF x
    fig, ax = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_mom = np.linspace(0, 100, moment_simlen_nat)[0:int(moment_simlen_nat/2)]
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    # Hip angle
    ax[0, 0].plot(x_kin, ref_hip_angle_r_nat, label='Nat. Ref - Right', color='orange')
    ax[0, 0].plot(x_kin, ref_hip_angle_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[0, 0].plot(x_kin, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[0, 0].fill_between(x_kin, ref_hip_angle_r_nat - std_hip_angle_r_nat, ref_hip_angle_r_nat + std_hip_angle_r_nat, color='orange', alpha=0.2)
    ax[0, 0].fill_between(x_kin, ref_hip_angle_l_nat - std_hip_angle_l_nat, ref_hip_angle_l_nat + std_hip_angle_l_nat, color='purple', alpha=0.2)
    # ax[0, 0].fill_between(x_kin, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0, 0].plot(x_kin, sim_hip_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[0, 0].plot(x_kin, sim_hip_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[0, 0].plot(x_kin, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0, 0].set_title('Hip Angle', fontsize=14)
    ax[0, 0].tick_params(axis='y', labelsize=14)    
    ax[0, 0].tick_params(axis='x', labelsize=14)
    ax[0, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[0, 0].legend(fontsize=14) #, loc='lower right')

    # Knee angle
    ax[1, 0].plot(x_kin, ref_knee_angle_r_nat, label='Nat. Ref - Right', color='orange')
    ax[1, 0].plot(x_kin, ref_knee_angle_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[1, 0].plot(x_kin, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[1, 0].fill_between(x_kin, ref_knee_angle_r_nat - std_knee_angle_r_nat, ref_knee_angle_r_nat + std_knee_angle_r_nat, color='orange', alpha=0.2)
    ax[1, 0].fill_between(x_kin, ref_knee_angle_l_nat - std_knee_angle_l_nat, ref_knee_angle_l_nat + std_knee_angle_l_nat, color='purple', alpha=0.2)
    # ax[1, 0].fill_between(x_kin, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[1, 0].plot(x_kin, sim_knee_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[1, 0].plot(x_kin, sim_knee_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[1, 0].plot(x_kin, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 0].set_title('Knee Angle', fontsize=14)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[1, 0].tick_params(axis='y', labelsize=14)    
    ax[1, 0].tick_params(axis='x', labelsize=14)
    ax[1, 0].legend(fontsize=14) #, loc='upper left')

    # Ankle angle
    ax[2, 0].plot(x_kin, ref_ankle_angle_r_nat, label='Nat. Ref - Right', color='orange')
    ax[2, 0].plot(x_kin, ref_ankle_angle_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[2, 0].plot(x_kin, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[2, 0].fill_between(x_kin, ref_ankle_angle_r_nat - std_ankle_angle_r_nat, ref_ankle_angle_r_nat + std_ankle_angle_r_nat, color='orange', alpha=0.2)
    ax[2, 0].fill_between(x_kin, ref_ankle_angle_l_nat - std_ankle_angle_l_nat, ref_ankle_angle_l_nat + std_ankle_angle_l_nat, color='purple', alpha=0.2)
    # ax[2, 0].fill_between(x_kin, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[2, 0].plot(x_kin, sim_ankle_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[2, 0].plot(x_kin, sim_ankle_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[2, 0].plot(x_kin, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 0].set_title('Ankle Angle', fontsize=14)
    ax[2, 0].set_ylabel('Angle (deg)', fontsize=14)
    ax[2, 0].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 0].tick_params(axis='y', labelsize=14)    
    ax[2, 0].tick_params(axis='x', labelsize=14)
    ax[2, 0].legend(fontsize=14) #, loc='upper right')

    # Hip moment
    ax[0, 1].plot(x_mom, ref_hip_moment_r_nat, label='Nat. Ref - Right', color='orange')
    ax[0, 1].plot(x_mom, ref_hip_moment_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[0, 1].plot(x_mom, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[0, 1].fill_between(x_mom, ref_hip_moment_r_nat - std_hip_moment_r_nat, ref_hip_moment_r_nat + std_hip_moment_r_nat, color='orange', alpha=0.2)
    ax[0, 1].fill_between(x_mom, ref_hip_moment_l_nat - std_hip_moment_l_nat, ref_hip_moment_l_nat + std_hip_moment_l_nat, color='purple', alpha=0.2)
    # ax[0, 1].fill_between(x_mom, ref_hip_moment_exo - 2*stdmomexo['hip_flexion_r_moment'], ref_hip_moment_exo + 2*stdmomexo['hip_flexion_r_moment'], color='purple', alpha=0.2)
    ax[0, 1].plot(x_mom, sim_hip_moment_r_nat, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[0, 1].plot(x_mom, sim_hip_moment_l_nat, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[0, 1].plot(x_mom, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0, 1].set_title('Hip Moment', fontsize=14)
    ax[0, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[0, 1].tick_params(axis='y', labelsize=14)    
    ax[0, 1].tick_params(axis='x', labelsize=14)
    ax[0, 1].legend(fontsize=14) #, loc='upper right')

    # Knee moment
    ax[1, 1].plot(x_mom, ref_knee_moment_r_nat, label='Nat. Ref - Right', color='orange')
    ax[1, 1].plot(x_mom, ref_knee_moment_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[1, 1].plot(x_mom, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[1, 1].fill_between(x_mom, ref_knee_moment_r_nat - std_knee_moment_r_nat, ref_knee_moment_r_nat + std_knee_moment_r_nat, color='orange', alpha=0.2)
    ax[1, 1].fill_between(x_mom, ref_knee_moment_l_nat - std_knee_moment_l_nat, ref_knee_moment_l_nat + std_knee_moment_l_nat, color='purple', alpha=0.2)
    # ax[1, 1].fill_between(x_mom, ref_knee_moment_exo - 2*stdmomexo['knee_angle_r_moment'], ref_knee_moment_exo + 2*stdmomexo['knee_angle_r_moment'], color='purple', alpha=0.2)
    ax[1, 1].plot(x_mom, sim_knee_moment_r_nat, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[1, 1].plot(x_mom, sim_knee_moment_l_nat, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[1, 1].plot(x_mom, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 1].set_title('Knee Moment', fontsize=14)
    ax[1, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[1, 1].tick_params(axis='y', labelsize=14)    
    ax[1, 1].tick_params(axis='x', labelsize=14)
    ax[1, 1].legend(fontsize=14) #, loc='lower right')

    # Ankle moment
    ax[2, 1].plot(x_mom, ref_ankle_moment_r_nat, label='Nat. Ref - Right', color='orange')
    ax[2, 1].plot(x_mom, ref_ankle_moment_l_nat, label='Nat. Ref - Left', color='purple')
    # ax[2, 1].plot(x_mom, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[2, 1].fill_between(x_mom, ref_ankle_moment_r_nat - std_ankle_moment_r_nat, ref_ankle_moment_r_nat + std_ankle_moment_r_nat, color='orange', alpha=0.2)
    ax[2, 1].fill_between(x_mom, ref_ankle_moment_l_nat - std_ankle_moment_l_nat, ref_ankle_moment_l_nat + std_ankle_moment_l_nat, color='purple', alpha=0.2)
    # ax[2, 1].fill_between(x_mom, ref_ankle_moment_exo - 2*stdmomexo['ankle_angle_r_moment'], ref_ankle_moment_exo + 2*stdmomexo['ankle_angle_r_moment'], color='purple', alpha=0.2)
    ax[2, 1].plot(x_mom, sim_ankle_moment_r_nat, label='Nat. Sim - Right', color='orange', linestyle='--')
    ax[2, 1].plot(x_mom, sim_ankle_moment_l_nat, label='Nat. Sim - Left', color='purple', linestyle='--')
    # ax[2, 1].plot(x_mom, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 1].set_title('Ankle Moment', fontsize=14)
    ax[2, 1].set_ylabel('Moment (Nm)', fontsize=14)
    ax[2, 1].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 1].tick_params(axis='y', labelsize=14)    
    ax[2, 1].tick_params(axis='x', labelsize=14)
    ax[2, 1].legend(fontsize=14) #, loc='lower right')

    # # Pelvis ty
    # # ax[0, 2].plot(x, ref_ty_nat, label='Nat. Ref', color='orange')
    # # ax[0, 2].plot(x, ref_ty_exo, label='Exo Ref', color='purple')
    # ax[0, 2].fill_between(x, ref_ty_nat - 2*std_nat['pelvis_ty'], ref_ty_nat + 2*std_nat['pelvis_ty'], color='orange', alpha=0.2)
    # # ax[0, 2].fill_between(x, ref_ty_exo - 2*std_exo['pelvis_ty'], ref_ty_exo + 2*std_exo['pelvis_ty'], color='purple', alpha=0.2)
    # ax[0, 2].plot(x, sim_ty_nat, label='Nat. Sim', color='orange', linestyle='--')
    # # ax[0, 2].plot(x, sim_ty_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[0, 2].set_title('Pelvis Vertical Translation', fontsize=14)
    # ax[0, 2].set_ylabel('Translation (m/height)', fontsize=14)
    # ax[0, 2].tick_params(axis='y', labelsize=14)    
    # ax[0, 2].tick_params(axis='x', labelsize=14)
    # ax[0, 2].legend(fontsize=14) #, loc='upper right')

    # GRF y
    # Fill between for stdgrfnat and stdgrfexo for y GRF
    ax[1, 2].plot(x_grf, ref_grf_y_nat, label='Nat. Ref - Right', color='orange')
    ax[1, 2].fill_between(x_grf, ref_grf_y_nat - std_grf_y_nat, ref_grf_y_nat + std_grf_y_nat, color='orange', alpha=0.2)
    # ax[1, 2].fill_between(x_grf, ref_grf_y_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fy'], ref_grf_y_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fy'], color='purple', alpha=0.2)
    # ax[1, 2].plot(x_grf, ref_grf_y_exo, label='Exo Ref', color='purple')
    ax[1, 2].plot(x_grf, sim_grf_y_nat, label='Nat. Sim - Right', color='orange', linestyle='--')
    # ax[1, 2].plot(x_grf, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[1, 2].set_title('Vertical GRF', fontsize=14)
    ax[1, 2].set_ylabel('Force (BW)', fontsize=14)
    ax[1, 2].tick_params(axis='y', labelsize=14)    
    ax[1, 2].tick_params(axis='x', labelsize=14)
    ax[1, 2].legend(fontsize=14) #, loc='upper right')

    # GRF x
    ax[2, 2].plot(x_grf, ref_grf_x_nat, label='Nat. Ref - Right', color='orange')
    ax[2, 2].fill_between(x_grf, ref_grf_x_nat - std_grf_x_nat, ref_grf_x_nat + std_grf_x_nat, color='orange', alpha=0.2)
    # ax[2, 2].fill_between(x_grf, ref_grf_x_exo - 2*stdgrfexo['calcn_r_Right_GRF_Fx'], ref_grf_x_exo + 2*stdgrfexo['calcn_r_Right_GRF_Fx'], color='purple', alpha=0.2)
    # ax[2, 2].plot(x_grf, ref_grf_x_exo, label='Exo Ref', color='purple')
    ax[2, 2].plot(x_grf, sim_grf_x_nat, label='Nat. Sim', color='orange', linestyle='--')
    # ax[2, 2].plot(x_grf, sim_grf_x_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[2, 2].set_title('Horizontal GRF', fontsize=14)
    ax[2, 2].set_ylabel('Force (BW)', fontsize=14)
    ax[2, 2].set_xlabel('Gait Cycle %', fontsize=14)
    ax[2, 2].tick_params(axis='y', labelsize=14)    
    ax[2, 2].tick_params(axis='x', labelsize=14)
    ax[2, 2].legend(fontsize=14) #, loc='upper right')

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_27.png')
    plt.show()



    return

# create a function for plotting a nice figure for use as a paper validation figure. Focus on saggital metrics. 
def saggitalValidationSplit27(simNat, simExo, iknat2D, ikexo2D, labels2D, coordinates_sim_clean, mean_nat, std_nat, mean_exo, std_exo, GRFsimnat, GRFsimexo, GRFrefnat, GRFrefexo, meangrfnat, stdgrfnat, meangrfexo, stdgrfexo, natmomentfile, exomomentfile, idnat, idexo, meanmomnat, stdmomnat, meanmomexo, stdmomexo, modelfile):
    # load the model and get the mass
    model = osim.Model(modelfile)
    mass = model.getTotalMass(model.initSystem())
    height = 1.78
    ## starting with the kinematics
    # get the length of the simulation data
    kin_simlen_nat = len(simNat.getIndependentColumn())
    kin_simlen_exo = len(simExo.getIndependentColumn())
    if kin_simlen_nat != kin_simlen_exo:
        print('Simulation kinematic data lengths do not match. Exiting.')
        return
    # get the sim kinematics
    sim_hip_angle_r_nat = simNat.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_r_nat = simNat.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_r_nat = simNat.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()    
    sim_hip_angle_l_nat = simNat.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_l_nat = simNat.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_l_nat = simNat.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ty_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy() / height
    sim_tx_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tx/value').to_numpy() / height
    sim_pelvtilt_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvlist_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvrot_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbarRot_nat = simNat.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_lumbarBend_nat = simNat.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbarExt_nat = simNat.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()

    sim_hip_angle_r_exo = simExo.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_r_exo = simExo.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_r_exo = simExo.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_hip_angle_l_exo = simExo.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_l_exo = simExo.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_l_exo = simExo.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ty_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy() / height
    sim_tx_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_tx/value').to_numpy() / height
    sim_pelvtilt_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvlist_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvrot_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbarRot_exo = simExo.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_lumbarBend_exo = simExo.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbarExt_exo = simExo.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()    

    # # get the reference data for the kinematics. 
    # ref_hip_angle_r_nat = iknat2D.getDependentColumn('hip_flexion_r').to_numpy()
    # ref_knee_angle_r_nat = iknat2D.getDependentColumn('knee_angle_r').to_numpy()
    # ref_ankle_angle_r_nat = iknat2D.getDependentColumn('ankle_angle_r').to_numpy()
    # ref_ty_nat = iknat2D.getDependentColumn('pelvis_ty').to_numpy()
    # ref_tx_nat = iknat2D.getDependentColumn('pelvis_tx').to_numpy()
    # ref_pelvtilt_nat = iknat2D.getDependentColumn('pelvis_tilt').to_numpy()
    # ref_pelvlist_nat = iknat2D.getDependentColumn('pelvis_list').to_numpy()
    # ref_pelvrot_nat = iknat2D.getDependentColumn('pelvis_rotation').to_numpy()
    # ref_lumbarRot_nat = iknat2D.getDependentColumn('lumbar_rotation').to_numpy()
    # ref_lumbarBend_nat = iknat2D.getDependentColumn('lumbar_bending').to_numpy()
    # ref_lumbarExt_nat = iknat2D.getDependentColumn('lumbar_extension').to_numpy()
    # ref_hip_angle_l_nat = iknat2D.getDependentColumn('hip_flexion_l').to_numpy()
    # ref_knee_angle_l_nat = iknat2D.getDependentColumn('knee_angle_l').to_numpy()
    # ref_ankle_angle_l_nat = iknat2D.getDependentColumn('ankle_angle_l').to_numpy()

    # ref_hip_angle_r_exo = ikexo2D.getDependentColumn('hip_flexion_r').to_numpy()
    # ref_knee_angle_r_exo = ikexo2D.getDependentColumn('knee_angle_r').to_numpy()
    # ref_ankle_angle_r_exo = ikexo2D.getDependentColumn('ankle_angle_r').to_numpy()
    # ref_ty_exo = ikexo2D.getDependentColumn('pelvis_ty').to_numpy()
    # ref_tx_exo = ikexo2D.getDependentColumn('pelvis_tx').to_numpy()
    # ref_pelvtilt_exo = ikexo2D.getDependentColumn('pelvis_tilt').to_numpy()
    # ref_pelvlist_exo = ikexo2D.getDependentColumn('pelvis_list').to_numpy()
    # ref_pelvrot_exo = ikexo2D.getDependentColumn('pelvis_rotation').to_numpy()
    # ref_lumbarRot_exo = ikexo2D.getDependentColumn('lumbar_rotation').to_numpy()
    # ref_lumbarBend_exo = ikexo2D.getDependentColumn('lumbar_bending').to_numpy()
    # ref_lumbarExt_exo = ikexo2D.getDependentColumn('lumbar_extension').to_numpy()
    # ref_hip_angle_l_exo = ikexo2D.getDependentColumn('hip_flexion_l').to_numpy()
    # ref_knee_angle_l_exo = ikexo2D.getDependentColumn('knee_angle_l').to_numpy()
    # ref_ankle_angle_l_exo = ikexo2D.getDependentColumn('ankle_angle_l').to_numpy()

    # second option for the reference data - actual mean data from experiments. 
    ref_hip_angle_r_nat = mean_nat['hip_flexion_r']
    ref_knee_angle_r_nat = mean_nat['knee_angle_r']
    ref_ankle_angle_r_nat = mean_nat['ankle_angle_r']
    ref_ty_nat = mean_nat['pelvis_ty']
    ref_tx_nat = mean_nat['pelvis_tx']
    ref_pelvtilt_nat = mean_nat['pelvis_tilt']
    ref_pelvlist_nat = mean_nat['pelvis_list']
    ref_pelvrot_nat = mean_nat['pelvis_rotation']
    ref_lumbarRot_nat = mean_nat['lumbar_rotation']
    ref_lumbarBend_nat = mean_nat['lumbar_bending']
    ref_lumbarExt_nat = mean_nat['lumbar_extension']
    ref_hip_angle_l_nat = mean_nat['hip_flexion_l']
    ref_knee_angle_l_nat = mean_nat['knee_angle_l']
    ref_ankle_angle_l_nat = mean_nat['ankle_angle_l']

    ref_hip_angle_r_exo = mean_exo['hip_flexion_r']
    ref_knee_angle_r_exo = mean_exo['knee_angle_r']
    ref_ankle_angle_r_exo = mean_exo['ankle_angle_r']
    ref_ty_exo = mean_exo['pelvis_ty']
    ref_tx_exo = mean_exo['pelvis_tx'] 
    ref_pelvtilt_exo = mean_exo['pelvis_tilt']
    ref_pelvlist_exo = mean_exo['pelvis_list']
    ref_pelvrot_exo = mean_exo['pelvis_rotation']
    ref_lumbarRot_exo = mean_exo['lumbar_rotation']
    ref_lumbarBend_exo = mean_exo['lumbar_bending']
    ref_lumbarExt_exo = mean_exo['lumbar_extension']
    ref_hip_angle_l_exo = mean_exo['hip_flexion_l']
    ref_knee_angle_l_exo = mean_exo['knee_angle_l']
    ref_ankle_angle_l_exo = mean_exo['ankle_angle_l']


    # get the length of the reference data
    kin_reflen_nat = len(ref_hip_angle_r_nat)
    kin_reflen_exo = len(ref_hip_angle_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation. 
    kin_xsim = np.linspace(0,100,kin_simlen_nat)
    kin_xref_nat = np.linspace(0,100,kin_reflen_nat)
    kin_xref_exo = np.linspace(0,100,kin_reflen_exo)
    
    # interpolate the reference data to the simulation data length
    ref_hip_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_r_nat)
    ref_knee_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_r_nat)
    ref_ankle_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_r_nat)
    ref_ty_nat = np.interp(kin_xsim, kin_xref_nat, ref_ty_nat)
    ref_tx_nat = np.interp(kin_xsim, kin_xref_nat, ref_tx_nat)
    ref_pelvtilt_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvtilt_nat)
    ref_pelvlist_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvlist_nat)
    ref_pelvrot_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvrot_nat)
    ref_lumbarRot_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarRot_nat)
    ref_lumbarBend_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarBend_nat)
    ref_lumbarExt_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarExt_nat)
    ref_hip_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_l_nat)
    ref_knee_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_l_nat)
    ref_ankle_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_l_nat)
    #
    ref_hip_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_r_exo)
    ref_knee_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_r_exo)
    ref_ankle_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_r_exo)
    ref_ty_exo = np.interp(kin_xsim, kin_xref_exo, ref_ty_exo)
    ref_tx_exo = np.interp(kin_xsim, kin_xref_exo, ref_tx_exo)
    ref_pelvtilt_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvtilt_exo)
    ref_pelvlist_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvlist_exo)
    ref_pelvrot_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvrot_exo)
    ref_lumbarRot_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarRot_exo)
    ref_lumbarBend_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarBend_exo)
    ref_lumbarExt_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarExt_exo)
    ref_hip_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_l_exo)
    ref_knee_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_l_exo)
    ref_ankle_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_l_exo)
    
    # # normalize the ty values to the height of the subject
    # ref_ty_nat = ref_ty_nat / height
    # ref_ty_exo = ref_ty_exo / height
    # sim_ty_nat = sim_ty_nat / height
    # sim_ty_exo = sim_ty_exo / height
    
    # now shorten to be one step, rather than the full gait cycle. 
    ref_hip_angle_r_nat = ref_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_r_nat = ref_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_r_nat = ref_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_hip_angle_l_nat = ref_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_l_nat = ref_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_l_nat = ref_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_ty_nat = ref_ty_nat[0:int(kin_simlen_nat/2)]
    ref_tx_nat = ref_tx_nat[0:int(kin_simlen_nat/2)]
    ref_pelvtilt_nat = ref_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    ref_pelvlist_nat = ref_pelvlist_nat[0:int(kin_simlen_nat/2)]
    ref_pelvrot_nat = ref_pelvrot_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarRot_nat = ref_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarBend_nat = ref_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarExt_nat = ref_lumbarExt_nat[0:int(kin_simlen_nat/2)]

    ref_hip_angle_r_exo = ref_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_knee_angle_r_exo = ref_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_r_exo = ref_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    ref_hip_angle_l_exo = ref_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    ref_knee_angle_l_exo = ref_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_l_exo = ref_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    ref_ty_exo = ref_ty_exo[0:int(kin_simlen_nat/2)]
    ref_tx_exo = ref_tx_exo[0:int(kin_simlen_nat/2)]
    ref_pelvtilt_exo = ref_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    ref_pelvlist_exo = ref_pelvlist_exo[0:int(kin_simlen_nat/2)]
    ref_pelvrot_exo = ref_pelvrot_exo[0:int(kin_simlen_nat/2)]
    ref_lumbarRot_exo = ref_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    ref_lumbarBend_exo = ref_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    ref_lumbarExt_exo = ref_lumbarExt_exo[0:int(kin_simlen_nat/2)]

    sim_hip_angle_r_nat = sim_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_nat = sim_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_nat = sim_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_nat = sim_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_nat = sim_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_nat = sim_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_ty_nat = sim_ty_nat[0:int(kin_simlen_nat/2)]
    sim_tx_nat = sim_tx_nat[0:int(kin_simlen_nat/2)]
    sim_pelvtilt_nat = sim_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    sim_pelvlist_nat = sim_pelvlist_nat[0:int(kin_simlen_nat/2)]
    sim_pelvrot_nat = sim_pelvrot_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarRot_nat = sim_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarBend_nat = sim_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarExt_nat = sim_lumbarExt_nat[0:int(kin_simlen_nat/2)]
    
    sim_hip_angle_r_exo = sim_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_exo = sim_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_exo = sim_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_exo = sim_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_exo = sim_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_exo = sim_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_ty_exo = sim_ty_exo[0:int(kin_simlen_nat/2)]
    sim_tx_exo = sim_tx_exo[0:int(kin_simlen_nat/2)]
    sim_pelvtilt_exo = sim_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    sim_pelvlist_exo = sim_pelvlist_exo[0:int(kin_simlen_nat/2)]
    sim_pelvrot_exo = sim_pelvrot_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarRot_exo = sim_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarBend_exo = sim_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarExt_exo = sim_lumbarExt_exo[0:int(kin_simlen_nat/2)]

    # now get the ref std for plus and minus
    std_hip_angle_r_nat = 2*std_nat['hip_flexion_r']
    std_knee_angle_r_nat = 2*std_nat['knee_angle_r']
    std_ankle_angle_r_nat = 2*std_nat['ankle_angle_r']
    std_hip_angle_l_nat = 2*std_nat['hip_flexion_l']
    std_knee_angle_l_nat = 2*std_nat['knee_angle_l']
    std_ankle_angle_l_nat = 2*std_nat['ankle_angle_l']
    std_ty_nat = 2*std_nat['pelvis_ty']
    std_tx_nat = 2*std_nat['pelvis_tx']
    std_pelvtilt_nat = 2*std_nat['pelvis_tilt']
    std_pelvlist_nat = 2*std_nat['pelvis_list']
    std_pelvrot_nat = 2*std_nat['pelvis_rotation']
    std_lumbarRot_nat = 2*std_nat['lumbar_rotation']
    std_lumbarBend_nat = 2*std_nat['lumbar_bending']
    std_lumbarExt_nat = 2*std_nat['lumbar_extension']

    std_hip_angle_r_exo = 2*std_exo['hip_flexion_r']
    std_knee_angle_r_exo = 2*std_exo['knee_angle_r']
    std_ankle_angle_r_exo = 2*std_exo['ankle_angle_r']
    std_hip_angle_l_exo = 2*std_exo['hip_flexion_l']
    std_knee_angle_l_exo = 2*std_exo['knee_angle_l']
    std_ankle_angle_l_exo = 2*std_exo['ankle_angle_l']
    std_ty_exo = 2*std_exo['pelvis_ty']
    std_tx_exo = 2*std_exo['pelvis_tx']
    std_pelvtilt_exo = 2*std_exo['pelvis_tilt']
    std_pelvlist_exo = 2*std_exo['pelvis_list']
    std_pelvrot_exo = 2*std_exo['pelvis_rotation']
    std_lumbarRot_exo = 2*std_exo['lumbar_rotation']
    std_lumbarBend_exo = 2*std_exo['lumbar_bending']
    std_lumbarExt_exo = 2*std_exo['lumbar_extension']

    # now resample
    std_hip_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_nat)), std_hip_angle_r_nat)
    std_knee_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_nat)), std_knee_angle_r_nat)
    std_ankle_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_nat)), std_ankle_angle_r_nat)
    std_hip_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_hip_angle_l_nat)
    std_knee_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_nat)), std_knee_angle_l_nat)
    std_ankle_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_ankle_angle_l_nat)
    std_ty_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ty_nat)), std_ty_nat)
    std_tx_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_tx_nat)), std_tx_nat)
    std_pelvtilt_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvtilt_nat)), std_pelvtilt_nat)
    std_pelvlist_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvlist_nat)), std_pelvlist_nat)
    std_pelvrot_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvrot_nat)), std_pelvrot_nat)
    std_lumbarRot_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarRot_nat)), std_lumbarRot_nat)
    std_lumbarBend_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarBend_nat)), std_lumbarBend_nat)
    std_lumbarExt_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarExt_nat)), std_lumbarExt_nat)
    
    std_hip_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_exo)), std_hip_angle_r_exo)
    std_knee_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_exo)), std_knee_angle_r_exo)
    std_ankle_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_exo)), std_ankle_angle_r_exo)
    std_hip_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_l_exo)), std_hip_angle_l_exo)
    std_knee_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_exo)), std_knee_angle_l_exo)
    std_ankle_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_exo)), std_ankle_angle_l_exo)
    std_ty_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ty_exo)), std_ty_exo)
    std_tx_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_tx_exo)), std_tx_exo)
    std_pelvtilt_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvtilt_exo)), std_pelvtilt_exo)
    std_pelvlist_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvlist_exo)), std_pelvlist_exo)
    std_pelvrot_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvrot_exo)), std_pelvrot_exo)
    std_lumbarRot_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarRot_exo)), std_lumbarRot_exo)
    std_lumbarBend_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarBend_exo)), std_lumbarBend_exo)
    std_lumbarExt_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarExt_exo)), std_lumbarExt_exo)

    
    # and cut down to a step rather than the full gait cycle
    std_hip_angle_r_nat = std_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_r_nat = std_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_r_nat = std_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_hip_angle_l_nat = std_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_l_nat = std_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_l_nat = std_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_ty_nat = std_ty_nat[0:int(kin_simlen_nat/2)]
    std_tx_nat = std_tx_nat[0:int(kin_simlen_nat/2)]
    std_pelvtilt_nat = std_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    std_pelvlist_nat = std_pelvlist_nat[0:int(kin_simlen_nat/2)]
    std_pelvrot_nat = std_pelvrot_nat[0:int(kin_simlen_nat/2)]
    std_lumbarRot_nat = std_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    std_lumbarBend_nat = std_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    std_lumbarExt_nat = std_lumbarExt_nat[0:int(kin_simlen_nat/2)]

    std_hip_angle_r_exo = std_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_knee_angle_r_exo = std_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_ankle_angle_r_exo = std_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    std_hip_angle_l_exo = std_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    std_knee_angle_l_exo = std_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    std_ankle_angle_l_exo = std_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    std_ty_exo = std_ty_exo[0:int(kin_simlen_nat/2)]
    std_tx_exo = std_tx_exo[0:int(kin_simlen_nat/2)]
    std_pelvtilt_exo = std_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    std_pelvlist_exo = std_pelvlist_exo[0:int(kin_simlen_nat/2)]
    std_pelvrot_exo = std_pelvrot_exo[0:int(kin_simlen_nat/2)]
    std_lumbarRot_exo = std_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    std_lumbarBend_exo = std_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    std_lumbarExt_exo = std_lumbarExt_exo[0:int(kin_simlen_nat/2)]


    ## get the GRF data
    grfsimnat = osim.TimeSeriesTable(GRFsimnat)
    grfsimexo = osim.TimeSeriesTable(GRFsimexo)
    grfrefnat = osim.TimeSeriesTable(GRFrefnat)
    grfrefexo = osim.TimeSeriesTable(GRFrefexo)
    # get the length of the simulation data
    grf_simlen_nat = len(grfsimnat.getIndependentColumn())
    grf_simlen_exo = len(grfsimexo.getIndependentColumn())
    if grf_simlen_nat != grf_simlen_exo:
        print('Simulation GRF data lengths do not match. Exiting.')
        return
    # get the sim GRF data
    sim_grf_y_nat = grfsimnat.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_y_exo = grfsimexo.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_x_nat = grfsimnat.getDependentColumn('ground_force_r_vx').to_numpy()
    sim_grf_x_exo = grfsimexo.getDependentColumn('ground_force_r_vx').to_numpy()
    
    # get the reference GRF data
    # ref_grf_y_nat = grfrefnat.getDependentColumn('rF_y').to_numpy()
    # ref_grf_y_exo = grfrefexo.getDependentColumn('rF_y').to_numpy()
    # ref_grf_x_nat = grfrefnat.getDependentColumn('rF_x').to_numpy()
    # ref_grf_x_exo = grfrefexo.getDependentColumn('rF_x').to_numpy()
    # another option for reference data from the actual means of the experimental values. 
    ref_grf_y_nat = meangrfnat['calcn_r_Right_GRF_Fy']
    ref_grf_y_exo = meangrfexo['calcn_r_Right_GRF_Fy']
    ref_grf_x_nat = meangrfnat['calcn_r_Right_GRF_Fx']
    ref_grf_x_exo = meangrfexo['calcn_r_Right_GRF_Fx']
    # get the length of the reference data
    grf_reflen_nat = len(ref_grf_y_nat)
    grf_reflen_exo = len(ref_grf_y_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    grf_xsim = np.linspace(0,100,grf_simlen_nat)
    grf_xref_nat = np.linspace(0,100,grf_reflen_nat)
    grf_xref_exo = np.linspace(0,100,grf_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_grf_y_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_y_nat)
    ref_grf_y_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_y_exo)
    ref_grf_x_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_x_nat)
    ref_grf_x_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_x_exo)
    # divide all of the GRF data based on the mass of the model
    sim_grf_y_nat = sim_grf_y_nat/(mass*9.81)
    sim_grf_y_exo = sim_grf_y_exo/(mass*9.81)
    sim_grf_x_nat = sim_grf_x_nat/(mass*9.81)
    sim_grf_x_exo = sim_grf_x_exo/(mass*9.81)
    ref_grf_y_nat = ref_grf_y_nat#/(mass*9.81)
    ref_grf_y_exo = ref_grf_y_exo#/(mass*9.81)
    ref_grf_x_nat = ref_grf_x_nat#/(mass*9.81)
    ref_grf_x_exo = ref_grf_x_exo#/(mass*9.81)

    # now shorten to be one step rather than the full gait cycle. 
    ref_grf_y_nat = ref_grf_y_nat[0:int(grf_simlen_nat/2)]
    ref_grf_y_exo = ref_grf_y_exo[0:int(grf_simlen_nat/2)]
    ref_grf_x_nat = ref_grf_x_nat[0:int(grf_simlen_nat/2)]
    ref_grf_x_exo = ref_grf_x_exo[0:int(grf_simlen_nat/2)]
    
    sim_grf_y_nat = sim_grf_y_nat[0:int(grf_simlen_nat/2)]
    sim_grf_y_exo = sim_grf_y_exo[0:int(grf_simlen_nat/2)]
    sim_grf_x_nat = sim_grf_x_nat[0:int(grf_simlen_nat/2)]
    sim_grf_x_exo = sim_grf_x_exo[0:int(grf_simlen_nat/2)]

    # get the std data for plus and minus
    std_grf_y_nat = 2*stdgrfnat['calcn_r_Right_GRF_Fy']
    std_grf_y_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fy']
    std_grf_x_nat = 2*stdgrfnat['calcn_r_Right_GRF_Fx']
    std_grf_x_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fx']
    # now resample
    std_grf_y_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_nat)), std_grf_y_nat)
    std_grf_y_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_exo)), std_grf_y_exo)
    std_grf_x_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_nat)), std_grf_x_nat)
    std_grf_x_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_exo)), std_grf_x_exo)
    # and cut down to a step rather than the full gait cycle
    std_grf_y_nat = std_grf_y_nat[0:int(grf_simlen_nat/2)]
    std_grf_y_exo = std_grf_y_exo[0:int(grf_simlen_nat/2)]
    std_grf_x_nat = std_grf_x_nat[0:int(grf_simlen_nat/2)]
    std_grf_x_exo = std_grf_x_exo[0:int(grf_simlen_nat/2)]

    
    ## get the moment data
    natmoment = osim.TimeSeriesTable(natmomentfile)
    exomoment = osim.TimeSeriesTable(exomomentfile)
    natrefmoment = osim.TimeSeriesTable(idnat)
    exorefmoment = osim.TimeSeriesTable(idexo)
    # get the length of the simulation data
    moment_simlen_nat = len(natmoment.getIndependentColumn())
    moment_simlen_exo = len(exomoment.getIndependentColumn())
    if moment_simlen_nat != moment_simlen_exo:
        print('Simulation moment data lengths do not match. Exiting.')
        return
    
    # get the sim moment data
    sim_hip_moment_r_nat = natmoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_hip_moment_r_exo = exomoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_knee_moment_r_nat = natmoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_knee_moment_r_exo = exomoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_ankle_moment_r_nat = natmoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_ankle_moment_r_exo = exomoment.getDependentColumn('ankle_angle_r_moment').to_numpy()

    sim_hip_moment_l_nat = natmoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_hip_moment_l_exo = exomoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_knee_moment_l_nat = natmoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_knee_moment_l_exo = exomoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_ankle_moment_l_nat = natmoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_ankle_moment_l_exo = exomoment.getDependentColumn('ankle_angle_l_moment').to_numpy()

    # get the reference moment data 
    # ref_hip_moment_r_nat = idnat.getDependentColumn('hip_flexion_r_moment').to_numpy()
    # ref_hip_moment_r_exo = idexo.getDependentColumn('hip_flexion_r_moment').to_numpy()
    # ref_knee_moment_r_nat = idnat.getDependentColumn('knee_angle_r_moment').to_numpy()
    # ref_knee_moment_r_exo = idexo.getDependentColumn('knee_angle_r_moment').to_numpy()
    # ref_ankle_moment_r_nat = idnat.getDependentColumn('ankle_angle_r_moment').to_numpy()
    # ref_ankle_moment_r_exo = idexo.getDependentColumn('ankle_angle_r_moment').to_numpy()
    # ref_hip_moment_l_nat = idnat.getDependentColumn('hip_flexion_l_moment').to_numpy()
    # ref_hip_moment_l_exo = idexo.getDependentColumn('hip_flexion_l_moment').to_numpy()
    # ref_knee_moment_l_nat = idnat.getDependentColumn('knee_angle_l_moment').to_numpy()
    # ref_knee_moment_l_exo = idexo.getDependentColumn('knee_angle_l_moment').to_numpy()
    # ref_ankle_moment_l_nat = idnat.getDependentColumn('ankle_angle_l_moment').to_numpy()
    # ref_ankle_moment_l_exo = idexo.getDependentColumn('ankle_angle_l_moment').to_numpy()
    # another option for reference moment data from experimental means. 
    ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    ref_hip_moment_r_exo = meanmomexo['hip_flexion_r_moment']
    ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    ref_knee_moment_r_exo = meanmomexo['knee_angle_r_moment']
    ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    ref_ankle_moment_r_exo = meanmomexo['ankle_angle_r_moment']
    ref_hip_moment_l_nat = meanmomnat['hip_flexion_l_moment']
    ref_hip_moment_l_exo = meanmomexo['hip_flexion_l_moment']
    ref_knee_moment_l_nat = meanmomnat['knee_angle_l_moment']
    ref_knee_moment_l_exo = meanmomexo['knee_angle_l_moment']
    ref_ankle_moment_l_nat = meanmomnat['ankle_angle_l_moment']
    ref_ankle_moment_l_exo = meanmomexo['ankle_angle_l_moment']
    
    # ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    # ref_hip_moment_r_exo = meanmomexo['hip_flexion_r_moment']
    # ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    # ref_knee_moment_r_exo = meanmomexo['knee_angle_r_moment']
    # ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    # ref_ankle_moment_r_exo = meanmomexo['ankle_angle_r_moment']

    # get the length of the reference data
    moment_reflen_nat = len(ref_hip_moment_r_nat)
    moment_reflen_exo = len(ref_hip_moment_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    moment_xsim = np.linspace(0,100,moment_simlen_nat)
    moment_xref_nat = np.linspace(0,100,moment_reflen_nat)
    moment_xref_exo = np.linspace(0,100,moment_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_hip_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_r_nat)
    ref_hip_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_r_exo)
    ref_knee_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_r_nat)
    ref_knee_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_r_exo)
    ref_ankle_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_r_nat)
    ref_ankle_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_r_exo)

    ref_hip_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_l_nat)
    ref_hip_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_l_exo)
    ref_knee_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_l_nat)
    ref_knee_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_l_exo)
    ref_ankle_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_l_nat)
    ref_ankle_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_l_exo)
    
    # normalize all of it to body mass
    sim_hip_moment_r_nat = sim_hip_moment_r_nat/mass    
    sim_hip_moment_r_exo = sim_hip_moment_r_exo/mass
    sim_knee_moment_r_nat = sim_knee_moment_r_nat/mass
    sim_knee_moment_r_exo = sim_knee_moment_r_exo/mass
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat/mass
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo/mass
    sim_hip_moment_l_nat = sim_hip_moment_l_nat/mass
    sim_hip_moment_l_exo = sim_hip_moment_l_exo/mass
    sim_knee_moment_l_nat = sim_knee_moment_l_nat/mass
    sim_knee_moment_l_exo = sim_knee_moment_l_exo/mass
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat/mass
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo/mass

    ref_hip_moment_r_nat = ref_hip_moment_r_nat#/mass
    ref_hip_moment_r_exo = ref_hip_moment_r_exo#/mass
    ref_knee_moment_r_nat = ref_knee_moment_r_nat#/mass
    ref_knee_moment_r_exo = ref_knee_moment_r_exo#/mass
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat#/mass
    ref_ankle_moment_r_exo = ref_ankle_moment_r_exo#/mass
    ref_hip_moment_l_nat = ref_hip_moment_l_nat#/mass
    ref_hip_moment_l_exo = ref_hip_moment_l_exo#/mass
    ref_knee_moment_l_nat = ref_knee_moment_l_nat#/mass
    ref_knee_moment_l_exo = ref_knee_moment_l_exo#/mass
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat#/mass
    ref_ankle_moment_l_exo = ref_ankle_moment_l_exo#/mass
    
    # and now shorten to the single step, rather than gait cycle. 
    ref_hip_moment_r_nat = ref_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_hip_moment_r_exo = ref_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_knee_moment_r_nat = ref_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_r_exo = ref_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_r_exo = ref_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    ref_hip_moment_l_nat = ref_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_hip_moment_l_exo = ref_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    ref_knee_moment_l_nat = ref_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_l_exo = ref_knee_moment_l_exo[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_l_exo = ref_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]

    sim_hip_moment_r_nat = sim_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_r_exo = sim_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_nat = sim_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_exo = sim_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_nat = sim_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_exo = sim_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_nat = sim_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_exo = sim_knee_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    
    # now get the std data for plus and minus
    std_hip_moment_r_nat = 2*stdmomnat['hip_flexion_r_moment']
    std_hip_moment_r_exo = 2*stdmomexo['hip_flexion_r_moment']
    std_knee_moment_r_nat = 2*stdmomnat['knee_angle_r_moment']
    std_knee_moment_r_exo = 2*stdmomexo['knee_angle_r_moment']
    std_ankle_moment_r_nat = 2*stdmomnat['ankle_angle_r_moment']
    std_ankle_moment_r_exo = 2*stdmomexo['ankle_angle_r_moment']
    std_hip_moment_l_nat = 2*stdmomnat['hip_flexion_l_moment']
    std_hip_moment_l_exo = 2*stdmomexo['hip_flexion_l_moment']
    std_knee_moment_l_nat = 2*stdmomnat['knee_angle_l_moment']
    std_knee_moment_l_exo = 2*stdmomexo['knee_angle_l_moment']
    std_ankle_moment_l_nat = 2*stdmomnat['ankle_angle_l_moment']
    std_ankle_moment_l_exo = 2*stdmomexo['ankle_angle_l_moment']
    # now resample
    std_hip_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100,len(std_hip_moment_r_nat)), std_hip_moment_r_nat)
    std_hip_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_r_exo)), std_hip_moment_r_exo)
    std_knee_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_nat)), std_knee_moment_r_nat)
    std_knee_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_exo)), std_knee_moment_r_exo)
    std_ankle_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_nat)), std_ankle_moment_r_nat)
    std_ankle_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_exo)), std_ankle_moment_r_exo)
    std_hip_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_nat)), std_hip_moment_l_nat)
    std_hip_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_exo)), std_hip_moment_l_exo)
    std_knee_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_nat)), std_knee_moment_l_nat)
    std_knee_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_exo)), std_knee_moment_l_exo)
    std_ankle_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_nat)), std_ankle_moment_l_nat)
    std_ankle_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_exo)), std_ankle_moment_l_exo)
    # and cut down to a step rather than the full gait cycle
    std_hip_moment_r_nat = std_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_hip_moment_r_exo = std_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_r_nat = std_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_knee_moment_r_exo = std_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_ankle_moment_r_nat = std_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    std_ankle_moment_r_exo = std_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_hip_moment_l_nat = std_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_hip_moment_l_exo = std_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_l_nat = std_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_knee_moment_l_exo = std_knee_moment_l_exo[0:int(moment_simlen_nat/2)]   
    std_ankle_moment_l_nat = std_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    std_ankle_moment_l_exo = std_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    

    ##########################################################################
    # first is the natural validation figure for saggital coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    # x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_nat/2))
    x_mom = np.linspace(0, 100, moment_simlen_nat)[0:int(moment_simlen_nat/2)]
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    # Hip angle
    ax[0,0].plot(x_kin, ref_hip_angle_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,0].plot(x_kin, ref_hip_angle_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0, 0].plot(x_kin, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[0,0].fill_between(x_kin, ref_hip_angle_r_nat - std_hip_angle_r_nat, ref_hip_angle_r_nat + std_hip_angle_r_nat, color='orange', alpha=0.2)
    ax[1,0].fill_between(x_kin, ref_hip_angle_l_nat - std_hip_angle_l_nat, ref_hip_angle_l_nat + std_hip_angle_l_nat, color='orange', alpha=0.2)
    # ax[0, 0].fill_between(x_kin, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0,0].plot(x_kin, sim_hip_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,0].plot(x_kin, sim_hip_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0, 0].plot(x_kin, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,0].legend(fontsize=12) #, loc='lower right')
    # other leg
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=12)
    # ax[1,0].legend(fontsize=12) #, loc='lower right')

    # Knee angle
    ax[0,1].plot(x_kin, ref_knee_angle_r_nat, label='Experimental reference', color='orange', linestyle='--')
    ax[1,1].plot(x_kin, ref_knee_angle_l_nat, label='Experimental reference', color='orange', linestyle='--')
    # ax[0,0].plot(x_kin, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[0,1].fill_between(x_kin, ref_knee_angle_r_nat - std_knee_angle_r_nat, ref_knee_angle_r_nat + std_knee_angle_r_nat, color='orange', alpha=0.2)
    ax[1,1].fill_between(x_kin, ref_knee_angle_l_nat - std_knee_angle_l_nat, ref_knee_angle_l_nat + std_knee_angle_l_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    # ax[0,0].fill_between(x_kin, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[0,1].plot(x_kin, sim_knee_angle_r_nat * 180 / np.pi, label='Simulated', color='orange', linewidth=3)
    ax[1,1].plot(x_kin, sim_knee_angle_l_nat * 180 / np.pi, label='Simulated', color='orange', linewidth=3)
    # ax[0,0].plot(x_kin, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    # ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    # ax[0,1].legend(fontsize=12) #, loc='upper left')
    # other leg
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    # ax[1,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10) #, loc='upper left')

    # Ankle angle
    ax[0,2].plot(x_kin, ref_ankle_angle_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,2].plot(x_kin, ref_ankle_angle_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0,0].plot(x_kin, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[0,2].fill_between(x_kin, ref_ankle_angle_r_nat - std_ankle_angle_r_nat, ref_ankle_angle_r_nat + std_ankle_angle_r_nat, color='orange', alpha=0.2)
    ax[1,2].fill_between(x_kin, ref_ankle_angle_l_nat - std_ankle_angle_l_nat, ref_ankle_angle_l_nat + std_ankle_angle_l_nat, color='orange', alpha=0.2)
    # ax[0,0].fill_between(x_kin, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[0,2].plot(x_kin, sim_ankle_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,2].plot(x_kin, sim_ankle_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0,0].plot(x_kin, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    # ax[0,2].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,2].set_xlabel('Gait Cycle %', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    # ax[0,2].legend(fontsize=12) #, loc='upper right')
    # other leg
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    # ax[1,2].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)
    # ax[1,2].legend(fontsize=12) #, loc='upper right')
    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_nat_27.png')
    plt.show()
    ##########################################################################
    # second is the exotendon version of the validation figure for saggital coordinates.
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_exo)[0:int(kin_simlen_exo/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_exo/2))
    x_mom = np.linspace(0, 100, moment_simlen_exo)[0:int(moment_simlen_exo/2)]
    x_grf = np.linspace(0, 100, grf_simlen_exo)[0:int(grf_simlen_exo/2)]
    # Hip angle
    ax[0,0].plot(x_kin, ref_hip_angle_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    ax[1,0].plot(x_kin, ref_hip_angle_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    ax[0,0].fill_between(x_kin, ref_hip_angle_r_exo - std_hip_angle_r_exo, ref_hip_angle_r_exo + std_hip_angle_r_exo, color='purple', alpha=0.2)
    ax[1,0].fill_between(x_kin, ref_hip_angle_l_exo - std_hip_angle_l_exo, ref_hip_angle_l_exo + std_hip_angle_l_exo, color='purple', alpha=0.2)
    ax[0,0].plot(x_kin, sim_hip_angle_r_exo * 180 / np.pi, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[1,0].plot(x_kin, sim_hip_angle_l_exo * 180 / np.pi, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=12)
    # Knee angle
    ax[0,1].plot(x_kin, ref_knee_angle_r_exo, label='Experimental reference', color='purple', linestyle='--')
    ax[1,1].plot(x_kin, ref_knee_angle_l_exo, label='Experimental reference', color='purple', linestyle='--')
    ax[0,1].fill_between(x_kin, ref_knee_angle_r_exo - std_knee_angle_r_exo, ref_knee_angle_r_exo + std_knee_angle_r_exo, color='purple', alpha=0.2)
    ax[1,1].fill_between(x_kin, ref_knee_angle_l_exo - std_knee_angle_l_exo, ref_knee_angle_l_exo + std_knee_angle_l_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[0,1].plot(x_kin, sim_knee_angle_r_exo * 180 / np.pi, label='Simulated', color='purple', linewidth=3)
    ax[1,1].plot(x_kin, sim_knee_angle_l_exo * 180 / np.pi, label='Simulated', color='purple', linewidth=3)
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10)
    # Ankle angle
    ax[0,2].plot(x_kin, ref_ankle_angle_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    ax[1,2].plot(x_kin, ref_ankle_angle_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    ax[0,2].fill_between(x_kin, ref_ankle_angle_r_exo - std_ankle_angle_r_exo, ref_ankle_angle_r_exo + std_ankle_angle_r_exo, color='purple', alpha=0.2)
    ax[1,2].fill_between(x_kin, ref_ankle_angle_l_exo - std_ankle_angle_l_exo, ref_ankle_angle_l_exo + std_ankle_angle_l_exo, color='purple', alpha=0.2)
    ax[0,2].plot(x_kin, sim_ankle_angle_r_exo * 180 / np.pi, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[1,2].plot(x_kin, sim_ankle_angle_l_exo * 180 / np.pi, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_exo_27.png')
    plt.show()

    ##########################################################################
    # third is the grf data for both natural and exotendon.
    ##########################################################################
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    x_grf = np.linspace(0,100,int(grf_simlen_nat/2))
    # Natural GRF y
    ax[0, 0].plot(x_grf, ref_grf_y_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[0, 0].fill_between(x_grf, ref_grf_y_nat - std_grf_y_nat, ref_grf_y_nat + std_grf_y_nat, color='orange', alpha=0.2)
    ax[0, 0].plot(x_grf, sim_grf_y_nat, label='Nat. Sim', color='orange', linewidth=3)
    ax[0, 0].set_title('Natural Superior(+) Vertical GRF', fontsize=12)
    ax[0, 0].set_ylabel('Force (BW)', fontsize=12)
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].tick_params(axis='x', labelsize=12)
    # ax[0, 0].legend(fontsize=12)
    # Natural GRF x
    ax[0, 1].plot(x_grf, ref_grf_x_nat, label='Experimental Reference', color='orange', linestyle='--')
    ax[0, 1].fill_between(x_grf, ref_grf_x_nat - std_grf_x_nat, ref_grf_x_nat + std_grf_x_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    ax[0, 1].plot(x_grf, sim_grf_x_nat, label='Simulated', color='orange', linewidth=3)
    ax[0, 1].set_title('Natural Anterior(+) Horizontal GRF', fontsize=12)
    # ax[0, 1].set_ylabel('Force (BW)', fontsize=12)
    # ax[0, 1].set_xlabel('Step %', fontsize=12)
    ax[0, 1].tick_params(axis='y', labelsize=12)
    ax[0, 1].tick_params(axis='x', labelsize=12)
    ax[0, 1].legend(fontsize=10)
    # Exo GRF y
    ax[1, 0].plot(x_grf, ref_grf_y_exo, label='Exo Ref', color='purple', linestyle='--')
    ax[1, 0].fill_between(x_grf, ref_grf_y_exo - std_grf_y_exo, ref_grf_y_exo + std_grf_y_exo, color='purple', alpha=0.2)
    ax[1, 0].plot(x_grf, sim_grf_y_exo, label='Exo Sim', color='purple', linewidth=3)
    ax[1, 0].set_title('Exotendon Superior(+) Vertical GRF', fontsize=12)
    ax[1, 0].set_ylabel('Force (BW)', fontsize=12)
    ax[1, 0].set_xlabel('Step %', fontsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='x', labelsize=12)
    # ax[1, 0].legend(fontsize=12)
    # Exo GRF x
    ax[1, 1].plot(x_grf, ref_grf_x_exo, label='Experimental Reference', color='purple', linestyle='--')
    ax[1, 1].fill_between(x_grf, ref_grf_x_exo - std_grf_x_exo, ref_grf_x_exo + std_grf_x_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[1, 1].plot(x_grf, sim_grf_x_exo, label='Simulated', color='purple', linewidth=3)
    ax[1, 1].set_title('Exotendon Anterior(+) Horizontal GRF', fontsize=12)
    # ax[1, 1].set_ylabel('Force (BW)', fontsize=12)
    ax[1, 1].set_xlabel('Step %', fontsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='x', labelsize=12)
    ax[1, 1].legend(fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_grf_validation_27.png')
    plt.show()

    ##########################################################################
    # fourth is the pelvis coordinates for natural 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_nat/2))

    # Pelvis ty
    ax[0, 0].plot(x_kin, ref_ty_nat, label='Experimental Reference', color='orange', linestyle='--')
    ax[0, 0].fill_between(x_kin, ref_ty_nat - std_ty_nat, ref_ty_nat + std_ty_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    ax[0, 0].plot(x_kin, sim_ty_nat, label='Simulated', color='orange', linewidth=3)
    ax[0, 0].set_title('Pelvis Vertical Translation', fontsize=12)
    ax[0, 0].set_ylabel('Translation (m/height)', fontsize=12)
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].tick_params(axis='x', labelsize=12)
    # ax[0, 0].legend(fontsize=12)

    # Pelvis tx
    # ax[0, 1].plot(x_kin, ref_tx_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[0, 1].fill_between(x_kin, ref_tx_nat - std_tx_nat, ref_tx_nat + std_tx_nat, color='orange', alpha=0.2)
    # ax[0, 1].plot(x_kin, sim_tx_nat, label='Nat. Sim', color='orange', linewidth=3)
    # ax[0, 1].set_title('Pelvis Horizontal Translation', fontsize=12)
    # # ax[0, 1].set_ylabel('Translation (m/height)', fontsize=12)
    # ax[0, 1].tick_params(axis='y', labelsize=12)
    # ax[0, 1].tick_params(axis='x', labelsize=12)
    # ax[0, 1].legend(fontsize=12)

    # Pelvis list
    ax[1, 0].plot(x_kin, ref_pelvlist_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[1, 0].fill_between(x_kin, ref_pelvlist_nat - std_pelvlist_nat, ref_pelvlist_nat + std_pelvlist_nat, color='orange', alpha=0.2)
    ax[1, 0].plot(x_kin, sim_pelvlist_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 0].set_title('Pelvis List', fontsize=12)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='x', labelsize=12)
    ax[1, 0].set_xlabel('Step %', fontsize=12)
    # ax[1, 0].legend(fontsize=12)

    # Pelvis tilt
    ax[1, 1].plot(x_kin, ref_pelvtilt_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[1, 1].fill_between(x_kin, ref_pelvtilt_nat - std_pelvtilt_nat, ref_pelvtilt_nat + std_pelvtilt_nat, color='orange', alpha=0.2)
    ax[1, 1].plot(x_kin, sim_pelvtilt_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 1].set_title('Pelvis Tilt', fontsize=12)
    # ax[1, 1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='x', labelsize=12)
    ax[1, 1].set_xlabel('Step %', fontsize=12)
    # ax[1, 1].legend(fontsize=12)

    # Pelvis rotation
    ax[0,1].plot(x_kin, ref_pelvrot_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[0,1].fill_between(x_kin, ref_pelvrot_nat - std_pelvrot_nat, ref_pelvrot_nat + std_pelvrot_nat, color='orange', alpha=0.2)
    ax[0,1].plot(x_kin, sim_pelvrot_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[0,1].set_title('Pelvis Rotation', fontsize=12)
    ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='x', labelsize=12)    
    ax[0,1].set_xlabel('Step %', fontsize=12)
    # ax[0,1].legend(fontsize=12)

    # Hide the last subplot and use it to display the legend   
    ax[0,2].axis('off')
    ax[1,2].axis('off')
    # get the legend labels from the previous subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[0, 2].legend(handles, labels, loc='center left', fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_pelvis_validation_nat_27.png')
    plt.show()


    ##########################################################################
    # fourth is the pelvis coordinates for exotendon 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_nat/2))

    # Pelvis ty
    ax[0, 0].plot(x_kin, ref_ty_exo, label='Experimental Reference', color='purple', linestyle='--')
    ax[0, 0].fill_between(x_kin, ref_ty_exo - std_ty_exo, ref_ty_exo + std_ty_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[0, 0].plot(x_kin, sim_ty_exo, label='Simulated', color='purple', linewidth=3)
    ax[0, 0].set_title('Pelvis Vertical Translation', fontsize=12)
    ax[0, 0].set_ylabel('Translation (m/height)', fontsize=12)
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].tick_params(axis='x', labelsize=12)
    # ax[0, 0].legend(fontsize=12)

    # Pelvis tx
    # ax[0, 1].plot(x_kin, ref_tx_exo, label='Nat. Ref', color='purple', linestyle='--')
    # ax[0, 1].fill_between(x_kin, ref_tx_exo - std_tx_exo, ref_tx_exo + std_tx_exo, color='purple', alpha=0.2)
    # ax[0, 1].plot(x_kin, sim_tx_exo, label='Nat. Sim', color='purple', linewidth=3)
    # ax[0, 1].set_title('Pelvis Horizontal Translation', fontsize=12)
    # # ax[0, 1].set_ylabel('Translation (m/height)', fontsize=12)
    # ax[0, 1].tick_params(axis='y', labelsize=12)
    # ax[0, 1].tick_params(axis='x', labelsize=12)
    # ax[0, 1].legend(fontsize=12)

    # Pelvis list
    ax[1, 0].plot(x_kin, ref_pelvlist_exo, label='Nat. Ref', color='purple', linestyle='--')
    ax[1, 0].fill_between(x_kin, ref_pelvlist_exo - std_pelvlist_exo, ref_pelvlist_exo + std_pelvlist_exo, color='purple', alpha=0.2)
    ax[1, 0].plot(x_kin, sim_pelvlist_exo*180/np.pi, label='Nat. Sim', color='purple', linewidth=3)
    ax[1, 0].set_title('Pelvis List', fontsize=12)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='x', labelsize=12)
    ax[1, 0].set_xlabel('Step %', fontsize=12)
    # ax[1, 0].legend(fontsize=12)

    # Pelvis tilt
    ax[1, 1].plot(x_kin, ref_pelvtilt_exo, label='Nat. Ref', color='purple', linestyle='--')
    ax[1, 1].fill_between(x_kin, ref_pelvtilt_exo - std_pelvtilt_exo, ref_pelvtilt_exo + std_pelvtilt_exo, color='purple', alpha=0.2)
    ax[1, 1].plot(x_kin, sim_pelvtilt_exo*180/np.pi, label='Nat. Sim', color='purple', linewidth=3)
    ax[1, 1].set_title('Pelvis Tilt', fontsize=12)
    # ax[1, 1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='x', labelsize=12)
    ax[1, 1].set_xlabel('Step %', fontsize=12)
    # ax[1, 1].legend(fontsize=12)

    # Pelvis rotation
    ax[0,1].plot(x_kin, ref_pelvrot_exo, label='Nat. Ref', color='purple', linestyle='--')
    ax[0,1].fill_between(x_kin, ref_pelvrot_exo - std_pelvrot_exo, ref_pelvrot_exo + std_pelvrot_exo, color='purple', alpha=0.2)
    ax[0,1].plot(x_kin, sim_pelvrot_exo*180/np.pi, label='Nat. Sim', color='purple', linewidth=3)
    ax[0,1].set_title('Pelvis Rotation', fontsize=12)
    ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='x', labelsize=12)    
    ax[0,1].set_xlabel('Step %', fontsize=12)
    # ax[0,1].legend(fontsize=12)

    # Hide the last subplot and use it to display the legend   
    ax[0,2].axis('off')
    ax[1,2].axis('off')
    # get the legend labels from the previous subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[0, 2].legend(handles, labels, loc='center left', fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_pelvis_validation_exo_27.png')
    plt.show()


    ##### now move to the moments.
    ##########################################################################
    # first is the natural validation figure for saggital coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    # x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_nat/2))
    x_mom = np.linspace(0,100,int(moment_simlen_nat/2))

    # Hip moment
    ax[0,0].plot(x_mom, ref_hip_moment_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,0].plot(x_mom, ref_hip_moment_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0, 0].plot(x_mom, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[0,0].fill_between(x_mom, ref_hip_moment_r_nat - std_hip_moment_r_nat, ref_hip_moment_r_nat + std_hip_moment_r_nat, color='orange', alpha=0.2)
    ax[1,0].fill_between(x_mom, ref_hip_moment_l_nat - std_hip_moment_l_nat, ref_hip_moment_l_nat + std_hip_moment_l_nat, color='orange', alpha=0.2)
    # ax[0, 0].fill_between(x_mom, ref_hip_moment_exo - 2*std_exo['hip_flexion_r'], ref_hip_moment_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0,0].plot(x_mom, sim_hip_moment_r_nat, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,0].plot(x_mom, sim_hip_moment_l_nat, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0, 0].plot(x_mom, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    # ax[0,0].legend(fontsize=12) #, loc='lower right')
    # other leg
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    # ax[1,0].legend(fontsize=12) #, loc='lower right')

    # Knee angle
    ax[0,1].plot(x_mom, ref_knee_moment_r_nat, label='Experimental reference', color='orange', linestyle='--')
    ax[1,1].plot(x_mom, ref_knee_moment_l_nat, label='Experimental reference', color='orange', linestyle='--')
    # ax[0,0].plot(x_mom, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[0,1].fill_between(x_mom, ref_knee_moment_r_nat - std_knee_moment_r_nat, ref_knee_moment_r_nat + std_knee_moment_r_nat, color='orange', alpha=0.2)
    ax[1,1].fill_between(x_mom, ref_knee_moment_l_nat - std_knee_moment_l_nat, ref_knee_moment_l_nat + std_knee_moment_l_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    # ax[0,0].fill_between(x_mom, ref_knee_moment_exo - 2*std_exo['knee_moment_r'], ref_knee_moment_exo + 2*std_exo['knee_moment_r'], color='purple', alpha=0.2)
    ax[0,1].plot(x_mom, sim_knee_moment_r_nat, label='Simulated', color='orange', linewidth=3)
    ax[1,1].plot(x_mom, sim_knee_moment_l_nat, label='Simulated', color='orange', linewidth=3)
    # ax[0,0].plot(x_mom, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    # ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    # ax[0,1].legend(fontsize=12) #, loc='upper left')
    # other leg
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    # ax[1,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10) #, loc='upper left')

    # Ankle angle
    ax[0,2].plot(x_mom, ref_ankle_moment_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,2].plot(x_mom, ref_ankle_moment_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0,0].plot(x_mom, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[0,2].fill_between(x_mom, ref_ankle_moment_r_nat - std_ankle_moment_r_nat, ref_ankle_moment_r_nat + std_ankle_moment_r_nat, color='orange', alpha=0.2)
    ax[1,2].fill_between(x_mom, ref_ankle_moment_l_nat - std_ankle_moment_l_nat, ref_ankle_moment_l_nat + std_ankle_moment_l_nat, color='orange', alpha=0.2)
    # ax[0,0].fill_between(x_mom, ref_ankle_moment_exo - 2*std_exo['ankle_moment_r'], ref_ankle_moment_exo + 2*std_exo['ankle_moment_r'], color='purple', alpha=0.2)
    ax[0,2].plot(x_mom, sim_ankle_moment_r_nat, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,2].plot(x_mom, sim_ankle_moment_l_nat, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0,0].plot(x_mom, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    # ax[0,2].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,2].set_xlabel('Gait Cycle %', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    # ax[0,2].legend(fontsize=12) #, loc='upper right')
    # other leg
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    # ax[1,2].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)
    # ax[1,2].legend(fontsize=12) #, loc='upper right')
    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_moment_nat_27.png')
    plt.show()


    ##########################################################################
    # second is the exotendon version of the validation figure for saggital coordinates.
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_mom_exo = np.linspace(0,100,int(moment_simlen_exo/2))
    x_mom_nat = np.linspace(0,100,int(moment_simlen_nat/2))

    # Hip angle
    ax[0,0].plot(x_mom_exo, ref_hip_moment_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    ax[1,0].plot(x_mom_exo, ref_hip_moment_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    ax[0,0].fill_between(x_mom_exo, ref_hip_moment_r_exo - std_hip_moment_r_exo, ref_hip_moment_r_exo + std_hip_moment_r_exo, color='purple', alpha=0.2)
    ax[1,0].fill_between(x_mom_exo, ref_hip_moment_l_exo - std_hip_moment_l_exo, ref_hip_moment_l_exo + std_hip_moment_l_exo, color='purple', alpha=0.2)
    ax[0,0].plot(x_mom_exo, sim_hip_moment_r_exo, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[1,0].plot(x_mom_exo, sim_hip_moment_l_exo, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    # Knee angle
    ax[0,1].plot(x_mom_exo, ref_knee_moment_r_exo, label='Experimental reference', color='purple', linestyle='--')
    ax[1,1].plot(x_mom_exo, ref_knee_moment_l_exo, label='Experimental reference', color='purple', linestyle='--')
    ax[0,1].fill_between(x_mom_exo, ref_knee_moment_r_exo - std_knee_moment_r_exo, ref_knee_moment_r_exo + std_knee_moment_r_exo, color='purple', alpha=0.2)
    ax[1,1].fill_between(x_mom_exo, ref_knee_moment_l_exo - std_knee_moment_l_exo, ref_knee_moment_l_exo + std_knee_moment_l_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[0,1].plot(x_mom_exo, sim_knee_moment_r_exo, label='Simulated', color='purple', linewidth=3)
    ax[1,1].plot(x_mom_exo, sim_knee_moment_l_exo, label='Simulated', color='purple', linewidth=3)
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10)
    # Ankle angle
    ax[0,2].plot(x_mom_exo, ref_ankle_moment_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    ax[1,2].plot(x_mom_exo, ref_ankle_moment_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    ax[0,2].fill_between(x_mom_exo, ref_ankle_moment_r_exo - std_ankle_moment_r_exo, ref_ankle_moment_r_exo + std_ankle_moment_r_exo, color='purple', alpha=0.2)
    ax[1,2].fill_between(x_mom_exo, ref_ankle_moment_l_exo - std_ankle_moment_l_exo, ref_ankle_moment_l_exo + std_ankle_moment_l_exo, color='purple', alpha=0.2)
    ax[0,2].plot(x_mom_exo, sim_ankle_moment_r_exo, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[1,2].plot(x_mom_exo, sim_ankle_moment_l_exo, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_moment_exo_27.png')
    plt.show()

    return

# create validation figure for saggital at 4 m/s
def saggitalValidationSplit40(simNat, simExo, iknat2D, labels2D, coordinates_sim_clean, mean_nat, std_nat, GRFsimnat, GRFsimexo, GRFrefnat, meangrfnat, stdgrfnat, natmomentfile, exomomentfile, idnat, meanmomnat, stdmomnat, modelfile):
    # load the model and get the mass
    model = osim.Model(modelfile)
    mass = model.getTotalMass(model.initSystem())
    massham = 65
    height = 1.78
    ## starting with the kinematics
    # get the length of the simulation data
    kin_simlen_nat = len(simNat.getIndependentColumn())
    kin_simlen_exo = len(simExo.getIndependentColumn())
    if kin_simlen_nat != kin_simlen_exo:
        print('Simulation kinematic data lengths do not match. Exiting.')
        return
    # get the sim kinematics
    sim_hip_angle_r_nat = simNat.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_r_nat = simNat.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_r_nat = simNat.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()    
    sim_hip_angle_l_nat = simNat.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_l_nat = simNat.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_l_nat = simNat.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ty_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    sim_tx_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tx/value').to_numpy()
    sim_pelvtilt_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvlist_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvrot_nat = simNat.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbarRot_nat = simNat.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_lumbarBend_nat = simNat.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbarExt_nat = simNat.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()

    sim_hip_angle_r_exo = simExo.getDependentColumn('/jointset/hip_r/hip_flexion_r/value').to_numpy()
    sim_knee_angle_r_exo = simExo.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
    sim_ankle_angle_r_exo = simExo.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value').to_numpy()
    sim_hip_angle_l_exo = simExo.getDependentColumn('/jointset/hip_l/hip_flexion_l/value').to_numpy()
    sim_knee_angle_l_exo = simExo.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value').to_numpy()
    sim_ankle_angle_l_exo = simExo.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value').to_numpy()
    sim_ty_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_ty/value').to_numpy()
    sim_tx_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_tx/value').to_numpy()
    sim_pelvtilt_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_tilt/value').to_numpy()
    sim_pelvlist_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_list/value').to_numpy()
    sim_pelvrot_exo = simExo.getDependentColumn('/jointset/groundPelvis/pelvis_rotation/value').to_numpy()
    sim_lumbarRot_exo = simExo.getDependentColumn('/jointset/back/lumbar_rotation/value').to_numpy()
    sim_lumbarBend_exo = simExo.getDependentColumn('/jointset/back/lumbar_bending/value').to_numpy()
    sim_lumbarExt_exo = simExo.getDependentColumn('/jointset/back/lumbar_extension/value').to_numpy()    

    # reference data for this speed kinematics. 
    # ref_hip_angle_r_nat = iknat2D.getDependentColumn('hip_flexion_r').to_numpy()
    # ref_knee_angle_r_nat = iknat2D.getDependentColumn('knee_angle_r').to_numpy()
    # ref_ankle_angle_r_nat = iknat2D.getDependentColumn('ankle_angle_r').to_numpy()
    # ref_ty_nat = iknat2D.getDependentColumn('pelvis_ty').to_numpy()
    # ref_tx_nat = iknat2D.getDependentColumn('pelvis_tx').to_numpy()
    # ref_pelvtilt_nat = iknat2D.getDependentColumn('pelvis_tilt').to_numpy()
    # ref_pelvlist_nat = iknat2D.getDependentColumn('pelvis_list').to_numpy()
    # ref_pelvrot_nat = iknat2D.getDependentColumn('pelvis_rotation').to_numpy()
    # ref_lumbarRot_nat = iknat2D.getDependentColumn('lumbar_rotation').to_numpy()
    # ref_lumbarBend_nat = iknat2D.getDependentColumn('lumbar_bending').to_numpy()
    # ref_lumbarExt_nat = iknat2D.getDependentColumn('lumbar_extension').to_numpy()
    # ref_hip_angle_l_nat = iknat2D.getDependentColumn('hip_flexion_l').to_numpy()
    # ref_knee_angle_l_nat = iknat2D.getDependentColumn('knee_angle_l').to_numpy()
    # ref_ankle_angle_l_nat = iknat2D.getDependentColumn('ankle_angle_l').to_numpy()

    # second option for the reference data from experimental data means
    ref_hip_angle_r_nat = mean_nat['hip_flexion_r']
    ref_knee_angle_r_nat = mean_nat['knee_angle_r']
    ref_ankle_angle_r_nat = mean_nat['ankle_angle_r']
    ref_ty_nat = mean_nat['pelvis_ty']
    ref_tx_nat = mean_nat['pelvis_tx']
    ref_pelvtilt_nat = mean_nat['pelvis_tilt']
    ref_pelvlist_nat = mean_nat['pelvis_list']
    ref_pelvrot_nat = mean_nat['pelvis_rotation']
    ref_lumbarRot_nat = mean_nat['lumbar_rotation']
    ref_lumbarBend_nat = mean_nat['lumbar_bending']
    ref_lumbarExt_nat = mean_nat['lumbar_extension']
    ref_hip_angle_l_nat = mean_nat['hip_flexion_l']
    ref_knee_angle_l_nat = mean_nat['knee_angle_l']
    ref_ankle_angle_l_nat = mean_nat['ankle_angle_l']

    # get the length of the reference data
    kin_reflen_nat = len(ref_hip_angle_r_nat)
    # kin_reflen_exo = len(ref_hip_angle_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation. 
    kin_xsim = np.linspace(0,100,kin_simlen_nat)
    kin_xref_nat = np.linspace(0,100,kin_reflen_nat)
    # kin_xref_exo = np.linspace(0,100,kin_reflen_exo)
    
    # interpolate the reference data to the simulation data length
    ref_hip_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_r_nat)
    ref_knee_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_r_nat)
    ref_ankle_angle_r_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_r_nat)
    ref_ty_nat = np.interp(kin_xsim, kin_xref_nat, ref_ty_nat)
    ref_tx_nat = np.interp(kin_xsim, kin_xref_nat, ref_tx_nat)
    ref_pelvtilt_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvtilt_nat)
    ref_pelvlist_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvlist_nat)
    ref_pelvrot_nat = np.interp(kin_xsim, kin_xref_nat, ref_pelvrot_nat)
    ref_lumbarRot_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarRot_nat)
    ref_lumbarBend_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarBend_nat)
    ref_lumbarExt_nat = np.interp(kin_xsim, kin_xref_nat, ref_lumbarExt_nat)
    ref_hip_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_hip_angle_l_nat)
    ref_knee_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_knee_angle_l_nat)
    ref_ankle_angle_l_nat = np.interp(kin_xsim, kin_xref_nat, ref_ankle_angle_l_nat)
    
    # ref_hip_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_r_exo)
    # ref_knee_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_r_exo)
    # ref_ankle_angle_r_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_r_exo)
    # ref_ty_exo = np.interp(kin_xsim, kin_xref_exo, ref_ty_exo)
    # ref_tx_exo = np.interp(kin_xsim, kin_xref_exo, ref_tx_exo)
    # ref_pelvtilt_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvtilt_exo)
    # ref_pelvlist_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvlist_exo)
    # ref_pelvrot_exo = np.interp(kin_xsim, kin_xref_exo, ref_pelvrot_exo)
    # ref_lumbarRot_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarRot_exo)
    # ref_lumbarBend_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarBend_exo)
    # ref_lumbarExt_exo = np.interp(kin_xsim, kin_xref_exo, ref_lumbarExt_exo)
    # ref_hip_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_hip_angle_l_exo)
    # ref_knee_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_knee_angle_l_exo)
    # ref_ankle_angle_l_exo = np.interp(kin_xsim, kin_xref_exo, ref_ankle_angle_l_exo)
    
    # normalize the ty values to the height of the subject
    # ref_ty_nat = ref_ty_nat / height
    # ref_ty_exo = ref_ty_exo / height
    # sim_ty_nat = sim_ty_nat / height
    # sim_ty_exo = sim_ty_exo / height
    
    # now shorten to be one step, rather than the full gait cycle. 
    ref_hip_angle_r_nat = ref_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_r_nat = ref_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_r_nat = ref_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    ref_hip_angle_l_nat = ref_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_knee_angle_l_nat = ref_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_ankle_angle_l_nat = ref_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    ref_ty_nat = ref_ty_nat[0:int(kin_simlen_nat/2)]
    ref_tx_nat = ref_tx_nat[0:int(kin_simlen_nat/2)]
    ref_pelvtilt_nat = ref_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    ref_pelvlist_nat = ref_pelvlist_nat[0:int(kin_simlen_nat/2)]
    ref_pelvrot_nat = ref_pelvrot_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarRot_nat = ref_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarBend_nat = ref_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    ref_lumbarExt_nat = ref_lumbarExt_nat[0:int(kin_simlen_nat/2)]

    # ref_hip_angle_r_exo = ref_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    # ref_knee_angle_r_exo = ref_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    # ref_ankle_angle_r_exo = ref_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    # ref_hip_angle_l_exo = ref_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    # ref_knee_angle_l_exo = ref_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    # ref_ankle_angle_l_exo = ref_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    # ref_ty_exo = ref_ty_exo[0:int(kin_simlen_nat/2)]
    # ref_tx_exo = ref_tx_exo[0:int(kin_simlen_nat/2)]
    # ref_pelvtilt_exo = ref_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    # ref_pelvlist_exo = ref_pelvlist_exo[0:int(kin_simlen_nat/2)]
    # ref_pelvrot_exo = ref_pelvrot_exo[0:int(kin_simlen_nat/2)]
    # ref_lumbarRot_exo = ref_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    # ref_lumbarBend_exo = ref_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    # ref_lumbarExt_exo = ref_lumbarExt_exo[0:int(kin_simlen_nat/2)]

    sim_hip_angle_r_nat = sim_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_nat = sim_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_nat = sim_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_nat = sim_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_nat = sim_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_nat = sim_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    sim_ty_nat = sim_ty_nat[0:int(kin_simlen_nat/2)]
    sim_tx_nat = sim_tx_nat[0:int(kin_simlen_nat/2)]
    sim_pelvtilt_nat = sim_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    sim_pelvlist_nat = sim_pelvlist_nat[0:int(kin_simlen_nat/2)]
    sim_pelvrot_nat = sim_pelvrot_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarRot_nat = sim_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarBend_nat = sim_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    sim_lumbarExt_nat = sim_lumbarExt_nat[0:int(kin_simlen_nat/2)]
    
    sim_hip_angle_r_exo = sim_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_r_exo = sim_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_r_exo = sim_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    sim_hip_angle_l_exo = sim_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_knee_angle_l_exo = sim_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_ankle_angle_l_exo = sim_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    sim_ty_exo = sim_ty_exo[0:int(kin_simlen_nat/2)]
    sim_tx_exo = sim_tx_exo[0:int(kin_simlen_nat/2)]
    sim_pelvtilt_exo = sim_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    sim_pelvlist_exo = sim_pelvlist_exo[0:int(kin_simlen_nat/2)]
    sim_pelvrot_exo = sim_pelvrot_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarRot_exo = sim_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarBend_exo = sim_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    sim_lumbarExt_exo = sim_lumbarExt_exo[0:int(kin_simlen_nat/2)]

    # now get the ref std for plus and minus
    std_hip_angle_r_nat = 2*std_nat['hip_flexion_r']
    std_knee_angle_r_nat = 2*std_nat['knee_angle_r']
    std_ankle_angle_r_nat = 2*std_nat['ankle_angle_r']
    std_hip_angle_l_nat = 2*std_nat['hip_flexion_l']
    std_knee_angle_l_nat = 2*std_nat['knee_angle_l']
    std_ankle_angle_l_nat = 2*std_nat['ankle_angle_l']
    std_ty_nat = 2*std_nat['pelvis_ty']
    std_tx_nat = 2*std_nat['pelvis_tx']
    std_pelvtilt_nat = 2*std_nat['pelvis_tilt']
    std_pelvlist_nat = 2*std_nat['pelvis_list']
    std_pelvrot_nat = 2*std_nat['pelvis_rotation']
    std_lumbarRot_nat = 2*std_nat['lumbar_rotation']
    std_lumbarBend_nat = 2*std_nat['lumbar_bending']
    std_lumbarExt_nat = 2*std_nat['lumbar_extension']

    # std_hip_angle_r_exo = 2*std_exo['hip_flexion_r']
    # std_knee_angle_r_exo = 2*std_exo['knee_angle_r']
    # std_ankle_angle_r_exo = 2*std_exo['ankle_angle_r']
    # std_hip_angle_l_exo = 2*std_exo['hip_flexion_l']
    # std_knee_angle_l_exo = 2*std_exo['knee_angle_l']
    # std_ankle_angle_l_exo = 2*std_exo['ankle_angle_l']
    # std_ty_exo = 2*std_exo['pelvis_ty']
    # std_tx_exo = 2*std_exo['pelvis_tx']
    # std_pelvtilt_exo = 2*std_exo['pelvis_tilt']
    # std_pelvlist_exo = 2*std_exo['pelvis_list']
    # std_pelvrot_exo = 2*std_exo['pelvis_rotation']
    # std_lumbarRot_exo = 2*std_exo['lumbar_rotation']
    # std_lumbarBend_exo = 2*std_exo['lumbar_bending']
    # std_lumbarExt_exo = 2*std_exo['lumbar_extension']

    # now resample
    std_hip_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_nat)), std_hip_angle_r_nat)
    std_knee_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_nat)), std_knee_angle_r_nat)
    std_ankle_angle_r_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_nat)), std_ankle_angle_r_nat)
    std_hip_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_hip_angle_l_nat)
    std_knee_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_nat)), std_knee_angle_l_nat)
    std_ankle_angle_l_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_nat)), std_ankle_angle_l_nat)
    std_ty_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_ty_nat)), std_ty_nat)
    std_tx_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_tx_nat)), std_tx_nat)
    std_pelvtilt_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvtilt_nat)), std_pelvtilt_nat)
    std_pelvlist_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvlist_nat)), std_pelvlist_nat)
    std_pelvrot_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvrot_nat)), std_pelvrot_nat)
    std_lumbarRot_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarRot_nat)), std_lumbarRot_nat)
    std_lumbarBend_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarBend_nat)), std_lumbarBend_nat)
    std_lumbarExt_nat = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarExt_nat)), std_lumbarExt_nat)
    
    # std_hip_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_r_exo)), std_hip_angle_r_exo)
    # std_knee_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_r_exo)), std_knee_angle_r_exo)
    # std_ankle_angle_r_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_r_exo)), std_ankle_angle_r_exo)
    # std_hip_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_hip_angle_l_exo)), std_hip_angle_l_exo)
    # std_knee_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_knee_angle_l_exo)), std_knee_angle_l_exo)
    # std_ankle_angle_l_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ankle_angle_l_exo)), std_ankle_angle_l_exo)
    # std_ty_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_ty_exo)), std_ty_exo)
    # std_tx_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_tx_exo)), std_tx_exo)
    # std_pelvtilt_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvtilt_exo)), std_pelvtilt_exo)
    # std_pelvlist_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvlist_exo)), std_pelvlist_exo)
    # std_pelvrot_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_pelvrot_exo)), std_pelvrot_exo)
    # std_lumbarRot_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarRot_exo)), std_lumbarRot_exo)
    # std_lumbarBend_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarBend_exo)), std_lumbarBend_exo)
    # std_lumbarExt_exo = np.interp(kin_xsim, np.linspace(0,100,len(std_lumbarExt_exo)), std_lumbarExt_exo)

    
    # and cut down to a step rather than the full gait cycle
    std_hip_angle_r_nat = std_hip_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_r_nat = std_knee_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_r_nat = std_ankle_angle_r_nat[0:int(kin_simlen_nat/2)]
    std_hip_angle_l_nat = std_hip_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_knee_angle_l_nat = std_knee_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_ankle_angle_l_nat = std_ankle_angle_l_nat[0:int(kin_simlen_nat/2)]
    std_ty_nat = std_ty_nat[0:int(kin_simlen_nat/2)]
    std_tx_nat = std_tx_nat[0:int(kin_simlen_nat/2)]
    std_pelvtilt_nat = std_pelvtilt_nat[0:int(kin_simlen_nat/2)]
    std_pelvlist_nat = std_pelvlist_nat[0:int(kin_simlen_nat/2)]
    std_pelvrot_nat = std_pelvrot_nat[0:int(kin_simlen_nat/2)]
    std_lumbarRot_nat = std_lumbarRot_nat[0:int(kin_simlen_nat/2)]
    std_lumbarBend_nat = std_lumbarBend_nat[0:int(kin_simlen_nat/2)]
    std_lumbarExt_nat = std_lumbarExt_nat[0:int(kin_simlen_nat/2)]

    # std_hip_angle_r_exo = std_hip_angle_r_exo[0:int(kin_simlen_nat/2)]
    # std_knee_angle_r_exo = std_knee_angle_r_exo[0:int(kin_simlen_nat/2)]
    # std_ankle_angle_r_exo = std_ankle_angle_r_exo[0:int(kin_simlen_nat/2)]
    # std_hip_angle_l_exo = std_hip_angle_l_exo[0:int(kin_simlen_nat/2)]
    # std_knee_angle_l_exo = std_knee_angle_l_exo[0:int(kin_simlen_nat/2)]
    # std_ankle_angle_l_exo = std_ankle_angle_l_exo[0:int(kin_simlen_nat/2)]
    # std_ty_exo = std_ty_exo[0:int(kin_simlen_nat/2)]
    # std_tx_exo = std_tx_exo[0:int(kin_simlen_nat/2)]
    # std_pelvtilt_exo = std_pelvtilt_exo[0:int(kin_simlen_nat/2)]
    # std_pelvlist_exo = std_pelvlist_exo[0:int(kin_simlen_nat/2)]
    # std_pelvrot_exo = std_pelvrot_exo[0:int(kin_simlen_nat/2)]
    # std_lumbarRot_exo = std_lumbarRot_exo[0:int(kin_simlen_nat/2)]
    # std_lumbarBend_exo = std_lumbarBend_exo[0:int(kin_simlen_nat/2)]
    # std_lumbarExt_exo = std_lumbarExt_exo[0:int(kin_simlen_nat/2)]


    ## get the GRF data
    grfsimnat = osim.TimeSeriesTable(GRFsimnat)
    grfsimexo = osim.TimeSeriesTable(GRFsimexo)
    grfrefnat = osim.TimeSeriesTable(GRFrefnat)
    # grfrefexo = osim.TimeSeriesTable(GRFrefexo)
    # get the length of the simulation data
    grf_simlen_nat = len(grfsimnat.getIndependentColumn())
    grf_simlen_exo = len(grfsimexo.getIndependentColumn())
    if grf_simlen_nat != grf_simlen_exo:
        print('Simulation GRF data lengths do not match. Exiting.')
        return
    # get the sim GRF data
    sim_grf_y_nat = grfsimnat.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_y_exo = grfsimexo.getDependentColumn('ground_force_r_vy').to_numpy()
    sim_grf_x_nat = grfsimnat.getDependentColumn('ground_force_r_vx').to_numpy()
    sim_grf_x_exo = grfsimexo.getDependentColumn('ground_force_r_vx').to_numpy()

    # get the reference GRF data
    # ref_grf_y_nat = grfrefnat.getDependentColumn('R_ground_force_vy').to_numpy()
    # ref_grf_x_nat = grfrefnat.getDependentColumn('R_ground_force_vx').to_numpy()
    # second option for the reference data from mean experimental data. 
    ref_grf_y_nat = meangrfnat['ground_force_vy']
    ref_grf_x_nat = meangrfnat['ground_force_vx']

    # get the length of the reference data
    grf_reflen_nat = len(ref_grf_y_nat)
    # grf_reflen_exo = len(ref_grf_y_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    grf_xsim = np.linspace(0,100,grf_simlen_nat)
    grf_xref_nat = np.linspace(0,100,grf_reflen_nat)
    # grf_xref_exo = np.linspace(0,100,grf_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_grf_y_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_y_nat)
    # ref_grf_y_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_y_exo)
    ref_grf_x_nat = np.interp(grf_xsim, grf_xref_nat, ref_grf_x_nat)
    # ref_grf_x_exo = np.interp(grf_xsim, grf_xref_exo, ref_grf_x_exo)
    # divide all of the GRF data based on the mass of the model
    sim_grf_y_nat = sim_grf_y_nat/(mass*9.81)
    sim_grf_y_exo = sim_grf_y_exo/(mass*9.81)
    sim_grf_x_nat = sim_grf_x_nat/(mass*9.81)
    sim_grf_x_exo = sim_grf_x_exo/(mass*9.81)
    ref_grf_y_nat = ref_grf_y_nat#/(mass*9.81)
    ref_grf_x_nat = ref_grf_x_nat#/(mass*9.81)

    # now shorten to be one step rather than the full gait cycle. 
    ref_grf_y_nat = ref_grf_y_nat[0:int(grf_simlen_nat/2)]
    # ref_grf_y_exo = ref_grf_y_exo[0:int(grf_simlen_nat/2)]
    ref_grf_x_nat = ref_grf_x_nat[0:int(grf_simlen_nat/2)]
    # ref_grf_x_exo = ref_grf_x_exo[0:int(grf_simlen_nat/2)]
    
    sim_grf_y_nat = sim_grf_y_nat[0:int(grf_simlen_nat/2)]
    sim_grf_y_exo = sim_grf_y_exo[0:int(grf_simlen_nat/2)]
    sim_grf_x_nat = sim_grf_x_nat[0:int(grf_simlen_nat/2)]
    sim_grf_x_exo = sim_grf_x_exo[0:int(grf_simlen_nat/2)]

    # get the std data for plus and minus
    std_grf_y_nat = 2*stdgrfnat['ground_force_vy']
    # std_grf_y_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fy']
    std_grf_x_nat = 2*stdgrfnat['ground_force_vx']
    # std_grf_x_exo = 2*stdgrfexo['calcn_r_Right_GRF_Fx']
    # now resample
    std_grf_y_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_nat)), std_grf_y_nat)
    # std_grf_y_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_y_exo)), std_grf_y_exo)
    std_grf_x_nat = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_nat)), std_grf_x_nat)
    # std_grf_x_exo = np.interp(grf_xsim, np.linspace(0,100,len(std_grf_x_exo)), std_grf_x_exo)
    # and cut down to a step rather than the full gait cycle
    std_grf_y_nat = std_grf_y_nat[0:int(grf_simlen_nat/2)]
    # std_grf_y_exo = std_grf_y_exo[0:int(grf_simlen_nat/2)]
    std_grf_x_nat = std_grf_x_nat[0:int(grf_simlen_nat/2)]
    # std_grf_x_exo = std_grf_x_exo[0:int(grf_simlen_nat/2)]

    
    ## get the moment data
    natmoment = osim.TimeSeriesTable(natmomentfile)
    exomoment = osim.TimeSeriesTable(exomomentfile)
    natrefmoment = osim.TimeSeriesTable(idnat)
    # exorefmoment = osim.TimeSeriesTable(idexo)
    # get the length of the simulation data
    moment_simlen_nat = len(natmoment.getIndependentColumn())
    moment_simlen_exo = len(exomoment.getIndependentColumn())
    if moment_simlen_nat != moment_simlen_exo:
        print('Simulation moment data lengths do not match. Exiting.')
        return
    
    # get the sim moment data
    sim_hip_moment_r_nat = natmoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_hip_moment_r_exo = exomoment.getDependentColumn('hip_flexion_r_moment').to_numpy()
    sim_knee_moment_r_nat = natmoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_knee_moment_r_exo = exomoment.getDependentColumn('knee_angle_r_moment').to_numpy()
    sim_ankle_moment_r_nat = natmoment.getDependentColumn('ankle_angle_r_moment').to_numpy()
    sim_ankle_moment_r_exo = exomoment.getDependentColumn('ankle_angle_r_moment').to_numpy()

    sim_hip_moment_l_nat = natmoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_hip_moment_l_exo = exomoment.getDependentColumn('hip_flexion_l_moment').to_numpy()
    sim_knee_moment_l_nat = natmoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_knee_moment_l_exo = exomoment.getDependentColumn('knee_angle_l_moment').to_numpy()
    sim_ankle_moment_l_nat = natmoment.getDependentColumn('ankle_angle_l_moment').to_numpy()
    sim_ankle_moment_l_exo = exomoment.getDependentColumn('ankle_angle_l_moment').to_numpy()

    # # get the reference moment data    
    # ref_hip_moment_r_nat = idnat.getDependentColumn('hip_flexion_r_moment').to_numpy()
    # ref_knee_moment_r_nat = idnat.getDependentColumn('knee_angle_r_moment').to_numpy()
    # ref_ankle_moment_r_nat = idnat.getDependentColumn('ankle_angle_r_moment').to_numpy()
    # ref_hip_moment_l_nat = idnat.getDependentColumn('hip_flexion_l_moment').to_numpy()
    # ref_knee_moment_l_nat = idnat.getDependentColumn('knee_angle_l_moment').to_numpy()
    # ref_ankle_moment_l_nat = idnat.getDependentColumn('ankle_angle_l_moment').to_numpy()
    # second option from mean experimental data
    ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    ref_hip_moment_l_nat = meanmomnat['hip_flexion_l_moment']
    ref_knee_moment_l_nat = meanmomnat['knee_angle_l_moment']
    ref_ankle_moment_l_nat = meanmomnat['ankle_angle_l_moment']

    # ref_hip_moment_r_nat = meanmomnat['hip_flexion_r_moment']
    # ref_hip_moment_r_exo = meanmomexo['hip_flexion_r_moment']
    # ref_knee_moment_r_nat = meanmomnat['knee_angle_r_moment']
    # ref_knee_moment_r_exo = meanmomexo['knee_angle_r_moment']
    # ref_ankle_moment_r_nat = meanmomnat['ankle_angle_r_moment']
    # ref_ankle_moment_r_exo = meanmomexo['ankle_angle_r_moment']

    # get the length of the reference data
    moment_reflen_nat = len(ref_hip_moment_r_nat)
    # moment_reflen_exo = len(ref_hip_moment_r_exo)
    # get x vectors for later interpolation of the reference data to the same length as the simulation.
    moment_xsim = np.linspace(0,100,moment_simlen_nat)
    moment_xref_nat = np.linspace(0,100,moment_reflen_nat)
    # moment_xref_exo = np.linspace(0,100,moment_reflen_exo)
    # interpolate the reference data to the simulation data length
    ref_hip_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_r_nat)
    # ref_hip_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_r_exo)
    ref_knee_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_r_nat)
    # ref_knee_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_r_exo)
    ref_ankle_moment_r_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_r_nat)
    # ref_ankle_moment_r_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_r_exo)

    ref_hip_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_hip_moment_l_nat)
    # ref_hip_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_hip_moment_l_exo)
    ref_knee_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_knee_moment_l_nat)
    # ref_knee_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_knee_moment_l_exo)
    ref_ankle_moment_l_nat = np.interp(moment_xsim, moment_xref_nat, ref_ankle_moment_l_nat)
    # ref_ankle_moment_l_exo = np.interp(moment_xsim, moment_xref_exo, ref_ankle_moment_l_exo)
    
    # normalize all of it to body mass
    sim_hip_moment_r_nat = sim_hip_moment_r_nat/mass    
    sim_hip_moment_r_exo = sim_hip_moment_r_exo/mass
    sim_knee_moment_r_nat = sim_knee_moment_r_nat/mass
    sim_knee_moment_r_exo = sim_knee_moment_r_exo/mass
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat/mass
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo/mass
    sim_hip_moment_l_nat = sim_hip_moment_l_nat/mass
    sim_hip_moment_l_exo = sim_hip_moment_l_exo/mass
    sim_knee_moment_l_nat = sim_knee_moment_l_nat/mass
    sim_knee_moment_l_exo = sim_knee_moment_l_exo/mass
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat/mass
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo/mass

    ref_hip_moment_r_nat = ref_hip_moment_r_nat#/mass
    ref_knee_moment_r_nat = ref_knee_moment_r_nat#/mass
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat#/mass
    ref_hip_moment_l_nat = ref_hip_moment_l_nat#/mass
    ref_knee_moment_l_nat = ref_knee_moment_l_nat#/mass
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat#/mass
    
    # and now shorten to the single step, rather than gait cycle. 
    ref_hip_moment_r_nat = ref_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_r_nat = ref_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_r_nat = ref_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    ref_hip_moment_l_nat = ref_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_knee_moment_l_nat = ref_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    ref_ankle_moment_l_nat = ref_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]

    sim_hip_moment_r_nat = sim_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_r_exo = sim_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_nat = sim_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_r_exo = sim_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_nat = sim_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_r_exo = sim_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_nat = sim_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_hip_moment_l_exo = sim_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_nat = sim_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_knee_moment_l_exo = sim_knee_moment_l_exo[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_nat = sim_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    sim_ankle_moment_l_exo = sim_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    
    # now get the std data for plus and minus
    std_hip_moment_r_nat = 2*stdmomnat['hip_flexion_r_moment']
    # std_hip_moment_r_exo = 2*stdmomexo['hip_flexion_r_moment']
    std_knee_moment_r_nat = 2*stdmomnat['knee_angle_r_moment']
    # std_knee_moment_r_exo = 2*stdmomexo['knee_angle_r_moment']
    std_ankle_moment_r_nat = 2*stdmomnat['ankle_angle_r_moment']
    # std_ankle_moment_r_exo = 2*stdmomexo['ankle_angle_r_moment']
    std_hip_moment_l_nat = 2*stdmomnat['hip_flexion_l_moment']
    # std_hip_moment_l_exo = 2*stdmomexo['hip_flexion_l_moment']
    std_knee_moment_l_nat = 2*stdmomnat['knee_angle_l_moment']
    # std_knee_moment_l_exo = 2*stdmomexo['knee_angle_l_moment']
    std_ankle_moment_l_nat = 2*stdmomnat['ankle_angle_l_moment']
    # std_ankle_moment_l_exo = 2*stdmomexo['ankle_angle_l_moment']
    # now resample
    std_hip_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100,len(std_hip_moment_r_nat)), std_hip_moment_r_nat)
    # std_hip_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_r_exo)), std_hip_moment_r_exo)
    std_knee_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_nat)), std_knee_moment_r_nat)
    # std_knee_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_r_exo)), std_knee_moment_r_exo)
    std_ankle_moment_r_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_nat)), std_ankle_moment_r_nat)
    # std_ankle_moment_r_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_r_exo)), std_ankle_moment_r_exo)
    std_hip_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_nat)), std_hip_moment_l_nat)
    # std_hip_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_hip_moment_l_exo)), std_hip_moment_l_exo)
    std_knee_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_nat)), std_knee_moment_l_nat)
    # std_knee_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_knee_moment_l_exo)), std_knee_moment_l_exo)
    std_ankle_moment_l_nat = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_nat)), std_ankle_moment_l_nat)
    # std_ankle_moment_l_exo = np.interp(moment_xsim, np.linspace(0,100, len(std_ankle_moment_l_exo)), std_ankle_moment_l_exo)
    # and cut down to a step rather than the full gait cycle
    std_hip_moment_r_nat = std_hip_moment_r_nat[0:int(moment_simlen_nat/2)]
    # std_hip_moment_r_exo = std_hip_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_r_nat = std_knee_moment_r_nat[0:int(moment_simlen_nat/2)]
    # std_knee_moment_r_exo = std_knee_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_ankle_moment_r_nat = std_ankle_moment_r_nat[0:int(moment_simlen_nat/2)]
    # std_ankle_moment_r_exo = std_ankle_moment_r_exo[0:int(moment_simlen_nat/2)]
    std_hip_moment_l_nat = std_hip_moment_l_nat[0:int(moment_simlen_nat/2)]
    # std_hip_moment_l_exo = std_hip_moment_l_exo[0:int(moment_simlen_nat/2)]
    std_knee_moment_l_nat = std_knee_moment_l_nat[0:int(moment_simlen_nat/2)]
    # std_knee_moment_l_exo = std_knee_moment_l_exo[0:int(moment_simlen_nat/2)]   
    std_ankle_moment_l_nat = std_ankle_moment_l_nat[0:int(moment_simlen_nat/2)]
    # std_ankle_moment_l_exo = std_ankle_moment_l_exo[0:int(moment_simlen_nat/2)]
    

    ##########################################################################
    # first is the natural validation figure for saggital coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    # x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin_nat = np.linspace(0,100,int(kin_simlen_nat/2))
    x_mom = np.linspace(0, 100, moment_simlen_nat)[0:int(moment_simlen_nat/2)]
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    # Hip angle
    ax[0,0].plot(x_kin_nat, ref_hip_angle_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,0].plot(x_kin_nat, ref_hip_angle_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0, 0].plot(x_kin_nat, ref_hip_angle_exo, label='Exo Ref', color='purple')
    ax[0,0].fill_between(x_kin_nat, ref_hip_angle_r_nat - std_hip_angle_r_nat, ref_hip_angle_r_nat + std_hip_angle_r_nat, color='orange', alpha=0.2)
    ax[1,0].fill_between(x_kin_nat, ref_hip_angle_l_nat - std_hip_angle_l_nat, ref_hip_angle_l_nat + std_hip_angle_l_nat, color='orange', alpha=0.2)
    # ax[0, 0].fill_between(x_kin_nat, ref_hip_angle_exo - 2*std_exo['hip_flexion_r'], ref_hip_angle_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0,0].plot(x_kin_nat, sim_hip_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,0].plot(x_kin_nat, sim_hip_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0, 0].plot(x_kin_nat, sim_hip_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,0].legend(fontsize=12) #, loc='lower right')
    # other leg
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=12)
    # ax[1,0].legend(fontsize=12) #, loc='lower right')

    # Knee angle
    ax[0,1].plot(x_kin_nat, -ref_knee_angle_r_nat, label='Experimental reference', color='orange', linestyle='--')
    ax[1,1].plot(x_kin_nat, -ref_knee_angle_l_nat, label='Experimental reference', color='orange', linestyle='--')
    # ax[0,0].plot(x_kin_nat, ref_knee_angle_exo, label='Exo Ref', color='purple')
    ax[0,1].fill_between(x_kin_nat, -(ref_knee_angle_r_nat + std_knee_angle_r_nat), -(ref_knee_angle_r_nat - std_knee_angle_r_nat), color='orange', alpha=0.2)
    ax[1,1].fill_between(x_kin_nat, -(ref_knee_angle_l_nat + std_knee_angle_l_nat), -(ref_knee_angle_l_nat - std_knee_angle_l_nat), color='orange', alpha=0.2, label='2 Standard Deviations')
    # ax[0,0].fill_between(x_kin_nat, ref_knee_angle_exo - 2*std_exo['knee_angle_r'], ref_knee_angle_exo + 2*std_exo['knee_angle_r'], color='purple', alpha=0.2)
    ax[0,1].plot(x_kin_nat, sim_knee_angle_r_nat * 180 / np.pi, label='Simulated', color='orange', linewidth=3)
    ax[1,1].plot(x_kin_nat, sim_knee_angle_l_nat * 180 / np.pi, label='Simulated', color='orange', linewidth=3)
    # ax[0,0].plot(x_kin_nat, sim_knee_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    # ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    # ax[0,1].legend(fontsize=12) #, loc='upper left')
    # other leg
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    # ax[1,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10) #, loc='upper left')

    # Ankle angle
    ax[0,2].plot(x_kin_nat, ref_ankle_angle_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,2].plot(x_kin_nat, ref_ankle_angle_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0,0].plot(x_kin_nat, ref_ankle_angle_exo, label='Exo Ref', color='purple')
    ax[0,2].fill_between(x_kin_nat, ref_ankle_angle_r_nat - std_ankle_angle_r_nat, ref_ankle_angle_r_nat + std_ankle_angle_r_nat, color='orange', alpha=0.2)
    ax[1,2].fill_between(x_kin_nat, ref_ankle_angle_l_nat - std_ankle_angle_l_nat, ref_ankle_angle_l_nat + std_ankle_angle_l_nat, color='orange', alpha=0.2)
    # ax[0,0].fill_between(x_kin_nat, ref_ankle_angle_exo - 2*std_exo['ankle_angle_r'], ref_ankle_angle_exo + 2*std_exo['ankle_angle_r'], color='purple', alpha=0.2)
    ax[0,2].plot(x_kin_nat, sim_ankle_angle_r_nat * 180 / np.pi, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,2].plot(x_kin_nat, sim_ankle_angle_l_nat * 180 / np.pi, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0,0].plot(x_kin_nat, sim_ankle_angle_exo * 180 / np.pi, label='Exo Sim', color='purple', linestyle='--')
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    # ax[0,2].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,2].set_xlabel('Gait Cycle %', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    # ax[0,2].legend(fontsize=12) #, loc='upper right')
    # other leg
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    # ax[1,2].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)
    # ax[1,2].legend(fontsize=12) #, loc='upper right')
    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_nat_40.png')
    plt.show()


    ##########################################################################
    # second is the exotendon version of the validation figure for saggital coordinates.
    # this is going to plot both the natural and the exotendon coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin_exo = np.linspace(0, 100, kin_simlen_exo)[0:int(kin_simlen_exo/2)]
    x_kin_exo = np.linspace(0,100,int(kin_simlen_exo/2))
    x_mom = np.linspace(0, 100, moment_simlen_exo)[0:int(moment_simlen_exo/2)]
    x_grf = np.linspace(0, 100, grf_simlen_exo)[0:int(grf_simlen_exo/2)]
    # Hip angle
    # ax[0,0].plot(x_kin_exo, ref_hip_angle_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    # ax[1,0].plot(x_kin_exo, ref_hip_angle_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    # ax[0,0].fill_between(x_kin_exo, ref_hip_angle_r_exo - std_hip_angle_r_exo, ref_hip_angle_r_exo + std_hip_angle_r_exo, color='purple', alpha=0.2)
    # ax[1,0].fill_between(x_kin_exo, ref_hip_angle_l_exo - std_hip_angle_l_exo, ref_hip_angle_l_exo + std_hip_angle_l_exo, color='purple', alpha=0.2)
    ax[0,0].plot(x_kin_nat, sim_hip_angle_r_nat * 180 / np.pi, label='Nat Sim - Right', color='orange', linewidth=3)
    ax[0,0].plot(x_kin_exo, sim_hip_angle_r_exo * 180 / np.pi, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Angle (deg)', fontsize=12)
    
    ax[1,0].plot(x_kin_nat, sim_hip_angle_l_nat * 180 / np.pi, label='Nat Sim - Left', color='orange', linewidth=3)
    ax[1,0].plot(x_kin_exo, sim_hip_angle_l_exo * 180 / np.pi, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Angle (deg)', fontsize=12)
    
    # Knee angle
    # ax[0,1].plot(x_kin_exo, ref_knee_angle_r_exo, label='Experimental reference', color='purple', linestyle='--')
    # ax[1,1].plot(x_kin_exo, ref_knee_angle_l_exo, label='Experimental reference', color='purple', linestyle='--')
    # ax[0,1].fill_between(x_kin_exo, ref_knee_angle_r_exo - std_knee_angle_r_exo, ref_knee_angle_r_exo + std_knee_angle_r_exo, color='purple', alpha=0.2)
    # ax[1,1].fill_between(x_kin_exo, ref_knee_angle_l_exo - std_knee_angle_l_exo, ref_knee_angle_l_exo + std_knee_angle_l_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[0,1].plot(x_kin_nat, sim_knee_angle_r_nat * 180 / np.pi, label='Simulated', color='orange', linewidth=3)
    ax[0,1].plot(x_kin_exo, sim_knee_angle_r_exo * 180 / np.pi, label='Simulated', color='purple', linewidth=3)
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)

    ax[1,1].plot(x_kin_nat, sim_knee_angle_l_nat * 180 / np.pi, label='Natural simulation', color='orange', linewidth=3)
    ax[1,1].plot(x_kin_exo, sim_knee_angle_l_exo * 180 / np.pi, label='Exotendon simulation', color='purple', linewidth=3)
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10)
    # Ankle angle
    # ax[0,2].plot(x_kin_exo, ref_ankle_angle_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    # ax[1,2].plot(x_kin_exo, ref_ankle_angle_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    # ax[0,2].fill_between(x_kin_exo, ref_ankle_angle_r_exo - std_ankle_angle_r_exo, ref_ankle_angle_r_exo + std_ankle_angle_r_exo, color='purple', alpha=0.2)
    # ax[1,2].fill_between(x_kin_exo, ref_ankle_angle_l_exo - std_ankle_angle_l_exo, ref_ankle_angle_l_exo + std_ankle_angle_l_exo, color='purple', alpha=0.2)
    ax[0,2].plot(x_kin_nat, sim_ankle_angle_r_nat * 180 / np.pi, label='Exo Sim - Right', color='orange', linewidth=3)
    ax[0,2].plot(x_kin_exo, sim_ankle_angle_r_exo * 180 / np.pi, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)

    ax[1,2].plot(x_kin_nat, sim_ankle_angle_l_nat * 180 / np.pi, label='Nat Sim - Left', color='orange', linewidth=3)
    ax[1,2].plot(x_kin_exo, sim_ankle_angle_l_exo * 180 / np.pi, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_exo_40.png')
    plt.show()

    ##########################################################################
    # third is the grf data for natural 
    ##########################################################################
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    x_grf_nat = np.linspace(0,100,int(grf_simlen_nat/2))
    x_grf_exo = np.linspace(0,100,int(grf_simlen_exo/2))
    
    # Natural GRF y
    ax[0].plot(x_grf_nat, ref_grf_y_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[0].fill_between(x_grf_nat, ref_grf_y_nat - std_grf_y_nat, ref_grf_y_nat + std_grf_y_nat, color='orange', alpha=0.2)
    ax[0].plot(x_grf_nat, sim_grf_y_nat, label='Nat. Sim', color='orange', linewidth=3)
    # ax[0].plot(x_grf_exo, sim_grf_y_exo, label='Exo Sim', color='purple', linewidth=3)
    ax[0].set_title('Natural Superior(+) Vertical GRF', fontsize=12)
    ax[0].set_ylabel('Force (BW)', fontsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    # ax[0].legend(fontsize=12)
    # Natural GRF x
    ax[1].plot(x_grf_nat, ref_grf_x_nat, label='Experimental Reference', color='orange', linestyle='--')
    ax[1].fill_between(x_grf_nat, ref_grf_x_nat - std_grf_x_nat, ref_grf_x_nat + std_grf_x_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    ax[1].plot(x_grf_nat, sim_grf_x_nat, label='Natural Simulation', color='orange', linewidth=3)
    # ax[1].plot(x_grf_exo, sim_grf_x_exo, label='Exotendon Simulation', color='purple', linewidth=3)
    ax[1].set_title('Natural Anterior(+) Horizontal GRF', fontsize=12)
    # ax[1].set_ylabel('Force (BW)', fontsize=12)
    # ax[1].set_xlabel('Step %', fontsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].legend(fontsize=10)
    # Exo GRF y
    # # ax[1, 0].plot(x_grf_nat, ref_grf_y_exo, label='Exo Ref', color='purple')
    # # ax[1, 0].fill_between(x_grf_nat, ref_grf_y_exo - std_grf_y_exo, ref_grf_y_exo + std_grf_y_exo, color='purple', alpha=0.2)
    # ax[1, 0].plot(x_grf_nat, sim_grf_y_exo, label='Exo Sim', color='purple', linestyle='--')
    # ax[1, 0].set_title('Exo Vertical GRF', fontsize=12)
    # ax[1, 0].set_ylabel('Force (BW)', fontsize=12)
    # ax[1, 0].set_xlabel('Step %', fontsize=12)
    # ax[1, 0].tick_params(axis='y', labelsize=12)
    # ax[1, 0].tick_params(axis='x', labelsize=12)
    # # ax[1, 0].legend(fontsize=12)
    # # Exo GRF x
    # # ax[1, 1].plot(x_grf_nat, ref_grf_x_exo, label='Experimental Reference', color='purple')
    # # ax[1, 1].fill_between(x_grf_nat, ref_grf_x_exo - std_grf_x_exo, ref_grf_x_exo + std_grf_x_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    # ax[1, 1].plot(x_grf_nat, sim_grf_x_exo, label='Simulated', color='purple', linestyle='--')
    # ax[1, 1].set_title('Exo Horizontal GRF', fontsize=12)
    # # ax[1, 1].set_ylabel('Force (BW)', fontsize=12)
    # ax[1, 1].set_xlabel('Step %', fontsize=12)
    # ax[1, 1].tick_params(axis='y', labelsize=12)
    # ax[1, 1].tick_params(axis='x', labelsize=12)
    # ax[1, 1].legend(fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_grf_validation_40.png')
    plt.show()

    ##########################################################################
    # next is the grf data for both natural and exotendon.
    ##########################################################################
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    x_grf = np.linspace(0, 100, grf_simlen_nat)[0:int(grf_simlen_nat/2)]
    x_grf_nat = np.linspace(0,100,int(grf_simlen_nat/2))
    x_grf_exo = np.linspace(0,100,int(grf_simlen_exo/2))
    
    # Natural GRF y
    # ax[0].plot(x_grf_nat, ref_grf_y_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[0].fill_between(x_grf_nat, ref_grf_y_nat - std_grf_y_nat, ref_grf_y_nat + std_grf_y_nat, color='orange', alpha=0.2)
    ax[0].plot(x_grf_nat, sim_grf_y_nat, label='Nat. Sim', color='orange', linewidth=3)
    # ax[0].plot(x_grf_exo, sim_grf_y_exo, label='Exo Sim', color='purple', linewidth=3)
    ax[0].set_title('Natural Superior(+) Vertical GRF', fontsize=12)
    ax[0].set_ylabel('Force (BW)', fontsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    # ax[0].legend(fontsize=12)
    # Natural GRF x
    # ax[1].plot(x_grf_nat, ref_grf_x_nat, label='Experimental Reference', color='orange', linestyle='--')
    # ax[1].fill_between(x_grf_nat, ref_grf_x_nat - std_grf_x_nat, ref_grf_x_nat + std_grf_x_nat, color='orange', alpha=0.2, label='2 Standard Deviations')
    ax[1].plot(x_grf_nat, sim_grf_x_nat, label='Natural Simulation', color='orange', linewidth=3)
    # ax[1].plot(x_grf_exo, sim_grf_x_exo, label='Exotendon Simulation', color='purple', linewidth=3)
    ax[1].set_title('Natural Anterior(+) Horizontal GRF', fontsize=12)
    # ax[1].set_ylabel('Force (BW)', fontsize=12)
    # ax[1].set_xlabel('Step %', fontsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].legend(fontsize=10)
    # Exo GRF y
    # ax[1, 0].plot(x_grf_nat, ref_grf_y_exo, label='Exo Ref', color='purple')
    # ax[1, 0].fill_between(x_grf_nat, ref_grf_y_exo - std_grf_y_exo, ref_grf_y_exo + std_grf_y_exo, color='purple', alpha=0.2)
    ax[0].plot(x_grf_nat, sim_grf_y_exo, label='Exo Sim', color='purple', linewidth=3)
    ax[0].set_title('Exo Vertical GRF', fontsize=12)
    ax[0].set_ylabel('Force (BW)', fontsize=12)
    ax[0].set_xlabel('Step %', fontsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    # ax[0].legend(fontsize=12)
    # Exo GRF x
    # ax[1, 1].plot(x_grf_nat, ref_grf_x_exo, label='Experimental Reference', color='purple')
    # ax[1, 1].fill_between(x_grf_nat, ref_grf_x_exo - std_grf_x_exo, ref_grf_x_exo + std_grf_x_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[1].plot(x_grf_nat, sim_grf_x_exo, label='Simulated', color='purple', linewidth=3)
    ax[1].set_title('Exo Horizontal GRF', fontsize=12)
    # ax[1].set_ylabel('Force (BW)', fontsize=12)
    ax[1].set_xlabel('Step %', fontsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].legend(fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_grf_validation_exo_40.png')
    plt.show()

    ##########################################################################
    # fourth is the pelvis coordinates for natural 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin = np.linspace(0,100,int(kin_simlen_nat/2))

    # Pelvis ty
    ax[0, 0].plot(x_kin, ref_ty_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[0, 0].fill_between(x_kin, ref_ty_nat - std_ty_nat, ref_ty_nat + std_ty_nat, color='orange', alpha=0.2)
    ax[0, 0].plot(x_kin, sim_ty_nat, label='Nat. Sim', color='orange', linewidth=3)
    ax[0, 0].set_title('Pelvis Vertical Translation', fontsize=12)
    ax[0, 0].set_ylabel('Translation (m)', fontsize=12)
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].tick_params(axis='x', labelsize=12)
    # ax[0, 0].legend(fontsize=12)

    # # Pelvis tx
    # ax[0, 1].plot(x_kin, ref_tx_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[0, 1].fill_between(x_kin, ref_tx_nat - std_tx_nat, ref_tx_nat + std_tx_nat, color='orange', alpha=0.2)
    # ax[0, 1].plot(x_kin, sim_tx_nat, label='Nat. Sim', color='orange', linewidth=3)
    # ax[0, 1].set_title('Pelvis Horizontal Translation', fontsize=12)
    # # ax[0, 1].set_ylabel('Translation (m/height)', fontsize=12)
    # ax[0, 1].tick_params(axis='y', labelsize=12)
    # ax[0, 1].tick_params(axis='x', labelsize=12)
    # # ax[0, 1].legend(fontsize=12)

    # Pelvis list
    ax[1, 0].plot(x_kin, ref_pelvlist_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[1, 0].fill_between(x_kin, ref_pelvlist_nat - std_pelvlist_nat, ref_pelvlist_nat + std_pelvlist_nat, color='orange', alpha=0.2)
    ax[1, 0].plot(x_kin, sim_pelvlist_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 0].set_title('Pelvis List', fontsize=12)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='x', labelsize=12)
    ax[1, 0].set_xlabel('Step %', fontsize=12)
    # ax[1, 0].legend(fontsize=12)

    # Pelvis tilt
    ax[1, 1].plot(x_kin, ref_pelvtilt_nat, label='Nat. Ref', color='orange', linestyle='--')
    ax[1, 1].fill_between(x_kin, ref_pelvtilt_nat - std_pelvtilt_nat, ref_pelvtilt_nat + std_pelvtilt_nat, color='orange', alpha=0.2)
    ax[1, 1].plot(x_kin, sim_pelvtilt_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 1].set_title('Pelvis Tilt', fontsize=12)
    # ax[1, 1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='x', labelsize=12)
    ax[1, 1].set_xlabel('Step %', fontsize=12)
    # ax[1, 1].legend(fontsize=10)

    # Pelvis rotation
    ax[0,1].plot(x_kin, ref_pelvrot_nat, label='Natural Reference', color='orange', linestyle='--')
    ax[0,1].fill_between(x_kin, ref_pelvrot_nat - std_pelvrot_nat, ref_pelvrot_nat + std_pelvrot_nat, color='orange', alpha=0.2)
    ax[0,1].plot(x_kin, sim_pelvrot_nat*180/np.pi, label='Natural Simulation', color='orange', linewidth=3)
    ax[0,1].set_title('Pelvis Rotation', fontsize=12)
    ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='x', labelsize=12)    
    ax[0,1].set_xlabel('Step %', fontsize=12)
    # ax[0,1].legend(fontsize=10)

    # Hide the last subplot and use it to display the legend   
    ax[0,2].axis('off')
    ax[1,2].axis('off')
    # get the legend labels from the previous subplot
    handles, labels = ax[0, 1].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[0, 2].legend(handles, labels, loc='center left', fontsize=10)


    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_pelvis_validation_nat_40.png')
    plt.show()


    ##########################################################################
    # fifth is the pelvis coordinates for exotendon cose, plotted with the natural 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_kin = np.linspace(0, 100, kin_simlen_nat)[0:int(kin_simlen_nat/2)]
    x_kin_nat = np.linspace(0,100,int(kin_simlen_nat/2))
    x_kin_exo = np.linspace(0,100,int(kin_simlen_exo/2))

    # Pelvis ty
    # ax[0, 0].plot(x_kin_nat, ref_ty_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[0, 0].fill_between(x_kin_nat, ref_ty_nat - std_ty_nat, ref_ty_nat + std_ty_nat, color='orange', alpha=0.2)
    ax[0, 0].plot(x_kin_nat, sim_ty_nat, label='Nat Sim', color='orange', linewidth=3)
    ax[0, 0].plot(x_kin_exo, sim_ty_exo, label='Exo Sim', color='purple', linewidth=3)
    ax[0, 0].set_title('Pelvis Vertical Translation', fontsize=12)
    ax[0, 0].set_ylabel('Translation (m)', fontsize=12)
    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 0].tick_params(axis='x', labelsize=12)
    # ax[0, 0].legend(fontsize=12)

    # # Pelvis tx
    # ax[0, 1].plot(x_kin_nat, ref_tx_nat, label='Nat. Ref', color='orange', linestyle='--')
    # # ax[0, 1].fill_between(x_kin_nat, ref_tx_nat - std_tx_nat, ref_tx_nat + std_tx_nat, color='orange', alpha=0.2)
    # ax[0, 1].plot(x_kin_nat, sim_tx_nat, label='Nat. Sim', color='orange', linewidth=3)
    # ax[0, 1].plot(x_kin_exo, sim_tx_exo, label='Exo. Sim', color='purple', linewidth=3)
    # ax[0, 1].set_title('Pelvis Horizontal Translation', fontsize=12)
    # # ax[0, 1].set_ylabel('Translation (m/height)', fontsize=12)
    # ax[0, 1].tick_params(axis='y', labelsize=12)
    # ax[0, 1].tick_params(axis='x', labelsize=12)
    # # ax[0, 1].legend(fontsize=12)

    # Pelvis list
    # ax[1, 0].plot(x_kin_nat, ref_pelvlist_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[1, 0].fill_between(x_kin_nat, ref_pelvlist_nat - std_pelvlist_nat, ref_pelvlist_nat + std_pelvlist_nat, color='orange', alpha=0.2)
    ax[1, 0].plot(x_kin_nat, sim_pelvlist_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 0].plot(x_kin_exo, sim_pelvlist_exo*180/np.pi, label='Exo. Sim', color='purple', linewidth=3)
    ax[1, 0].set_title('Pelvis List', fontsize=12)
    ax[1, 0].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='x', labelsize=12)
    ax[1, 0].set_xlabel('Step %', fontsize=12)
    # ax[1, 0].legend(fontsize=12)

    # Pelvis tilt
    # ax[1, 1].plot(x_kin_nat, ref_pelvtilt_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[1, 1].fill_between(x_kin_nat, ref_pelvtilt_nat - std_pelvtilt_nat, ref_pelvtilt_nat + std_pelvtilt_nat, color='orange', alpha=0.2)
    ax[1, 1].plot(x_kin_nat, sim_pelvtilt_nat*180/np.pi, label='Nat. Sim', color='orange', linewidth=3)
    ax[1, 1].plot(x_kin_exo, sim_pelvtilt_exo*180/np.pi, label='Exo. Sim', color='purple', linewidth=3)
    ax[1, 1].set_title('Pelvis Tilt', fontsize=12)
    # ax[1, 1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='x', labelsize=12)
    ax[1, 1].set_xlabel('Step %', fontsize=12)
    # ax[1, 1].legend(fontsize=12)

    # Pelvis rotation
    # ax[0,1].plot(x_kin_nat, ref_pelvrot_nat, label='Nat. Ref', color='orange', linestyle='--')
    # ax[0,1].fill_between(x_kin_nat, ref_pelvrot_nat - std_pelvrot_nat, ref_pelvrot_nat + std_pelvrot_nat, color='orange', alpha=0.2)
    ax[0,1].plot(x_kin_nat, sim_pelvrot_nat*180/np.pi, label='Natural simulation', color='orange', linewidth=3)
    ax[0,1].plot(x_kin_exo, sim_pelvrot_exo*180/np.pi, label='Exotendon simulation', color='purple', linewidth=3)
    ax[0,1].set_title('Pelvis Rotation', fontsize=12)
    ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)
    ax[0,1].tick_params(axis='x', labelsize=12)    
    ax[0,1].set_xlabel('Step %', fontsize=12)
    # ax[0,1].legend(fontsize=12)
    
    # Hide the last subplot and use it to display the legend   
    ax[0,2].axis('off')
    ax[1,2].axis('off')
    # get the legend labels from the previous subplot
    handles, labels = ax[0, 1].get_legend_handles_labels()
    # display the legend entries in the last subplot 
    ax[0, 2].legend(handles, labels, loc='center left', fontsize=10)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_pelvis_validation_exo_40.png')
    plt.show()
    

    ##### now the moments. 
    ##########################################################################
    # first is the natural validation figure for saggital coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    x_mom_nat = np.linspace(0,100,int(moment_simlen_nat/2))
    x_mom_exo = np.linspace(0,100,int(moment_simlen_exo/2))

    # Hip angle
    ax[0,0].plot(x_mom_nat, ref_hip_moment_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,0].plot(x_mom_nat, ref_hip_moment_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0, 0].plot(x_mom_nat, ref_hip_moment_exo, label='Exo Ref', color='purple')
    ax[0,0].fill_between(x_mom_nat, ref_hip_moment_r_nat - std_hip_moment_r_nat, ref_hip_moment_r_nat + std_hip_moment_r_nat, color='orange', alpha=0.2)
    ax[1,0].fill_between(x_mom_nat, ref_hip_moment_l_nat - std_hip_moment_l_nat, ref_hip_moment_l_nat + std_hip_moment_l_nat, color='orange', alpha=0.2)
    # ax[0, 0].fill_between(x_mom_nat, ref_hip_moment_exo - 2*std_exo['hip_flexion_r'], ref_hip_moment_exo + 2*std_exo['hip_flexion_r'], color='purple', alpha=0.2)
    ax[0,0].plot(x_mom_nat, sim_hip_moment_r_nat, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,0].plot(x_mom_nat, sim_hip_moment_l_nat, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0, 0].plot(x_mom_nat, sim_hip_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    # ax[0,0].legend(fontsize=12) #, loc='lower right')
    # other leg
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    # ax[1,0].legend(fontsize=12) #, loc='lower right')

    # Knee angle
    ax[0,1].plot(x_mom_nat, -ref_knee_moment_r_nat, label='Experimental reference', color='orange', linestyle='--')
    ax[1,1].plot(x_mom_nat, -ref_knee_moment_l_nat, label='Experimental reference', color='orange', linestyle='--')
    # ax[0,0].plot(x_mom_nat, ref_knee_moment_exo, label='Exo Ref', color='purple')
    ax[0,1].fill_between(x_mom_nat, -(ref_knee_moment_r_nat + std_knee_moment_r_nat), -(ref_knee_moment_r_nat - std_knee_moment_r_nat), color='orange', alpha=0.2)
    ax[1,1].fill_between(x_mom_nat, -(ref_knee_moment_l_nat + std_knee_moment_l_nat), -(ref_knee_moment_l_nat - std_knee_moment_l_nat), color='orange', alpha=0.2, label='2 Standard Deviations')
    # ax[0,0].fill_between(x_mom_nat, ref_knee_moment_exo - 2*std_exo['knee_moment_r'], ref_knee_moment_exo + 2*std_exo['knee_moment_r'], color='purple', alpha=0.2)
    ax[0,1].plot(x_mom_nat, sim_knee_moment_r_nat, label='Simulated', color='orange', linewidth=3)
    ax[1,1].plot(x_mom_nat, sim_knee_moment_l_nat, label='Simulated', color='orange', linewidth=3)
    # ax[0,0].plot(x_mom_nat, sim_knee_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    # ax[0,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)
    # ax[0,1].legend(fontsize=12) #, loc='upper left')
    # other leg
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    # ax[1,1].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10) #, loc='upper left')

    # Ankle angle
    ax[0,2].plot(x_mom_nat, ref_ankle_moment_r_nat, label='Nat. Ref - Right', color='orange', linestyle='--')
    ax[1,2].plot(x_mom_nat, ref_ankle_moment_l_nat, label='Nat. Ref - Left', color='orange', linestyle='--')
    # ax[0,0].plot(x_mom_nat, ref_ankle_moment_exo, label='Exo Ref', color='purple')
    ax[0,2].fill_between(x_mom_nat, ref_ankle_moment_r_nat - std_ankle_moment_r_nat, ref_ankle_moment_r_nat + std_ankle_moment_r_nat, color='orange', alpha=0.2)
    ax[1,2].fill_between(x_mom_nat, ref_ankle_moment_l_nat - std_ankle_moment_l_nat, ref_ankle_moment_l_nat + std_ankle_moment_l_nat, color='orange', alpha=0.2)
    # ax[0,0].fill_between(x_mom_nat, ref_ankle_moment_exo - 2*std_exo['ankle_moment_r'], ref_ankle_moment_exo + 2*std_exo['ankle_moment_r'], color='purple', alpha=0.2)
    ax[0,2].plot(x_mom_nat, sim_ankle_moment_r_nat, label='Nat. Sim - Right', color='orange', linewidth=3)
    ax[1,2].plot(x_mom_nat, sim_ankle_moment_l_nat, label='Nat. Sim - Left', color='orange', linewidth=3)
    # ax[0,0].plot(x_mom_nat, sim_ankle_moment_exo, label='Exo Sim', color='purple', linestyle='--')
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    # ax[0,2].set_ylabel('Angle (deg)', fontsize=12)
    # ax[0,2].set_xlabel('Gait Cycle %', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)
    # ax[0,2].legend(fontsize=12) #, loc='upper right')
    # other leg
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    # ax[1,2].set_ylabel('Angle (deg)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)
    # ax[1,2].legend(fontsize=12) #, loc='upper right')
    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_moment_nat_40.png')
    plt.show()


    ##########################################################################
    # second is the exotendon version of the validation figure for saggital coordinates.
    # this is going to plot both the natural and the exotendon coordinates. 
    ##########################################################################
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300)

    # Hip angle
    # ax[0,0].plot(x_kin_exo, ref_hip_moment_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    # ax[1,0].plot(x_kin_exo, ref_hip_moment_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    # ax[0,0].fill_between(x_kin_exo, ref_hip_moment_r_exo - std_hip_moment_r_exo, ref_hip_moment_r_exo + std_hip_moment_r_exo, color='purple', alpha=0.2)
    # ax[1,0].fill_between(x_kin_exo, ref_hip_moment_l_exo - std_hip_moment_l_exo, ref_hip_moment_l_exo + std_hip_moment_l_exo, color='purple', alpha=0.2)
    ax[0,0].plot(x_mom_nat, sim_hip_moment_r_nat, label='Nat Sim - Right', color='orange', linewidth=3)
    ax[0,0].plot(x_mom_exo, sim_hip_moment_r_exo, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[0,0].set_title('R Hip Flexion (+)', fontsize=12)
    ax[0,0].tick_params(axis='y', labelsize=12)    
    ax[0,0].tick_params(axis='x', labelsize=12)
    ax[0,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    
    ax[1,0].plot(x_mom_nat, sim_hip_moment_l_nat, label='Nat Sim - Left', color='orange', linewidth=3)
    ax[1,0].plot(x_mom_exo, sim_hip_moment_l_exo, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[1,0].set_title('L Hip Flexion (+)', fontsize=12)
    ax[1,0].tick_params(axis='y', labelsize=12)    
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel('Step %', fontsize=12)
    ax[1,0].set_ylabel('Moment (Nm/kg)', fontsize=12)
    
    # Knee angle
    # ax[0,1].plot(x_mom_exo, ref_knee_moment_r_exo, label='Experimental reference', color='purple', linestyle='--')
    # ax[1,1].plot(x_mom_exo, ref_knee_moment_l_exo, label='Experimental reference', color='purple', linestyle='--')
    # ax[0,1].fill_between(x_mom_exo, ref_knee_moment_r_exo - std_knee_moment_r_exo, ref_knee_moment_r_exo + std_knee_moment_r_exo, color='purple', alpha=0.2)
    # ax[1,1].fill_between(x_mom_exo, ref_knee_moment_l_exo - std_knee_moment_l_exo, ref_knee_moment_l_exo + std_knee_moment_l_exo, color='purple', alpha=0.2, label='2 Standard Deviations')
    ax[0,1].plot(x_mom_nat, sim_knee_moment_r_nat, label='Simulated', color='orange', linewidth=3)
    ax[0,1].plot(x_mom_exo, sim_knee_moment_r_exo, label='Simulated', color='purple', linewidth=3)
    ax[0,1].set_title('R Knee Flexion (+)', fontsize=12)
    ax[0,1].tick_params(axis='y', labelsize=12)    
    ax[0,1].tick_params(axis='x', labelsize=12)

    ax[1,1].plot(x_mom_nat, sim_knee_moment_l_nat, label='Natural simulation', color='orange', linewidth=3)
    ax[1,1].plot(x_mom_exo, sim_knee_moment_l_exo, label='Exotendon simulation', color='purple', linewidth=3)
    ax[1,1].set_title('L Knee Flexion (+)', fontsize=12)
    ax[1,1].tick_params(axis='y', labelsize=12)    
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,1].set_xlabel('Step %', fontsize=12)
    ax[1,1].legend(fontsize=10)
    # Ankle angle
    # ax[0,2].plot(x_mom_exo, ref_ankle_moment_r_exo, label='Exo Ref - Right', color='purple', linestyle='--')
    # ax[1,2].plot(x_mom_exo, ref_ankle_moment_l_exo, label='Exo Ref - Left', color='purple', linestyle='--')
    # ax[0,2].fill_between(x_mom_exo, ref_ankle_moment_r_exo - std_ankle_moment_r_exo, ref_ankle_moment_r_exo + std_ankle_moment_r_exo, color='purple', alpha=0.2)
    # ax[1,2].fill_between(x_mom_exo, ref_ankle_moment_l_exo - std_ankle_moment_l_exo, ref_ankle_moment_l_exo + std_ankle_moment_l_exo, color='purple', alpha=0.2)
    ax[0,2].plot(x_mom_nat, sim_ankle_moment_r_nat, label='Exo Sim - Right', color='orange', linewidth=3)
    ax[0,2].plot(x_mom_exo, sim_ankle_moment_r_exo, label='Exo Sim - Right', color='purple', linewidth=3)
    ax[0,2].set_title('R Ankle Dorsiflexion (+)', fontsize=12)
    ax[0,2].tick_params(axis='y', labelsize=12)    
    ax[0,2].tick_params(axis='x', labelsize=12)

    ax[1,2].plot(x_mom_nat, sim_ankle_moment_l_nat, label='Nat Sim - Left', color='orange', linewidth=3)
    ax[1,2].plot(x_mom_exo, sim_ankle_moment_l_exo, label='Exo Sim - Left', color='purple', linewidth=3)
    ax[1,2].set_title('L Ankle Dorsiflexion (+)', fontsize=12)
    ax[1,2].set_xlabel('Step %', fontsize=12)
    ax[1,2].tick_params(axis='y', labelsize=12)    
    ax[1,2].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    figpath = 'G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\'
    plt.savefig(figpath + 'figure_saggitalvalidation_moment_exo_40.png')
    plt.show()

    return

# function that takes the grf and met files, and find the indices that are stance, pre stance, and post stance. 
def findStanceIndices(natmet, exomet, GRFnat, GRFexo):
    partindices = {}
    # load the GRF results
    natGRF = osim.TimeSeriesTable(GRFnat)
    exoGRF = osim.TimeSeriesTable(GRFexo)
    # get the time vectors
    natGRFtime = np.array(natGRF.getIndependentColumn())
    exoGRFtime = np.array(exoGRF.getIndependentColumn())
    # get the vertical forces
    natGRFy = natGRF.getDependentColumn('ground_force_r_vy').to_numpy()
    exoGRFy = exoGRF.getDependentColumn('ground_force_r_vy').to_numpy()
    # Find the peak in the first half of the vector nat
    half_index_nat = len(natGRFy) // 2
    first_half_nat = natGRFy[:half_index_nat]
    peak_index_nat = np.argmax(first_half_nat)
    threshold = 0.5
    # Find the point where the vector first rises above the threshold before the peak
    start_of_hump_index_nat = np.where(natGRFy[:peak_index_nat] > threshold)[0][0]
    return_to_zero_index_nat = peak_index_nat + np.where(natGRFy[peak_index_nat:] < threshold)[0][0]
    partindices['start_of_hump_index_nat'] = start_of_hump_index_nat
    partindices['return_to_zero_index_nat'] = return_to_zero_index_nat
    # Split the vector into three parts
    partindices['part_before_hump_naty'] = natGRFy[:start_of_hump_index_nat]
    partindices['part_main_hump_naty'] = natGRFy[start_of_hump_index_nat:return_to_zero_index_nat + 1]
    partindices['part_after_hump_naty'] = natGRFy[return_to_zero_index_nat + 1:]
    partindices['part_before_hump_nattime'] = natGRFtime[:start_of_hump_index_nat]
    partindices['part_main_hump_nattime'] = natGRFtime[start_of_hump_index_nat:return_to_zero_index_nat + 1]
    partindices['part_after_hump_nattime'] = natGRFtime[return_to_zero_index_nat + 1:]
    # Find the peak in the first half of the vector exo
    half_index_exo = len(exoGRFy) // 2
    first_half_exo = exoGRFy[:half_index_exo]
    peak_index_exo = np.argmax(first_half_exo)
    threshold = 0.5
    # Find the point where the vector first rises above the threshold before the peak
    start_of_hump_index_exo = np.where(exoGRFy[:peak_index_exo] > threshold)[0][0]
    return_to_zero_index_exo = peak_index_exo + np.where(exoGRFy[peak_index_exo:] < threshold)[0][0]
    partindices['start_of_hump_index_exo'] = start_of_hump_index_exo
    partindices['return_to_zero_index_exo'] = return_to_zero_index_exo
    # Split the vector into three parts
    partindices['part_before_hump_exoy'] = exoGRFy[:start_of_hump_index_exo]
    partindices['part_main_hump_exoy'] = exoGRFy[start_of_hump_index_exo:return_to_zero_index_exo + 1]
    partindices['part_after_hump_exoy'] = exoGRFy[return_to_zero_index_exo + 1:]
    partindices['part_before_hump_exotime'] = exoGRFtime[:start_of_hump_index_exo]
    partindices['part_main_hump_exotime'] = exoGRFtime[start_of_hump_index_exo:return_to_zero_index_exo + 1]
    partindices['part_after_hump_exotime'] = exoGRFtime[return_to_zero_index_exo + 1:]

    return partindices, natGRFtime

# create function for nice metabolics validation figure.
def metabolicValidationFigure27(natmet, exomet, modelfile, GRFnat, GRFexo):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    partindices, natGRFtime = findStanceIndices(natmet, exomet, GRFnat, GRFexo)
    # load the metabolic results
    natmet = osim.TimeSeriesTable(natmet)
    exomet = osim.TimeSeriesTable(exomet)
    # check that the time vectors are the same
    natmettime = np.array(natmet.getIndependentColumn())
    exomettime = np.array(exomet.getIndependentColumn())
    if not np.array_equal(natmettime, natGRFtime):
        print('Time vectors do not match')
        return
    # get the total metabolic rate for each phase
    total_stance_nat = np.zeros((len(partindices['part_main_hump_nattime']),9))
    total_stance_exo = np.zeros((len(partindices['part_main_hump_exotime']),9))
    total_swing_nat = np.zeros((len(partindices['part_after_hump_nattime']),9))
    total_swing_exo = np.zeros((len(partindices['part_after_hump_exotime']),9))
    count = 0
    musclenames = []
    for each in natmet.getColumnLabels():
        if 'total_metabolic_rate' in each and '_r_' in each:
            shortname = each.split('_r_')[0][1:]; musclenames.append(shortname)
            # get the data
            natdata = natmet.getDependentColumn(each).to_numpy()
            exodata = exomet.getDependentColumn(each).to_numpy()
            # get the stance and swing phases
            natstance = natdata[partindices['start_of_hump_index_nat']:partindices['return_to_zero_index_nat'] + 1]
            exostance = exodata[partindices['start_of_hump_index_exo']:partindices['return_to_zero_index_exo'] + 1]
            natswing = natdata[partindices['return_to_zero_index_nat'] + 1:]
            exoswing = exodata[partindices['return_to_zero_index_exo'] + 1:]
            total_stance_nat[:,count] = natstance
            total_stance_exo[:,count] = exostance
            total_swing_nat[:,count] = natswing
            total_swing_exo[:,count] = exoswing
            count += 1
    # compute stance and swing for each muscle costs individually, and print... 
    # for nat and for exo
    natstanceind = {}
    exostanceind = {}
    natswingind = {}
    exoswingind = {}
    for m in musclenames:
        natstanceind[m] = np.trapz(total_stance_nat[:,musclenames.index(m)], x=partindices['part_main_hump_nattime']) / (partindices['part_main_hump_nattime'][-1] - partindices['part_main_hump_nattime'][0]) / modelmass
        exostanceind[m] = np.trapz(total_stance_exo[:,musclenames.index(m)], x=partindices['part_main_hump_exotime']) / (partindices['part_main_hump_exotime'][-1] - partindices['part_main_hump_exotime'][0]) / modelmass
        natswingind[m] = np.trapz(total_swing_nat[:,musclenames.index(m)], x=partindices['part_after_hump_nattime']) / (partindices['part_after_hump_nattime'][-1] - partindices['part_after_hump_nattime'][0]) / modelmass
        exoswingind[m] = np.trapz(total_swing_exo[:,musclenames.index(m)], x=partindices['part_after_hump_exotime']) / (partindices['part_after_hump_exotime'][-1] - partindices['part_after_hump_exotime'][0]) / modelmass
    # Create a list to store the data
    datastanceind = []
    dataswingind = []
    # Populate the list with muscle names, differences, and percent changes
    for muscle in natstanceind.keys():
        difference_stance = exostanceind[muscle] - natstanceind[muscle]
        percent_change_stance = (difference_stance / natstanceind[muscle]) * 100
        datastanceind.append([muscle, difference_stance, percent_change_stance])
        difference_swing = exoswingind[muscle] - natswingind[muscle]
        percent_change_swing = (difference_swing / natswingind[muscle]) * 100
        dataswingind.append([muscle, difference_swing, percent_change_swing])
    # Create a DataFrame from the data
    dfstanceind = pd.DataFrame(datastanceind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    dfswingind = pd.DataFrame(dataswingind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    # Sort the DataFrame by the 'Difference (W/kg)' column
    df_sorted_stanceind = dfstanceind.sort_values(by='Difference (W/kg)', ascending=True)
    df_sorted_swingind = dfswingind.sort_values(by='Difference (W/kg)', ascending=True)
    # sum the metabolic rates for each phase
    total_stance_nat = np.sum(total_stance_nat, axis=1)
    total_stance_exo = np.sum(total_stance_exo, axis=1)
    total_swing_nat = np.sum(total_swing_nat, axis=1)
    total_swing_exo = np.sum(total_swing_exo, axis=1)
    natstanceavg = np.trapz(total_stance_nat, x=partindices['part_main_hump_nattime']) / (partindices['part_main_hump_nattime'][-1] - partindices['part_main_hump_nattime'][0]) / modelmass
    exostanceavg = np.trapz(total_stance_exo, x=partindices['part_main_hump_exotime']) / (partindices['part_main_hump_exotime'][-1] - partindices['part_main_hump_exotime'][0]) / modelmass
    natswingavg = np.trapz(total_swing_nat, x=partindices['part_after_hump_nattime']) / (partindices['part_after_hump_nattime'][-1] - partindices['part_after_hump_nattime'][0]) / modelmass
    exoswingavg = np.trapz(total_swing_exo, x=partindices['part_after_hump_exotime']) / (partindices['part_after_hump_exotime'][-1] - partindices['part_after_hump_exotime'][0]) / modelmass
    # get the average metabolic rate for the whole body
    natmetavg = np.trapz(natmet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy(), x=natmettime) / (natmettime[-1] - natmettime[0]) / modelmass
    exometavg = np.trapz(exomet.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy(), x=exomettime) / (exomettime[-1] - exomettime[0]) / modelmass
    # hard code the experimental values and values from the previous papers. metaboliccostcomparisons_rev2.xlsx
    # experimental values for whole body - from my experiment, not Simpson et al.
    expnatavg = 10.68
    expnatstd = 0.88
    expexoavg = 9.78
    expexostd = 0.69

    # compute the confidence interval for 

    perc_change_exp = ((expexoavg - expnatavg) / expnatavg) * 100
    # previous simulation whole body
    prevnatavg = 11.62
    prevnatstd = 0.95
    prevexoavg = 10.21
    prevexostd = 0.80
    # previous simulation stance values
    prevnatstance = 8.81
    prevnatstancestd = 1.05
    prevexostance = 7.31
    prevexostancestd = 1.05
    # previous simulations swing data
    prevnatswing = 3.90
    prevnatswingstd = 0.47
    prevexoswing = 3.60
    prevexoswingstd = 0.34
    ## okay we should have all the values now to make the figure
    # Create a figure and axis
    # Define the y values for the bars
    # Define the y values for the previous data bars
    y_prev_nat = [prevnatavg, prevnatstance, prevnatswing]
    y_prev_natstd = [prevnatstd, prevnatstancestd, prevnatswingstd]
    y_prev_exo = [prevexoavg, prevexostance, prevexoswing]
    y_prev_exostd = [prevexostd, prevexostancestd, prevexoswingstd]
    y_nat = [natmetavg, natstanceavg, natswingavg]
    y_exo = [exometavg, exostanceavg, exoswingavg]
    # create the values for the experimental data
    y_expnat = [expnatavg, 0, 0]
    y_expnatstd = [expnatstd, 0, 0]
    y_expexo = [expexoavg, 0, 0]
    y_expexostd = [expexostd, 0, 0]

    # Define the x values for the bars
    x = np.arange(3)
    # # Create a second subplot for percent change
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    # width = 0.15
    # wantexp = True
    # # First subplot: Comparison of Metabolic Rates
    # if wantexp:
    #     # ax1.bar(x - 2.5*width, y_expnat, width, label='Experimental Nat.', color='#a6cee3', yerr=y_expnatstd, capsize=5)
    #     # ax1.bar(x - 1.5*width, y_expexo, width, label='Experimental Exo.', color='#1f78b4', yerr=y_expexostd, capsize=5)
    #     # ax1.bar(x - 0.5*width, y_nat, width, label='Current simulation Nat.', color='#e66101')
    #     # ax1.bar(x + 0.5*width, y_exo, width, label='Current simulation Exo.', color='#5e3c99')
    #     # ax1.bar(x + 1.5*width, y_prev_nat, width, label='Previous simulation Nat.', color='#fdb863', yerr=y_prev_natstd, capsize=5)
    #     # ax1.bar(x + 2.5*width, y_prev_exo, width, label='Previous Exo. Simulations', color='#b2abd2', yerr=y_prev_exostd, capsize=5)
    #     ax1.bar(x[0] - 2.5*width, y_expnat[0], width, label='Experimental Nat.', color='#b35806', yerr=y_expnatstd[0], capsize=5)
    #     ax1.bar(x - 1.5*width, y_prev_nat, width, label='Previous simulation Nat.', color='#f1a340', yerr=y_prev_natstd, capsize=5)
    #     ax1.bar(x - 0.5*width, y_nat, width, label='Current simulation Nat.', color='#fee0b6')
    #     ax1.bar(x[0] + 0.5*width, y_expexo[0], width, label='Experimental Exo.', color='#542788', yerr=y_expexostd[0], capsize=5)
    #     ax1.bar(x + 1.5*width, y_prev_exo, width, label='Previous Exo. Simulations', color='#998ec3', yerr=y_prev_exostd, capsize=5)
    #     ax1.bar(x + 2.5*width, y_exo, width, label='Current simulation Exo.', color='#d8daeb')
    # else:
    #     # ax1.bar(x - 2.5*width, y_expnat, width, label='Experimental Nat.', color='#a6cee3', yerr=y_expnatstd, capsize=5)
    #     # ax1.bar(x - 1.5*width, y_expexo, width, label='Experimental Exo.', color='#1f78b4', yerr=y_expexostd, capsize=5)
    #     ax1.bar(x - 1.5*width, y_nat, width, label='Current simulation Nat.', color='#e66101')
    #     ax1.bar(x - 0.5*width, y_exo, width, label='Current simulation Exo.', color='#5e3c99')
    #     ax1.bar(x + 0.5*width, y_prev_nat, width, label='Previous simulation Nat.', color='#fdb863', yerr=y_prev_natstd, capsize=5)
    #     ax1.bar(x + 1.5*width, y_prev_exo, width, label='Previous Exo. Simulations', color='#b2abd2', yerr=y_prev_exostd, capsize=5)
    # ax1.bar([], [], yerr=2, capsize=5, error_kw=dict(label='Standard Dev.'))
    # ax1.set_xlabel('Phases', fontsize=14)
    # ax1.set_ylabel('Metabolic Rate (W/kg)', fontsize=14)
    # ax1.set_title('Comparison of Metabolic Rates', fontsize=14)
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(['Full Stride', 'Stance', 'Swing'], fontsize=14)
    # ax1.tick_params(axis='y', labelsize=14)
    # ax1.legend(fontsize=12)
    # ax1.set_ylim(0, 16)
    # # now set up the percent change standard deviations from the previous data - metaboliccostcomparisons_rev2
    prev_perc = -11.97
    prev_perc_std = 4.76
    prev_perc_stance = -12.83
    prev_perc_stance_std = 3.11
    prev_perc_swing = -2.51
    prev_perc_swing_std = 2.44
    # Calculate percent change
    # percent_change_prev = [(exo - nat) / y_prev_nat[0] * 100 for exo, nat in zip(y_prev_exo, y_prev_nat)]
    percent_change_current = [(exo - nat) / y_nat[0] * 100 for exo, nat in zip(y_exo, y_nat)]
    percent_change_prev = [prev_perc, prev_perc_stance, prev_perc_swing]
    percent_change_prev_std = [prev_perc_std, prev_perc_stance_std, prev_perc_swing_std]
    percent_change_exp = [(exo - nat) / y_expnat[0] * 100 for exo, nat in zip(y_expexo, y_expnat)]
    percent_change_exp_std = [(exo - nat) / y_expnat[0] * 100 for exo, nat in zip(y_expexostd, y_expnatstd)]
    # # Second subplot: Percent Change
    # width = 0.15
    # if wantexp:
    #     ax2.bar(x-width, percent_change_exp, width, label='Experimental', color='#a6cee3')
    #     ax2.bar(x, percent_change_prev, width, label='Previous Sim.', color='#018571', yerr=percent_change_prev_std, capsize=5)
    #     ax2.bar(x+width, percent_change_current, width, label='Current Sim.', color='#dfc27d')
    # else: 
    #     # ax2.bar(x, percent_change_exp, width, label='Experimental', color='#a6cee3')
    #     ax2.bar(x + 0.5*width, percent_change_prev, width, label='Previous Sim.', color='#018571', yerr=percent_change_prev_std, capsize=5)
    #     ax2.bar(x - 0.5*width, percent_change_current, width, label='Current Sim.', color='#dfc27d')
    # ax2.set_xlabel('Phases', fontsize=14)
    # ax2.set_ylabel('Percent Change (%)', fontsize=14)
    # ax2.set_title('Percent Change wrt. Whole body Natural Metabolic Rate', fontsize=14)
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(['Full Stride', 'Stance', 'Swing'], fontsize=14)
    # ax2.tick_params(axis='y', labelsize=14)
    # ax2.bar([], [], yerr=2, capsize=5, error_kw=dict(label='Standard Dev.'))
    # ax2.legend(fontsize=12, loc='lower right')
    # # Add a horizontal line at y=0 for the x-axis
    # for ax in [ax1, ax2]:
    #     ax.axhline(0, color='black', linewidth=0.8)
    # # Show the plot
    # ax2.set_position([0.55, 0.15, 0.65, 0.7])  # [left, bottom, width, height]
    # plt.tight_layout()
    # plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_metabolicvalidation_27.png')
    # plt.show()

    #########################
    new_perc_change_current = (y_exo[0] - y_nat[0]) / y_nat[0] * 100

    # Compute some error metrics for the metabolics 
    exp_vals_nat = [9.4, 9.8, 10.9, 10.3, 11.1, 12.3, 10.9]
    exp_vals_exo = [9.0, 9.5, 9.4, 9.4, 9.8, 11.3, 10.1]
    persub_perc_change_exp = [(exo - nat) / nat * 100 for exo, nat in zip(exp_vals_exo, exp_vals_nat)]
    # compute errors for new_perc_change_current, from each of the subjects in persub_perc_change_exp
    new_perc_change_filler = np.ones(len(persub_perc_change_exp)) * new_perc_change_current
    new_pc_diff = new_perc_change_filler - persub_perc_change_exp
    mean_diff = np.mean(new_pc_diff); print('Mean difference between current percent change and experimental percent change: ' + str(mean_diff))
    rmse_diff = np.sqrt(np.mean(new_pc_diff ** 2)); print('RMSE of differences: ' + str(rmse_diff))
    mae_diff = np.mean(np.abs(new_pc_diff)); print('MAE of differences: ' + str(mae_diff))
    std_diffs = np.std(new_pc_diff); print('Standard deviation of differences: ' + str(std_diffs))
    se_diffs = std_diffs / np.sqrt(len(persub_perc_change_exp)); print('Standard error of differences: ' + str(se_diffs))
    n = len(persub_perc_change_exp)
    # compute 95% CI
    t_crit = stats.t.ppf(0.975, df=n-1)  # two-tailed test
    ci_lower = mean_diff - t_crit * se_diffs
    ci_upper = mean_diff + t_crit * se_diffs
    print('95% CI: [' + str(ci_lower) + ', ' + str(ci_upper) + ']')
    

    print('\n' + '='*60)
    print('Experimental raw values: ' + 'exotendon:' + str(y_expexo[0]) + ' , natural:' + str(y_expnat[0]))
    print('Current simulation raw values: ' + 'exotendon:' + str(y_exo[0]) + ' , natural:' + str(y_nat[0]))
    
    new_perc_change_exp = (y_expexo[0] - y_expnat[0]) / y_expnat[0] * 100
    new_pc_rmse = np.sqrt((new_perc_change_exp - new_perc_change_current) ** 2)
    print('New Percent Change Experimental: ' + str(new_perc_change_exp))
    print('New Percent Change Current: ' + str(new_perc_change_current))
    print('New Percent Change Absolute error: ' + str(new_pc_rmse))
    
    


    print('\n' + '='*60)
    print(percent_change_prev)
    print(percent_change_current)

    metval27 = {}
    metval27['percent_change_27'] = percent_change_current
    metval27['nat27'] = y_nat
    metval27['exo27'] = y_exo


    # print data just for helping
    print('y_expnat  ' + str(y_expnat))
    print('y_expexo  ' + str(y_expexo))
    print('y_prev_nat  ' + str(y_prev_nat))
    print('y_prev_exo  ' + str(y_prev_exo))
    print('y_nat  ' + str(y_nat))
    print('y_exo  ' + str(y_exo))
    # pdb.set_trace()

    # create a second version of the figure, first panel will be full stride, 
    # and second will be stance and swing
    # Create a second subplot for percent change
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6), dpi=300) # figsize=(15, 6),
    yuplim = 13
    width = 0.15
    wantexp = True
    
    # First subplot: Comparison of Metabolic Rates
    if wantexp:
        ax1.bar(x[0] - 1.5*width, y_expexo[0], width, label='Exotendon experiment', color='#998ec3', yerr=y_expexostd[0], capsize=5)
        ax1.bar(x[0] - 0.5*width, y_prev_exo[0], width, label='Exotendon tracking simulation', color='#d8daeb', yerr=y_prev_exostd[0], capsize=5)
        ax1.bar(x[0] + 0.5*width, y_expnat[0], width, label='Natural experiment', color='#f1a340', yerr=y_expnatstd[0], capsize=5)
        ax1.bar(x[0] + 1.5*width, y_prev_nat[0], width, label='Natural tracking simulation', color='#fee0b6', yerr=y_prev_natstd[0], capsize=5)
        # ax1.bar(x[0] - 0.5*width, y_nat[0], width, label='Current simulation Nat.', color='#fee0b6')
        # ax1.bar(x[0] + 2.5*width, y_exo[0], width, label='Current simulation Exo.', color='#d8daeb')
    # else:
    #     ax1.bar(x - 1.5*width, y_nat, width, label='Current simulation Nat.', color='#e66101')
    #     ax1.bar(x - 0.5*width, y_exo, width, label='Current simulation Exo.', color='#5e3c99')
    #     ax1.bar(x + 0.5*width, y_prev_nat, width, label='Previous simulation Nat.', color='#fdb863', yerr=y_prev_natstd, capsize=5)
    #     ax1.bar(x + 1.5*width, y_prev_exo, width, label='Previous Exo. Simulations', color='#b2abd2', yerr=y_prev_exostd, capsize=5)
    ax1.bar([], [], yerr=2, capsize=5, error_kw=dict(label='Standard Dev.'))
    
    # Overlay simulation data
    ax1.hlines(exometavg, x[0] - 2*width, x[0], colors='#542788', linestyles='--', linewidth=4)#, label='Predicted Exotendon')
    ax1.hlines(natmetavg, x[0], x[0] + 2.0*width, colors='#b35806', linestyles=':', linewidth=4)#, label='Predicted Natural')

    
    # ax1.set_xlabel('Phases', fontsize=14)
    ax1.set_ylabel('Metabolic Rate (W/kg)', fontsize=14, fontweight='bold')
    # ax1.set_title('Comparison of Metabolic Rates', fontsize=14)
    ax1.set_xticks([x[0]])
    ax1.set_xticklabels(['Full Stride'], fontsize=14) #, 'Stance', 'Swing'
    ax1.tick_params(axis='y', labelsize=14)
    # ax1.legend(fontsize=12, loc='upper left')
    ax1.set_ylim(0, yuplim)
    
    ### second plot - this is the stance and swing values. 
    # start with a box plot for stance and swing based on the previous data. 
    # Create box plots for stance and swing phases based on the previous data
    prev_stance_nat_actual = [9.461884, 8.316323, 7.675539, 7.593565, 8.303889, 10.547504, 9.79799]
    prev_stance_exo_actual = [7.901537, 7.281104, 6.092534, 6.311269, 6.825258, 9.46874, 7.286447]
    prev_swing_nat_actual = [3.360173, 3.561437, 4.127728, 3.352854, 4.256651, 3.92321, 4.710525]
    prev_swing_exo_actual = [3.271432, 3.469623, 3.289909, 3.389188, 3.910336, 3.638481, 4.254498]

    # plot stances together with exo first
    # ax2.bar(x[1:] - 1.5*width, y_prev_nat[1:], width, label='Previous simulation Nat.', color='#f1a340', yerr=y_prev_natstd[1:], capsize=5)
    # ax2.bar(x[1:] + 1.5*width, y_prev_exo[1:], width, label='Previous Exo. Simulations', color='#998ec3', yerr=y_prev_exostd[1:], capsize=5)

    # test
    # Plot settings
    x = np.arange(2)  # Stance and Swing phases
    width = 0.35  # Bar width

    # format the data
    '''
    experimental_means = y_prev_nat, y_prev_exo
        a = exo
        b = nat
    
        stance condition a = prevnatstance, prevnatstancestd
        stance condition b = prevexostance, prevexostancestd
        swing condition a = prevnatswing, prevnatswingstd
        swing condition b = prevexoswing, prevexoswingstd
    simulation_data = y_nat, y_exo
        a = exo
        b = nat

        stance condition a = natstanceavg
        stance condition b = exostanceavg
        swing condition a = natswingavg
        swing condition b = exoswingavg
    '''

       # Bar plot for experimental data
    bars_condition_a = ax2.bar(
        x - width / 2,
        [prevexostance, prevexoswing],
        width,
        yerr=[
            prevexostancestd,
            prevexoswingstd,
        ],
        color="#d8daeb",
        capsize=8,
    )        # label="Previous Simulations Exotendon",

    bars_condition_b = ax2.bar(
        x + width / 2,
        [prevnatstance, prevnatswing],
        width,
        yerr=[
            prevnatstancestd,
            prevnatswingstd,
        ],
        color="#fee0b6",
        capsize=8,
    )        # label="Previous simulations Natural",


    # Overlay simulation data for each condition and phase
    ax2.plot(
        x - width / 2,
        [exostanceavg, exoswingavg],
        "o--",
        label="Simulated Exotendon",
        color="#542788",
        linewidth=3,
        markersize=12,
    )

    ax2.plot(
        x + width / 2,
        [natstanceavg, natswingavg],
        "s:",
        label="Simulated Natural",
        color="#b35806",
        linewidth=3,
        markersize=12,
    )

    # Formatting
    ax2.set_ylim(0, yuplim)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Stance Phase", "Swing Phase"], fontsize=14)
    # ax2.set_ylabel("Metabolic Rate (W/kg)", fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    # ax2.set_title("Stance and Swing costs", fontsize=14)
    # ax2.legend(fontsize=12, loc="upper right")

    # Hide the last subplot and use it to display the legend   
    ax3.axis('off')
    # get the legend labels from the previous subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Combine the handles and labels from both subplots
    handles = handles1 + handles2
    labels = labels1 + labels2
    order = [5,6,0,1,2,3,4]
    # display the legend entries in the last subplot 
    ax3.legend(handles, labels, loc='center left', fontsize=14, handlelength=4)
    ax3.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, handlelength=4, loc='center left')

    # add panel labels "a)" above the first subplot and "b)" between the first and second subplots
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    # small offsets to place labels just above the axes
    y_offset = 0.06
    x_offset = 0.12
    # 'a)' just above the first subplot (left)
    fig.text(pos1.x0 - x_offset, pos1.y1 + y_offset, 'a)', fontsize=16, fontweight='bold', va='bottom', ha='left')
    # 'b)' centered in the gap between first and second subplot (above)
    fig.text(((pos1.x1 + pos2.x0) / 2.0) - (x_offset/2) + 0.01, pos1.y1 + y_offset, 'b)', fontsize=16, fontweight='bold', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_metabolicvalidation_27.png')
    plt.show()

    return metval27

# metabolic validation figure for 4 m/s
def metabolicValidationFigure40(metval27, modelfile, GRFnat, GRFexo, natmet40, exomet40):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    partindices, natGRFtime = findStanceIndices(natmet40, exomet40, GRFnat, GRFexo)
    # load the metabolic results
    natmet40 = osim.TimeSeriesTable(natmet40)
    exomet40 = osim.TimeSeriesTable(exomet40)
    # check that the time vectors are the same
    natmet40time = np.array(natmet40.getIndependentColumn())
    exomet40time = np.array(exomet40.getIndependentColumn())
    if not np.array_equal(natmet40time, natGRFtime):
        print('Time vectors do not match')
        return
    # get the total metabolic rate for each phase
    total_stance_nat = np.zeros((len(partindices['part_main_hump_nattime']),9))
    total_stance_exo = np.zeros((len(partindices['part_main_hump_exotime']),9))
    total_swing_nat = np.zeros((len(partindices['part_after_hump_nattime']),9))
    total_swing_exo = np.zeros((len(partindices['part_after_hump_exotime']),9))
    count = 0
    musclenames = []
    for each in natmet40.getColumnLabels():
        if 'total_metabolic_rate' in each and '_r_' in each:
            shortname = each.split('_r_')[0][1:]; musclenames.append(shortname)
            # get the data
            natdata = natmet40.getDependentColumn(each).to_numpy()
            exodata = exomet40.getDependentColumn(each).to_numpy()
            # get the stance and swing phases
            natstance = natdata[partindices['start_of_hump_index_nat']:partindices['return_to_zero_index_nat'] + 1]
            exostance = exodata[partindices['start_of_hump_index_exo']:partindices['return_to_zero_index_exo'] + 1]
            natswing = natdata[partindices['return_to_zero_index_nat'] + 1:]
            exoswing = exodata[partindices['return_to_zero_index_exo'] + 1:]
            total_stance_nat[:,count] = natstance
            total_stance_exo[:,count] = exostance
            total_swing_nat[:,count] = natswing
            total_swing_exo[:,count] = exoswing
            count += 1
    # compute stance and swing for each muscle costs individually, and print... 
    # for nat and for exo
    natstanceind = {}
    exostanceind = {}
    natswingind = {}
    exoswingind = {}
    for m in musclenames:
        natstanceind[m] = np.trapz(total_stance_nat[:,musclenames.index(m)], x=partindices['part_main_hump_nattime']) / (partindices['part_main_hump_nattime'][-1] - partindices['part_main_hump_nattime'][0]) / modelmass
        exostanceind[m] = np.trapz(total_stance_exo[:,musclenames.index(m)], x=partindices['part_main_hump_exotime']) / (partindices['part_main_hump_exotime'][-1] - partindices['part_main_hump_exotime'][0]) / modelmass
        natswingind[m] = np.trapz(total_swing_nat[:,musclenames.index(m)], x=partindices['part_after_hump_nattime']) / (partindices['part_after_hump_nattime'][-1] - partindices['part_after_hump_nattime'][0]) / modelmass
        exoswingind[m] = np.trapz(total_swing_exo[:,musclenames.index(m)], x=partindices['part_after_hump_exotime']) / (partindices['part_after_hump_exotime'][-1] - partindices['part_after_hump_exotime'][0]) / modelmass
    # Create a list to store the data
    datastanceind = []
    dataswingind = []

    # Populate the list with muscle names, differences, and percent changes
    for muscle in natstanceind.keys():
        difference_stance = exostanceind[muscle] - natstanceind[muscle]
        percent_change_stance = (difference_stance / natstanceind[muscle]) * 100
        datastanceind.append([muscle, difference_stance, percent_change_stance])
        difference_swing = exoswingind[muscle] - natswingind[muscle]
        percent_change_swing = (difference_swing / natswingind[muscle]) * 100
        dataswingind.append([muscle, difference_swing, percent_change_swing])
    # Create a DataFrame from the data
    dfstanceind = pd.DataFrame(datastanceind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    dfswingind = pd.DataFrame(dataswingind, columns=['Muscle', 'Difference (W/kg)', 'Percent Change (%)'])
    # Sort the DataFrame by the 'Difference (W/kg)' column
    df_sorted_stanceind = dfstanceind.sort_values(by='Difference (W/kg)', ascending=True)
    df_sorted_swingind = dfswingind.sort_values(by='Difference (W/kg)', ascending=True)
    # sum the metabolic rates for each phase
    total_stance_nat = np.sum(total_stance_nat, axis=1)
    total_stance_exo = np.sum(total_stance_exo, axis=1)
    total_swing_nat = np.sum(total_swing_nat, axis=1)
    total_swing_exo = np.sum(total_swing_exo, axis=1)
    natstanceavg = np.trapz(total_stance_nat, x=partindices['part_main_hump_nattime']) / (partindices['part_main_hump_nattime'][-1] - partindices['part_main_hump_nattime'][0]) / modelmass
    exostanceavg = np.trapz(total_stance_exo, x=partindices['part_main_hump_exotime']) / (partindices['part_main_hump_exotime'][-1] - partindices['part_main_hump_exotime'][0]) / modelmass
    natswingavg = np.trapz(total_swing_nat, x=partindices['part_after_hump_nattime']) / (partindices['part_after_hump_nattime'][-1] - partindices['part_after_hump_nattime'][0]) / modelmass
    exoswingavg = np.trapz(total_swing_exo, x=partindices['part_after_hump_exotime']) / (partindices['part_after_hump_exotime'][-1] - partindices['part_after_hump_exotime'][0]) / modelmass
    # get the average metabolic rate for the whole body
    natmet40avg = np.trapz(natmet40.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy(), x=natmet40time) / (natmet40time[-1] - natmet40time[0]) / modelmass
    exomet40avg = np.trapz(exomet40.getDependentColumn('/metabolic_cost|total_metabolic_rate').to_numpy(), x=exomet40time) / (exomet40time[-1] - exomet40time[0]) / modelmass
    # hard code the experimental values and values from the previous papers. metaboliccostcomparisons_rev2.xlsx
    # experimental values for whole body

    expnatavg = 10.68
    expnatstd = 0.88
    expexoavg = 9.78
    expexostd = 0.69
    # previous simulation whole body
    prevnatavg = 11.62
    prevnatstd = 0.95
    prevexoavg = 10.21
    prevexostd = 0.80
    # previous simulation stance values
    prevnatstance = 8.81
    prevnatstancestd = 1.05
    prevexostance = 7.31
    prevexostancestd = 1.05
    # previous simulations swing data
    prevnatswing = 3.90
    prevnatswingstd = 0.47
    prevexoswing = 3.60
    prevexoswingstd = 0.34
    ## okay we should have all the values now to make the figure
    # Create a figure and axis
    # Define the y values for the bars
    # Define the y values for the previous data bars
    y_prev_nat = [prevnatavg, prevnatstance, prevnatswing]
    y_prev_natstd = [prevnatstd, prevnatstancestd, prevnatswingstd]
    y_prev_exo = [prevexoavg, prevexostance, prevexoswing]
    y_prev_exostd = [prevexostd, prevexostancestd, prevexoswingstd]
    y_nat = [natmet40avg, natstanceavg, natswingavg]
    y_exo = [exomet40avg, exostanceavg, exoswingavg]
    # Define the x values for the bars
    x = np.arange(3)

    # Create a second subplot for percent change
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=500)
    width = 0.15
    # First subplot: Comparison of Metabolic Rates
    ax1.bar(x - width, y_nat, width, label='4.0 m/s Nat. simulations', color='#e66101')
    ax1.bar(x, y_exo, width, label='4.0 m/s Exo. simulations', color='#5e3c99')
    # ax1.bar(x + width, metval27['nat27'], width, label='2.7 m/s Nat. simulations', color='#fdb863')
    # ax1.bar(x + 2*width, metval27['exo27'], width, label='2.7 m/s Exo. simulations', color='#b2abd2')
    # ax1.bar([], [], yerr=2, capsize=5, error_kw=dict(label='Standard Dev.'))
    # ax1.set_xlabel('Phases', fontsize=14)
    ax1.set_ylabel('Metabolic Rate (W/kg)', fontsize=14)
    ax1.set_title('Comparison of Metabolic Rates', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Full Stride', 'Stance', 'Swing'], fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(fontsize=12)
    # ax1.set_ylim(0, max(y_prev_nat, y_prev_exo, y_nat + y_exo) + 1)
    # now set up the percent change standard deviations from the previous data - metaboliccostcomparisons_rev2
    prev_perc = -11.97
    prev_perc_std = 4.76
    prev_perc_stance = -12.83
    prev_perc_stance_std = 3.11
    prev_perc_swing = -2.51
    prev_perc_swing_std = 2.44
    # Calculate percent change
    # percent_change_prev = [(exo - nat) / y_prev_nat[0] * 100 for exo, nat in zip(y_prev_exo, y_prev_nat)]
    percent_change_current = [(exo - nat) / y_nat[0] * 100 for exo, nat in zip(y_exo, y_nat)]
    percent_change_prev = [prev_perc, prev_perc_stance, prev_perc_swing]
    percent_change_prev_std = [prev_perc_std, prev_perc_stance_std, prev_perc_swing_std]
    # Second subplot: Percent Change
    width = 0.15
    # ax2.bar(x + width/2, metval27['percent_change_27'], width, label='2.7 m/s Sim.', color='#018571')
    # ax2.bar(x - width/2, percent_change_current, width, label='4.0 m/s Sim.', color='#dfc27d')
    # ax2.set_xlabel('Phases', fontsize=14)
    # ax2.set_ylabel('Percent Change (%)', fontsize=14)
    # ax2.set_title('Percent Change wrt. Whole body Natural Metabolic Rate', fontsize=14)
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(['Full Stride', 'Stance', 'Swing'], fontsize=14)
    # ax2.tick_params(axis='y', labelsize=14)
    # # ax2.bar([], [], yerr=2, capsize=5, error_kw=dict(label='Standard Dev.'))
    # ax2.legend(fontsize=12, loc='lower right')
    # # Add a horizontal line at y=0 for the x-axis
    # for ax in [ax1, ax2]:
    #     ax.axhline(0, color='black', linewidth=0.8)
    # # Show the plot
    # ax2.set_position([0.55, 0.15, 0.65, 0.7])  # [left, bottom, width, height]
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_metabolicvalidation_40.png')
    plt.show()
    print(percent_change_prev)
    print(percent_change_current)

    return

# create function for muscle savers validation figure. 
def muscleSaversValidationFigure27(natmet, exomet, modelfile, GRFnat, GRFexo):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the GRF results
    # load the metabolic results
    natmetdata = osim.TimeSeriesTable(natmet)
    exometdata = osim.TimeSeriesTable(exomet)    
    
    # first lets create a plot for all the right muscles and total met rate. 
    # fig, ax = plt.subplots(2,5, figsize=(15,6), dpi=300) # , dpi=300
    # ax = ax.flatten()
    # count = 0
    metvalues = []
    # loop and get the right muscle total rates
    for each in natmetdata.getColumnLabels():
        if '_r_' in each and 'total_metabolic_rate' in each:
            tempname = each.split('_r_')[0]
            # print(tempname)
            # get the data
            sim1data = natmetdata.getDependentColumn(each).to_numpy()
            sim2data = exometdata.getDependentColumn(each).to_numpy()
            # get the time
            time = natmetdata.getIndependentColumn()
            x1 = np.linspace(0,100,len(sim1data))
            x2 = np.linspace(0,100,len(sim2data))
            sim2data = np.interp(x1, x2, sim2data)
            # plot it
            # ax[count].plot(x1, sim1data, label='sim1', color='orange')
            # ax[count].plot(x2, sim2data, label='sim2', color='purple')
            # ax[count].set_title(tempname[1:])
            # ax[count].set_ylabel('Metabolic rate W/kg')
            # ax[count].set_xlabel('Gait cycle (%)')
            # count += 1
            model = osim.Model(modelfile); st = model.initSystem()
            modelmass = model.getTotalMass(st)
            # now get the average values for each one 
            sim1_avg = np.trapz(sim1data, x=x1) / (x1[-1] - x1[0]) / modelmass
            sim2_avg = np.trapz(sim2data, x=x2) / (x2[-1] - x2[0]) / modelmass
            # print(f'Avg of sim1 (nat) for {tempname[1:]}: {sim1_avg}')
            # print(f'Avg of sim2 (exo) for {tempname[1:]}: {sim2_avg}')
            # now print out the difference and percent change between the two
            diff = sim2_avg - sim1_avg
            percent_change = (diff / sim1_avg) * 100
            # print(f'Difference: {diff}')
            # print(f'Percent change: {percent_change}%\n')            
            metvalues.append({'Muscle': tempname, 'Nat. Met': sim1_avg, 'Exo Met': sim2_avg, 'Difference': diff, 'Percent Change': percent_change})
    # plt.tight_layout()
    figurePath = os.getcwd() + '\\..\\..\\analysis\\'
    # plt.savefig(figurePath + 'individualMetabolics_Sim' + '2_7' + '.png')
    met_df = pd.DataFrame(metvalues)
    met_df.to_csv(figurePath + 'muscleMetabolics_Sim' + '2_7' + '.csv')
    # Sort the DataFrame based on the 'Difference' column
    met_df = met_df.sort_values(by='Difference', ascending=True)
    # print(met_df)
    # plt.show()

    # now gather up the individual muscle changes from the previous simulation paper results. 
    # prev_muscles = ['quadriceps', 'hip flexors', 'hip abductors', 'hamstrings', 'hip adductors', 'hip extensors', 'plantar flexors', 'dorsiflexors']
    # prev_values_raw = [-0.135380155, -0.110176431, -0.096573244, -0.092950201, -0.064876238, -0.045507821, 0.008920576, 0.033780946] 
    # prev_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference': prev_values_raw})
    prev_df = pd.read_csv('musclesStudy1Savings.csv')
    # print(prev_df)

    # work on formatting the current to match the previous groupings
    current_values = [(met_df[met_df['Muscle'] == '/vasint']['Difference'].values[0]) + (met_df[met_df['Muscle'] == '/recfem']['Difference'].values[0]),
                      met_df[met_df['Muscle'] == '/psoas']['Difference'].values[0],
                      0, 
                      met_df[met_df['Muscle'] == '/semimem']['Difference'].values[0],
                      0,
                      met_df[met_df['Muscle'] == '/glmax2']['Difference'].values[0],
                      (met_df[met_df['Muscle'] == '/soleus']['Difference'].values[0]) + (met_df[met_df['Muscle'] == '/gasmed']['Difference'].values[0]),
                      met_df[met_df['Muscle'] == '/tibant']['Difference'].values[0]]
    # create a new dataframe for the current values. 
    current_df = pd.DataFrame({'Muscle': prev_df['Muscles'], 'Difference': current_values})
    # print(current_df)

    # Filter out rows where the 'Difference' is zero in current_df
    non_zero_indices = current_df[current_df['Difference'] != 0].index
    # Filter both DataFrames to keep only the non-zero differences
    current_df = current_df.loc[non_zero_indices]
    prev_df = prev_df.loc[non_zero_indices]

    # Get the errors for the previous data
    prev_errors = prev_df['SE']

    # now create a plot for the ranked muscle groupings. 
    # Create a bar plot to show the differences between the previous and current data
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # Define the width of the bars
    bar_width = 0.35
    # Get the positions for the bars
    indices = np.arange(len(prev_df))
    # Plot the bars
    # Add a legend entry for the error bars
    ax.bar(indices - bar_width/2, prev_df['average'], bar_width, label='Previous Simulations', color='#018571', yerr=prev_errors, capsize=5, error_kw=dict(label='Standard Error'))
    ax.bar(indices + bar_width/2, current_df['Difference'], bar_width, label='Current Simulations', color='#dfc27d')
    # Set the labels and title
    ax.set_xlabel('Muscle Groups', fontsize=14)
    ax.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    ax.set_title('Comparison of Muscle Group Metabolic Rate Differences (full stride)', fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(prev_df['Muscles'], rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    # Add a horizontal line at y=0 for the x-axis
    ax.axhline(0, color='black', linewidth=0.8)
    # Show the plot
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_27.png')
    plt.show()

    # create another plot where we are plotting the current_df values on the x axis and the prev_df values on the y axis.
    # Create a scatter plot to compare current and previous differences
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.errorbar([], [], xerr=[], fmt='.', color='blue', label='Standard Error')
    # Stance phase scatter plot
    ax.errorbar(prev_df['average'], current_df['Difference'], xerr=prev_errors, fmt='o', color='blue', label='Muscle Groups')
    # Add labels for each point
    for i, muscle in enumerate(current_df['Muscle']):
        ax.text(prev_df['average'].iloc[i], current_df['Difference'].iloc[i] + 0.01, muscle + ' ', fontsize=10, ha='right')
    # Set the labels and title
    ax.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    ax.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    ax.set_title('Comparison of Muscle Group Metabolic Rate Differences (full cycle)', fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    # Add a dotted grey line to represent y=x
    ax.plot([-0.2, 0.05], [-0.2, 0.05], linestyle='--', color='grey', label='y=x')
    # Ensure the plot is square and axes have the same bounds
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=12)
    max_bound = max(ax.get_xlim()[1], ax.get_ylim()[1])
    min_bound = min(ax.get_xlim()[0], ax.get_ylim()[0])
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_scatter_27.png')
    plt.show()

    return current_df

# metabolic results figure for whole body savings in 4 m/s running
def muscleSaversValidationFigure40(natmet, exomet, modelfile, GRFnat, GRFexo, musclemet27):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the GRF results
    # load the metabolic results
    natmetdata = osim.TimeSeriesTable(natmet)
    exometdata = osim.TimeSeriesTable(exomet)    
    
    # first lets create a plot for all the right muscles and total met rate. 
    # fig, ax = plt.subplots(2,5, figsize=(15,6), dpi=300) # , dpi=300
    # ax = ax.flatten()
    # count = 0
    metvalues = []
    # loop and get the right muscle total rates
    for each in natmetdata.getColumnLabels():
        if '_r_' in each and 'total_metabolic_rate' in each:
            tempname = each.split('_r_')[0]
            # print(tempname)
            # get the data
            sim1data = natmetdata.getDependentColumn(each).to_numpy()
            sim2data = exometdata.getDependentColumn(each).to_numpy()
            # get the time
            time = natmetdata.getIndependentColumn()
            x1 = np.linspace(0,100,len(sim1data))
            x2 = np.linspace(0,100,len(sim2data))
            sim2data = np.interp(x1, x2, sim2data)
            # plot it
            # ax[count].plot(x1, sim1data, label='sim1', color='orange')
            # ax[count].plot(x2, sim2data, label='sim2', color='purple')
            # ax[count].set_title(tempname[1:])
            # ax[count].set_ylabel('Metabolic rate W/kg')
            # ax[count].set_xlabel('Gait cycle (%)')
            # count += 1
            model = osim.Model(modelfile); st = model.initSystem()
            modelmass = model.getTotalMass(st)
            # now get the average values for each one 
            sim1_avg = np.trapz(sim1data, x=x1) / (x1[-1] - x1[0]) / modelmass
            sim2_avg = np.trapz(sim2data, x=x2) / (x2[-1] - x2[0]) / modelmass
            # print(f'Avg of sim1 (nat) for {tempname[1:]}: {sim1_avg}')
            # print(f'Avg of sim2 (exo) for {tempname[1:]}: {sim2_avg}')
            # now print out the difference and percent change between the two
            diff = sim2_avg - sim1_avg
            percent_change = (diff / sim1_avg) * 100
            # print(f'Difference: {diff}')
            # print(f'Percent change: {percent_change}%\n')            
            metvalues.append({'Muscle': tempname, 'Nat. Met': sim1_avg, 'Exo Met': sim2_avg, 'Difference': diff, 'Percent Change': percent_change})
    # plt.tight_layout()
    figurePath = os.getcwd() + '\\..\\..\\analysis\\'
    # plt.savefig(figurePath + 'individualMetabolics_Sim' + '2_7' + '.png')
    met_df = pd.DataFrame(metvalues)
    met_df.to_csv(figurePath + 'muscleMetabolics_Sim' + '2_7' + '.csv')
    # Sort the DataFrame based on the 'Difference' column
    met_df = met_df.sort_values(by='Difference', ascending=True)
    # print(met_df)
    # plt.show()

    # now gather up the individual muscle changes from the previous simulation paper results. 
    # prev_muscles = ['quadriceps', 'hip flexors', 'hip abductors', 'hamstrings', 'hip adductors', 'hip extensors', 'plantar flexors', 'dorsiflexors']
    # prev_values_raw = [-0.135380155, -0.110176431, -0.096573244, -0.092950201, -0.064876238, -0.045507821, 0.008920576, 0.033780946] 
    # prev_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference': prev_values_raw})
    prev_df = pd.read_csv('musclesStudy1Savings.csv')
    # print(prev_df)

    # work on formatting the current to match the previous groupings
    current_values = [(met_df[met_df['Muscle'] == '/vasint']['Difference'].values[0]) + (met_df[met_df['Muscle'] == '/recfem']['Difference'].values[0]),
                      met_df[met_df['Muscle'] == '/psoas']['Difference'].values[0],
                      0, 
                      met_df[met_df['Muscle'] == '/semimem']['Difference'].values[0],
                      0,
                      met_df[met_df['Muscle'] == '/glmax2']['Difference'].values[0],
                      (met_df[met_df['Muscle'] == '/soleus']['Difference'].values[0]) + (met_df[met_df['Muscle'] == '/gasmed']['Difference'].values[0]),
                      met_df[met_df['Muscle'] == '/tibant']['Difference'].values[0]]
    # create a new dataframe for the current values. 
    current_df = pd.DataFrame({'Muscle': prev_df['Muscles'], 'Difference': current_values})
    # print(current_df)

    # Filter out rows where the 'Difference' is zero in current_df
    non_zero_indices = current_df[current_df['Difference'] != 0].index
    # Filter both DataFrames to keep only the non-zero differences
    current_df = current_df.loc[non_zero_indices]
    prev_df = prev_df.loc[non_zero_indices]

    # Get the errors for the previous data
    prev_errors = prev_df['SE']

    # now create a plot for the ranked muscle groupings. 
    # Create a bar plot to show the differences between the previous and current data
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # Define the width of the bars
    bar_width = 0.35
    # Get the positions for the bars
    indices = np.arange(len(prev_df))
    # Plot the bars
    # Add a legend entry for the error bars
    ax.bar(indices - bar_width/2, musclemet27['Difference'], bar_width, label='2.7 m/s Simulations', color='#018571')
    ax.bar(indices + bar_width/2, current_df['Difference'], bar_width, label='4.0 m/s Simulations', color='#dfc27d')
    # Set the labels and title
    ax.set_xlabel('Muscle Groups', fontsize=14)
    ax.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    ax.set_title('Comparison of Muscle Group Metabolic Rate Differences (full stride)', fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(prev_df['Muscles'], rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    # Add a horizontal line at y=0 for the x-axis
    ax.axhline(0, color='black', linewidth=0.8)
    # Show the plot
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_40.png')
    plt.show()

    # # create another plot where we are plotting the current_df values on the x axis and the prev_df values on the y axis.
    # # Create a scatter plot to compare current and previous differences
    # fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # ax.errorbar([], [], xerr=[], fmt='.', color='blue', label='Standard Error')
    # # Stance phase scatter plot
    # ax.errorbar(prev_df['average'], current_df['Difference'], xerr=prev_errors, fmt='o', color='blue', label='Muscle Groups')
    # # Add labels for each point
    # for i, muscle in enumerate(current_df['Muscle']):
    #     ax.text(prev_df['average'].iloc[i], current_df['Difference'].iloc[i] + 0.01, muscle + ' ', fontsize=10, ha='right')
    # # Set the labels and title
    # ax.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    # ax.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    # ax.set_title('Comparison of Muscle Group Metabolic Rate Differences (full cycle)', fontsize=14)
    # ax.axhline(0, color='black', linewidth=0.8)
    # ax.axvline(0, color='black', linewidth=0.8)
    # # Add a dotted grey line to represent y=x
    # ax.plot([-0.2, 0.05], [-0.2, 0.05], linestyle='--', color='grey', label='y=x')
    # # Ensure the plot is square and axes have the same bounds
    # ax.set_aspect('equal', 'box')
    # ax.legend(fontsize=12)
    # max_bound = max(ax.get_xlim()[1], ax.get_ylim()[1])
    # min_bound = min(ax.get_xlim()[0], ax.get_ylim()[0])
    # ax.set_xlim(min_bound, max_bound)
    # ax.set_ylim(min_bound, max_bound)
    # plt.tight_layout()
    # plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_scatter.png')
    # plt.show()

    return

# create function for stance and swing differences in the individual muscle metabolics. 
def muscleSaversStanceSwingValidationFigure27(natmet, exomet, modelfile, GRFnat, GRFexo):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the GRF results
    # load the metabolic results
    try: 
        stance = pd.read_csv('stanceStudy2Savings.csv')
        swing = pd.read_csv('swingStudy2Savings.csv')    
    except: 
        print('Could not find the study 2 savings files. Please run the study 2 savings script first. helperOsimFunctions.metabolicsStanceSwing()')
    # load in the previous study data
    prevstance = pd.read_csv('stanceStudy1Savings.csv')
    prevswing = pd.read_csv('swingStudy1Savings.csv')
    

    # now gather up the individual muscle changes from the previous simulation paper results. 
    prev_muscles = ['quads_r', 'hipflexors_r', 'hipabductors_r', 'hamstrings_r', 'hipadductors_r', 'hipextensor_r', 'plantarflex_r', 'dorsiflex_r']

    # change the names of the muscles to match the previous study
    currentstance = [
        (stance[stance['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (stance[stance['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        stance[stance['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        stance[stance['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        stance[stance['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (stance[stance['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (stance[stance['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        stance[stance['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    currentswing = [
        (swing[swing['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (swing[swing['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        swing[swing['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        swing[swing['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        swing[swing['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (swing[swing['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (swing[swing['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        swing[swing['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    # create a new dataframe for the current values.
    currentstance_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': currentstance})
    currentswing_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': currentswing})    

    # be sure to sort the structures based on the muscle group so that we are consistent
    currentstance_df = currentstance_df.sort_values(by='Muscle')
    currentswing_df = currentswing_df.sort_values(by='Muscle')
    prevstance = prevstance.sort_values(by='Muscles')
    prevswing = prevswing.sort_values(by='Muscles')
    # now get rid of any muscle groups that have a zero difference
    currentstance_df = currentstance_df[currentstance_df['Difference (W/kg)'] != 0]
    currentswing_df = currentswing_df[currentswing_df['Difference (W/kg)'] != 0]
    # check the previous data structures and get rid of any muscle rows that do not exist in the current data
    prevstance = prevstance[prevstance['Muscles'].isin(currentstance_df['Muscle'])]
    prevswing = prevswing[prevswing['Muscles'].isin(currentswing_df['Muscle'])]


    
    # # Create a bar plot to show the differences between the previous and current data for stance and swing phases
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Define the width of the bars
    bar_width = 0.35

    # Get the positions for the bars
    indices_stance = np.arange(len(prevstance))
    indices_swing = np.arange(len(prevswing))

    # Get the errors for the previous data
    prevstance_errors = prevstance['SE']
    prevswing_errors = prevswing['SE']

    # # Plot the bars for stance phase with error bars
    # ax1.bar(indices_stance - bar_width/2, prevstance['average'], bar_width, label='Previous Simulations', color='#018571', yerr=prevstance_errors, capsize=5, error_kw=dict(label='Standard Error'))
    # ax1.bar(indices_stance + bar_width/2, currentstance_df['Difference (W/kg)'], bar_width, label='Current Simulations', color='#dfc27d')
    # ax1.set_xlabel('Muscle Groups', fontsize=14)
    # ax1.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    # ax1.set_title('Stance phase muscle metabolic changes', fontsize=14)
    # ax1.set_xticks(indices_stance)
    # ax1.set_xticklabels(prevstance['Muscles'], rotation=30, ha='right', fontsize=14)
    # ax1.tick_params(axis='y', labelsize=14)
    # ax1.legend(fontsize=14)
    # ax1.axhline(0, color='black', linewidth=0.8)

    # # Plot the bars for swing phase with error bars
    # ax2.bar(indices_swing - bar_width/2, prevswing['average'], bar_width, label='Previous Simulations', color='#018571', yerr=prevswing_errors, capsize=5, error_kw=dict(label='Standard Error'))
    # ax2.bar(indices_swing + bar_width/2, currentswing_df['Difference (W/kg)'], bar_width, label='Current Simulations', color='#dfc27d')
    # ax2.set_xlabel('Muscle Groups', fontsize=14)
    # ax2.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    # ax2.set_title('Swing phase muscle metabolic changes', fontsize=14)
    # ax2.set_xticks(indices_swing)
    # ax2.set_xticklabels(prevswing['Muscles'], rotation=30, ha='right', fontsize=14)
    # ax2.tick_params(axis='y', labelsize=14)
    # ax2.legend(fontsize=14)
    # ax2.axhline(0, color='black', linewidth=0.8)

    # # Set the y-axis limits to be the same for both subplots
    # y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    # y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    # ax1.set_ylim([y_min, y_max])
    # ax2.set_ylim([y_min, y_max])
    # plt.tight_layout()
    # plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_stance_swing_27.png')
    # plt.show()



    # create another plot where we are plotting the current_df values on the x axis and the prev_df values on the y axis.
    # Create a scatter plot to compare current and previous differences for stance and swing phases
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    # Add a legend entry for the error bars
    ax3.errorbar([], [], xerr=[], fmt='none', color='blue', label='Standard Error')
    ax4.errorbar([], [], xerr=[], fmt='none', color='blue', label='Standard Error')
    # Stance phase scatter plot
    ax3.errorbar(prevstance['average'], currentstance_df['Difference (W/kg)'], xerr=prevstance_errors, fmt='o', color='blue', label='Muscle Groups')
    # Add labels for each point
    for i, muscle in enumerate(currentstance_df['Muscle']):
        if 'dorsi' in muscle:
            ax3.text(prevstance['average'].iloc[i], currentstance_df['Difference (W/kg)'].iloc[i] - 0.04, ' ' + muscle, fontsize=10, ha='left')
        elif 'hipflexor' in muscle: 
            ax3.text(prevstance['average'].iloc[i], currentstance_df['Difference (W/kg)'].iloc[i] + 0.01, ' ' + muscle, fontsize=10, ha='left')
        else:
            ax3.text(prevstance['average'].iloc[i], currentstance_df['Difference (W/kg)'].iloc[i] + 0.01, muscle + ' ', fontsize=10, ha='right')
    # Set the labels and title
    ax3.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    ax3.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    ax3.set_title('Stance phase muscle metabolic changes', fontsize=14)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.axvline(0, color='black', linewidth=0.8)
    # Add a dotted grey line to represent y=x
    ax3.plot([-0.75, 0.3], [-0.75, 0.3], linestyle='--', color='grey', label='y=x')
    # Ensure the plot is square and axes have the same bounds
    ax3.set_aspect('equal', 'box')
    ax3.legend(fontsize=12)
    
    # Swing phase scatter plot
    ax4.errorbar(prevswing['average'], currentswing_df['Difference (W/kg)'], xerr=prevswing_errors, fmt='o', color='blue', label='Muscle Groups')
    # Add labels for each point
    for i, muscle in enumerate(currentswing_df['Muscle']):
        if 'hipextensor' in muscle or 'dorsi' in muscle:
            ax4.text(prevswing['average'].iloc[i], currentswing_df['Difference (W/kg)'].iloc[i] + 0.01, '  ' + muscle, fontsize=10, ha='left')
        else:
            ax4.text(prevswing['average'].iloc[i], currentswing_df['Difference (W/kg)'].iloc[i] + 0.01, muscle + '  ', fontsize=10, ha='right')
    # Set the labels and title
    ax4.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    ax4.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    ax4.set_title('Swing phase muscle metabolic changes', fontsize=14)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.axvline(0, color='black', linewidth=0.8)
    # Add a dotted grey line to represent y=x
    ax4.plot([-0.75, 0.3], [-0.75, 0.3], linestyle='--', color='grey', label='y=x')
    # Ensure the plot is square and axes have the same bounds
    ax4.set_aspect('equal', 'box')
    ax4.legend(fontsize=12)
    
    y_min = min(ax3.get_ylim()[0], ax4.get_ylim()[0])
    y_max = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
    x_min = min(ax3.get_xlim()[0], ax4.get_xlim()[0])
    x_max = max(ax3.get_xlim()[1], ax4.get_xlim()[1])

    fullmin = min(y_min, x_min)
    fullmax = max(y_max, x_max)

    ax3.set_ylim([fullmin, fullmax])
    ax3.set_xlim([fullmin, fullmax])
    ax4.set_ylim([fullmin, fullmax])
    ax4.set_xlim([fullmin, fullmax])
    ax3.set_aspect('equal', 'box')
    ax4.set_aspect('equal', 'box')

    # Add A) and B) labels to the subplots
    ax3.text(-0.95, fullmax, 'a)', fontsize=16, weight='bold')
    ax4.text(-0.95, fullmax, 'b)', fontsize=16, weight='bold')

    # plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_stance_swing_scatter_27.png')
    plt.show()

    return currentstance_df, currentswing_df

# results figure of the stance and swing muscle level costs at 4 m/s
def muscleSaversStanceSwingValidationFigure40(natmet, exomet, modelfile, GRFnat, GRFexo, swingmet27, stancemet27):
    # load the model and mass
    model = osim.Model(modelfile); st = model.initSystem()
    modelmass = model.getTotalMass(st)
    # load the GRF results
    # load the metabolic results
    try: 
        stance27 = pd.read_csv('stanceStudy2Savings.csv')
        swing27 = pd.read_csv('swingStudy2Savings.csv')
        stance = pd.read_csv('stanceStudy2Savings4ms.csv')
        swing = pd.read_csv('swingStudy2Savings4ms.csv')
    except: 
        print('Could not find the study 2 savings files. Please run the study 2 savings script first. helperOsimFunctions.metabolicsStanceSwing()')
    # load in the previous study data
    # prevstance = pd.read_csv('stanceStudy1Savings.csv')
    # prevswing = pd.read_csv('swingStudy1Savings.csv')
    prevstance = stance27
    prevswing = swing27


    # now gather up the individual muscle changes from the previous simulation paper results. 
    prev_muscles = ['quads_r', 'hipflexors_r', 'hipabductors_r', 'hamstrings_r', 'hipadductors_r', 'hipextensor_r', 'plantarflex_r', 'dorsiflex_r']

    # change the names of the muscles to match the previous study
    currentstance = [
        (stance[stance['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (stance[stance['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        stance[stance['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        stance[stance['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        stance[stance['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (stance[stance['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (stance[stance['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        stance[stance['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    currentswing = [
        (swing[swing['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (swing[swing['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        swing[swing['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        swing[swing['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        swing[swing['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (swing[swing['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (swing[swing['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        swing[swing['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    # create a new dataframe for the current values.
    currentstance_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': currentstance})
    currentswing_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': currentswing})    

    
    # be sure to sort the structures based on the muscle group so that we are consistent
    currentstance_df = currentstance_df.sort_values(by='Muscle')
    currentswing_df = currentswing_df.sort_values(by='Muscle')
    
    # change the names of the muscles to match the previous study
    slowstance = [
        (stance27[stance27['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (stance27[stance27['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        stance27[stance27['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        stance27[stance27['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        stance27[stance27['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (stance27[stance27['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (stance27[stance27['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        stance27[stance27['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    slowswing = [
        (swing27[swing27['Muscle'] == 'vasint']['Difference (W/kg)'].values[0]) + (swing27[swing27['Muscle'] == 'recfem']['Difference (W/kg)'].values[0]),
        swing27[swing27['Muscle'] == 'psoas']['Difference (W/kg)'].values[0],
        0,
        swing27[swing27['Muscle'] == 'semimem']['Difference (W/kg)'].values[0],
        0,
        swing27[swing27['Muscle'] == 'glmax2']['Difference (W/kg)'].values[0],
        (swing27[swing27['Muscle'] == 'soleus']['Difference (W/kg)'].values[0]) + (swing27[swing27['Muscle'] == 'gasmed']['Difference (W/kg)'].values[0]),
        swing27[swing27['Muscle'] == 'tibant']['Difference (W/kg)'].values[0]]
    # create a new dataframe for the current values.
    slowstance_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': slowstance})
    slowswing_df = pd.DataFrame({'Muscle': prev_muscles, 'Difference (W/kg)': slowswing})    

    # be sure to sort the structures based on the muscle group so that we are consistent
    slowstance_df = slowstance_df.sort_values(by='Muscle')
    slowswing_df = slowswing_df.sort_values(by='Muscle')
    
    # now get rid of any muscle groups that have a zero difference
    currentstance_df = currentstance_df[currentstance_df['Difference (W/kg)'] != 0]
    currentswing_df = currentswing_df[currentswing_df['Difference (W/kg)'] != 0]
    # check the previous data structures and get rid of any muscle rows that do not exist in the current data
    prevstance = slowstance_df[slowstance_df['Difference (W/kg)'] != 0]
    prevswing = slowswing_df[slowswing_df['Difference (W/kg)'] != 0]
    # prevstance = slowstance_df[slowstance_df['Muscles'].isin(currentstance_df['Muscle'])]
    # prevswing = slowswing_df[slowswing_df['Muscles'].isin(currentswing_df['Muscle'])]
    

    # Create a bar plot to show the differences between the previous and current data for stance and swing phases
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    # Define the width of the bars
    bar_width = 0.35
    # Get the positions for the bars
    indices_stance = np.arange(len(prevstance))
    indices_swing = np.arange(len(prevswing))
    # Get the errors for the previous data
    # prevstance_errors = prevstance['SE']
    # prevswing_errors = prevswing['SE']
    # Plot the bars for stance phase with error bars
    ax1.bar(indices_stance - bar_width/2, prevstance['Difference (W/kg)'], bar_width, label='2.7 m/s Simulations', color='#018571')
    ax1.bar(indices_stance + bar_width/2, currentstance_df['Difference (W/kg)'], bar_width, label='4 m/s Simulations', color='#dfc27d')
    ax1.set_xlabel('Muscle Groups', fontsize=14)
    ax1.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    ax1.set_title('Comparison of Muscle Group Metabolic Rate Differences (Stance Phase)', fontsize=14)
    ax1.set_xticks(indices_stance)
    ax1.set_xticklabels(prevstance['Muscle'], rotation=30, ha='right', fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(fontsize=14)
    ax1.axhline(0, color='black', linewidth=0.8)

    # Plot the bars for swing phase with error bars
    ax2.bar(indices_swing - bar_width/2, prevswing['Difference (W/kg)'], bar_width, label='2.7 m/s Simulations', color='#018571')
    ax2.bar(indices_swing + bar_width/2, currentswing_df['Difference (W/kg)'], bar_width, label='4.0 m/s Simulations', color='#dfc27d')
    ax2.set_xlabel('Muscle Groups', fontsize=14)
    ax2.set_ylabel('Difference in Metabolic Rate (W/kg)', fontsize=14)
    ax2.set_title('Comparison of Muscle Group Metabolic Rate Differences (Swing Phase)', fontsize=14)
    ax2.set_xticks(indices_swing)
    ax2.set_xticklabels(prevswing['Muscle'], rotation=30, ha='right', fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.legend(fontsize=14)
    ax2.axhline(0, color='black', linewidth=0.8)

    # Set the y-axis limits to be the same for both subplots
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim([y_min, y_max])
    ax2.set_ylim([y_min, y_max])
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_stance_swing_40.png')
    plt.show()
    
    # # create another plot where we are plotting the current_df values on the x axis and the prev_df values on the y axis.
    # # Create a scatter plot to compare current and previous differences for stance and swing phases
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    # # Add a legend entry for the error bars
    # ax1.errorbar([], [], xerr=[], fmt='.', color='blue', label='Standard Error')
    # ax2.errorbar([], [], xerr=[], fmt='.', color='blue', label='Standard Error')
    # # Stance phase scatter plot
    # ax1.errorbar(prevstance['average'], currentstance_df['Difference (W/kg)'], xerr=prevstance_errors, fmt='o', color='blue', label='Muscle Groups')
    # # Add labels for each point
    # for i, muscle in enumerate(currentstance_df['Muscle']):
    #     ax1.text(prevstance['average'].iloc[i], currentstance_df['Difference (W/kg)'].iloc[i] + 0.01, muscle + ' ', fontsize=10, ha='right')
    # # Set the labels and title
    # ax1.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    # ax1.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    # ax1.set_title('Comparison of Muscle Group Metabolic Rate Differences (Stance Phase)', fontsize=14)
    # ax1.axhline(0, color='black', linewidth=0.8)
    # ax1.axvline(0, color='black', linewidth=0.8)
    # # Add a dotted grey line to represent y=x
    # ax1.plot([-0.6, 0.15], [-0.6, 0.15], linestyle='--', color='grey', label='y=x')
    # # Ensure the plot is square and axes have the same bounds
    # ax1.set_aspect('equal', 'box')
    # ax1.legend(fontsize=12)
    
    # # Swing phase scatter plot
    # ax2.errorbar(prevswing['average'], currentswing_df['Difference (W/kg)'], xerr=prevswing_errors, fmt='o', color='blue', label='Muscle Groups')
    # # Add labels for each point
    # for i, muscle in enumerate(currentswing_df['Muscle']):
    #     ax2.text(prevswing['average'].iloc[i], currentswing_df['Difference (W/kg)'].iloc[i] + 0.01, muscle + ' ', fontsize=10, ha='right')
    # # Set the labels and title
    # ax2.set_xlabel('Previous Simulation Difference (W/kg)', fontsize=14)
    # ax2.set_ylabel('Current Simulation Difference (W/kg)', fontsize=14)
    # ax2.set_title('Comparison of Muscle Group Metabolic Rate Differences (Swing Phase)', fontsize=14)
    # ax2.axhline(0, color='black', linewidth=0.8)
    # ax2.axvline(0, color='black', linewidth=0.8)
    # # Add a dotted grey line to represent y=x
    # ax2.plot([-0.6, 0.15], [-0.6, 0.15], linestyle='--', color='grey', label='y=x')
    # # Ensure the plot is square and axes have the same bounds
    # ax2.set_aspect('equal', 'box')
    # ax2.legend(fontsize=12)
    
    
    # y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    # y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    # x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    # x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    # ax1.set_ylim([y_min, y_max])
    # ax1.set_xlim([x_min, x_max])
    # ax2.set_ylim([y_min, y_max])
    # ax2.set_xlim([x_min, x_max])
    # ax1.set_aspect('equal', 'box')
    # ax2.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_musclemetabolicvalidation_stance_swing_scatter_40.png')
    plt.show()

    # pdb.set_trace()
    return

# make sure the model probes are activated
def probeActivate(model):
    # % get the probeset
    probeset = model.getProbeSet();
    numProbes = probeset.getSize();

    # % need to loop through and set them all to be enabled hopefully
    for p in range(numProbes):
        probe = probeset.get(p);
        probe.setEnabled(True);

    # % update the model to be returned. 
    model.updProbeSet();
    return model

# make sure all coordinates and states have bounds
def constrainBoundsAllTrackedStates(track, problem, fractionExtraBoundSize, initialTime, guessfile):
    # Set each state's bounds based on a fraction (fractionExtraBoundSize) of
    # the range of the state's value from the tracked data. The bounds are
    # set as the following:
    #   - Lower bound: (minimum value) - fractionExtraBoundSize * (range of value)
    #   - Upper bound: (maximum value) + fractionExtraBoundSize * (range of value)

    # print('need to take a look at all my bounds.')
    # pdb.set_trace()

    # # Load the tracked_states file into a table to find the range of the
    # # coordinate values in the data.
    # mocoName = track.getName();

    # # we have a tracking problem - use the tracked states
    # trackedStatesFile = mocoName + '_tracked_states.sto';
    # trackedStatesTable = osim.TimeSeriesTable(trackedStatesFile);
    # trackedStatesTable.trimFrom(initialTime);
    # trackedStatesTable.trimTo(finalTime);

    # # Get all of the state names from the table column labels but omit
    # # the toe joint since it was locked in the model above. Store the
    # # state names in a vector to pass into the function that constrains the 
    # # bounds.
    # colLabelsStdVec = trackedStatesTable.getColumnLabels();

    # # get all the coordinate labels except for mtps
    # colLabelsCell = [eachlabel for eachlabel in colLabelsStdVec if 'mtp_angle' not in eachlabel];


    # attempting with guessfile
    guessStatesTable = osim.TimeSeriesTable(guessfile)
    colLabelsStdVec = guessStatesTable.getColumnLabels()
    # get all the coordinate labels except for mtps
    colLabelsCell = [eachlabel for eachlabel in colLabelsStdVec if 'jointset' in eachlabel]
    
    # pdb.set_trace()

    print('\n\nsetting bounds info for the problem states')
    # loop to get the coordinates max and min to set acceptable bounds and add to problem
    for i in range(len(colLabelsCell)):
        # want to open up the mtp and let it free
        if 'mtp' in colLabelsCell[i]:
            print('mtp skipped')
        elif 'tx' in colLabelsCell[i]:
            print('tx skipped')
        elif 'beta' in colLabelsCell[i]:
            print('patella beta skipped')
        else:
            thisStatePath = colLabelsCell[i];
            # stateColumn = trackedStatesTable.getDependentColumn(thisStatePath).to_numpy();
            stateColumn = guessStatesTable.getDependentColumn(thisStatePath).to_numpy();
            thisColMin = np.min(stateColumn);
            thisColMax = np.max(stateColumn);
            thisColRange = thisColMax - thisColMin;
            extraBoundSize = thisColRange * fractionExtraBoundSize;
            thisBounds = [thisColMin - extraBoundSize, thisColMax + extraBoundSize];
            print(thisStatePath)
            print(thisBounds)
            problem.setStateInfo(thisStatePath, thisBounds, [], []);


    # the bounds should be returned in the problem.
    return

# function to update the google drive with the local results/scripts
def syncDrives(localDir, destDir):
    # # method removes the tree and then copies the local one to dest
    # try:
    #     shutil.rmtree(destDir)
    # except OSError:
    #     print('\n###############################################################\nDirectory does not exist - check on your backup drive!!!')

    # # then replace with the up-to-date local
    # print('\nWait for file system to sync...')
    # time.sleep(1)
    # try:
    #     shutil.copytree(localDir, destDir)
    # except:
    #     print('\n###############################################################\nCould not copy files to backup - check on your backup drive!!!')


    # using flag to overwrite all the files in the destination.
    # above works cleaner, but is slow with google backup 
    print('\nWait for file system to sync...')
    shutil.copytree(localDir, destDir, dirs_exist_ok=True)
    time.sleep(1)

    print('... Files should be "up-to-date."')
    return

# function to update the google drive with the local results/scripts
def wipeSyncDrives(localDir, destDir):
    # method removes the tree and then copies the local one to dest
    try:
        shutil.rmtree(destDir)
    except OSError:
        print('\n###############################################################\nDirectory does not exist - check on your backup drive!!!')

    # then replace with the up-to-date local
    print('\nWait for file system to sync...')
    time.sleep(1)
    try:
        shutil.copytree(localDir, destDir)
    except:
        print('\n###############################################################\nCould not copy files to backup - check on your backup drive!!!')


    # # using flag to overwrite all the files in the destination.
    # # above works cleaner, but is slow with google backup 
    # shutil.copytree(localDir, destDir, dirs_exist_ok=True)
    # print('\nWait for file system to sync...')
    # time.sleep(1)

    # print('... Files should be "up-to-date."')
    return

# take in trajectory and populate a guess
def fillGuess(randomguess, trajectoryFile):    
    # % set our initial guesses
    twosteptraj = osim.MocoTrajectory(trajectoryFile);
    # % twosteptraj = MocoTrajectory('muscle_stateprescribe_grfprescribe_solution.sto');
    # % twosteptraj = MocoTrajectory('muscle_statetrack_grfprescribe_solution_100con.sto');

    # get the number of timesteps
    steps = twosteptraj.getNumTimes();
    # resample your random guess trajectory
    randomguess.resampleWithNumTimes(steps);
    
    # % go through and overwrite the states first
    randomstatenames = randomguess.getStateNames();
    # % this will cover joint values, speeds, muscle activations, and norm
    # % tendon force
    for s in range(len(randomstatenames)):
        statename = randomstatenames[s];
        try:
            # % temprandom = randomguess.getStateMat(statename);
            temp2step = twosteptraj.getStateMat(statename);
            randomguess.setState(statename,temp2step);       
        except:
            print('did not have state: %s' % statename)
    
    # % go through all the controls - excitations
    randomcontrolnames = randomguess.getControlNames();
    # % this covers all excitations and reserves
    for c in range(len(randomcontrolnames)):
        controlname = randomcontrolnames[c];
        try:
            # % temprandom = randomguess.getControlMat(controlname);
            temp2step = twosteptraj.getControlMat(controlname);
            randomguess.setControl(controlname, temp2step);
        except:
            print('did not have control: %s' % controlname)
    
    # % go through others??
    # % randomparamnames = randomguess.getParameterNames();
    # % this is empty in the normal condition
        
    # % multipliers
    randommultnames = randomguess.getMultiplierNames();
    for m in range(len(randommultnames)):
        multname = randommultnames[m];
        # % temprandom = randomguess.getMultiplierMat(multname)
        try:
            temp2step = twosteptraj.getMultiplierMat(multname);
            randomguess.setMultiplier(multname, temp2step);
        except:
            print('did not have the multiplier in the 2 step problem solution');
        
    
    
    # % now for the implicit derivatives
    randomderivnames = randomguess.getDerivativeNames();
    for d in range(len(randomderivnames)):
        derivname = randomderivnames[d];
        try:
            # % temprandom = randomguess.getDerivativeMat(derivname);
            temp2step = twosteptraj.getDerivativeMat(derivname);
            randomguess.setDerivative(derivname, temp2step);
        except:
            print('did not have deriv: %s' % derivname)

    newguess = randomguess;
    return newguess

# results figure for the 2.7 m/s grid search
def resultsGridSearch27(figurepath):
    import plotly.graph_objects as go
    # load in the data and get the right sheet
    df = pd.read_excel(figurepath, sheet_name="2.7 grid_figure (2)")
    # Set the 'stiffness' column as the index
    df.set_index('stiffness', inplace=True)
    # now load in the 4 m/s data
    df4 = pd.read_excel(figurepath, sheet_name="4 grid_figure (2)")
    # Set the 'stiffness' column as the index
    df4.set_index('stiffness', inplace=True)
    
    # Extract the column labels (lengths) and convert them to numeric values
    lengths = pd.to_numeric(df.columns)
    lengths4 = pd.to_numeric(df4.columns)
    # Create a meshgrid for stiffness and lengths
    stiffness, length = np.meshgrid(df.index, lengths)
    stiffness4, length4 = np.meshgrid(df4.index, lengths4)
    # Extract the speed values from the DataFrame
    metabolic = df.values.T
    metabolic4 = df4.values.T

    # Create a 3D surface plot with two surfaces
    fig = go.Figure()
    
    # Add the first surface for 2.7 m/s
    fig.add_trace(go.Surface(z=metabolic, x=stiffness, y=length, colorscale='Viridis', name='2.7 m/s'))

    # Add the second surface for 4.0 m/s
    fig.add_trace(go.Surface(z=metabolic4, x=stiffness4, y=length4, colorscale='Cividis', name='4.0 m/s'))

    # Set the axis labels
    fig.update_layout(
        title='3D Surface Plot of Metabolic Reduction',
        scene=dict(
            xaxis_title='Stiffness',
            yaxis_title='Length',
            zaxis_title='Metabolic Reduction (% Change)'
        ),
        height=800  # Adjust the height of the figure
    )
    fig.show()
    return

# function for plotting all the fiber lengths and mtu lengths
def fiberPlotting(modelfile, natfibers, exofibers, normalizedornot):
    # Load in the model 
    model = osim.Model(modelfile)
    st = model.initSystem()
    # Get the muscle set
    muscles = model.getMuscles()
    # Get the muscle names
    muscle_names = [muscle.getName() for muscle in muscles]
    # Get the muscle paths
    muscle_paths = [muscle.getAbsolutePathString() for muscle in muscles]
    
    # load the natfiber and exo fiber data
    natfiber = osim.TimeSeriesTable(natfibers)
    exofiber = osim.TimeSeriesTable(exofibers)
    # get the column labels
    natfiber_labels = natfiber.getColumnLabels()
    exofiber_labels = exofiber.getColumnLabels()
    # get the time
    nattime = natfiber.getIndependentColumn()
    exotime = exofiber.getIndependentColumn()
    # get the number of fibers
    numfibers = len(natfiber_labels)
    
    # Create a figure with subplots for each muscle for nat case.
    fig, axes = plt.subplots(ncols=1, nrows=int(len(muscle_names)/2), figsize=(4, len(muscle_names) * 2.5))
    # Loop through each muscle and plot the fiber lengths
    for i, muscle_name in enumerate(muscle_names):
        if '_r' in muscle_name:
            ax = axes[i]
            mus = muscles.get(i)
            # print(mus.getName())
            # print(muscle_name)

            nat_optimal = mus.get_optimal_fiber_length()
            if normalizedornot:
                nat_fiber_length = natfiber.getDependentColumn('/forceset/' + muscle_name + '|normalized_fiber_length').to_numpy()
                nat_mtu_length = natfiber.getDependentColumn('/forceset/' + muscle_name + '/path|length').to_numpy() / nat_optimal
                # and get the tendon slack length
                nat_tendon_slack_len = mus.get_tendon_slack_length()
                # nat_tendon_slack_len = np.ones(len(nat_fiber_length)) * nat_tendon_slack_len
                nat_actual_tendon = natfiber.getDependentColumn('/forceset/' + muscle_name + '|tendon_length').to_numpy() / nat_tendon_slack_len
                # Plot the fiber lengths
                ax.plot(nattime, nat_fiber_length, label='Norm. fiber length', color='blue')
                # ax.plot(nattime, nat_mtu_length, label='Norm. MTU length', color='red')
                # ax.plot(nattime, nat_tendon_slack_len, label='Tendon Slack Length', color='green')
                ax.plot(nattime, nat_actual_tendon, label='Norm. Tendon Length', color='purple')
                ax.plot(nattime, np.ones(len(nattime)), color='grey', linestyle='--', label='Optimal Fiber Length')
                ax.plot(nattime, np.ones(len(nattime))*0.5, color='grey', linestyle='--', label='0.5 * Optimal Fiber Length', antialiased=False)
                # ax.plot(nattime, np.ones(len(nattime)) * nat_optimal, color='orange', linestyle='--', label='Optimal Fiber Length') 
            
            else:
                nat_fiber_length = natfiber.getDependentColumn('/forceset/' + muscle_name + '|fiber_length').to_numpy()
                nat_mtu_length = natfiber.getDependentColumn('/forceset/' + muscle_name + '/path|length').to_numpy()
                # and get the tendon slack length
                nat_tendon_slack_len = mus.get_tendon_slack_length()
                nat_tendon_slack_len = np.ones(len(nat_fiber_length)) * nat_tendon_slack_len
                nat_actual_tendon = natfiber.getDependentColumn('/forceset/' + muscle_name + '|tendon_length').to_numpy()
                # Plot the fiber lengths
                ax.plot(nattime, nat_fiber_length, label='fiber length', color='blue')
                ax.plot(nattime, nat_mtu_length, label='MTU length', color='red')
                ax.plot(nattime, nat_tendon_slack_len, label='Tendon Slack Length', color='green')
                ax.plot(nattime, nat_actual_tendon, label='Actual Tendon Length', color='purple')
                # ax.axhline(nat_optimal, color='orange', linestyle='--', label='Optimal Fiber Length', antialiased=False)
                ax.plot(nattime, np.ones(len(nattime)) * nat_optimal, color='orange', linestyle='--', label='Optimal Fiber Length') 
            
            
            # Set the title and labels
            ax.set_title(f'{muscle_name} lengths')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Length (m)')
            # ax.axhline(0, color='grey', linestyle='--')
            ax.plot(nattime, np.zeros(len(nattime)), color='grey', linestyle='--')
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    print('\nNatural Fiber Lengths')
    # plt.show()
    # Display the figures side by side
    fig1_html = mpld3.fig_to_html(fig)
    plt.close()

    # # create a figure with subplots for each muscle for exo case. 
    # fig, axes = plt.subplots(ncols=1, nrows=len(muscle_names), figsize=(10, len(muscle_names) * 3))
    # # Loop through each muscle and plot the fiber lengths
    # for i, muscle_name in enumerate(muscle_names):
    #     ax = axes[i]
    #     mus = muscles.get(i)
    #     # print(mus.getName())
    #     # print(muscle_name)
    #     # Get the fiber length data for the current muscle
    #     exo_fiber_length = exofiber.getDependentColumn('/forceset/' + muscle_name + '|fiber_length').to_numpy()
    #     exo_mtu_length = exofiber.getDependentColumn('/forceset/' + muscle_name + '/path|length').to_numpy()
    #     # and get the tendon slack length
    #     exo_tendon_slack_len = mus.get_tendon_slack_length()
    #     exo_tendon_slack_len = np.ones(len(exo_fiber_length)) * exo_tendon_slack_len
    #     # Plot the fiber lengths
    #     ax.plot(exotime, exo_fiber_length, label='fiber length', color='blue')
    #     ax.plot(exotime, exo_mtu_length, label='MTU length', color='red')
    #     ax.plot(exotime, exo_tendon_slack_len, label='Tendon Slack Length', color='green')
    #     # Set the title and labels
    #     ax.set_title(f'{muscle_name} lengths')
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Length (m)')
    #     ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    #     ax.legend()

    # # Adjust layout
    # plt.tight_layout()
    # print('\nExo Fiber Lengths')
    # plt.show()
    return fig1_html

# figure for the force generating capacity of the muscles throughout the gait. 
def forceCapacityPlotting(modelfile, natoutputs, exooutputs):
    # load the model
    model = osim.Model(modelfile)
    st = model.initSystem()
    # get the muscle set
    muscles = model.getMuscles()
    # get the muscle names
    muscle_names = [muscle.getName() for muscle in muscles]
    # get the muscle paths
    muscle_paths = [muscle.getAbsolutePathString() for muscle in muscles]
    # load the natoutputs and exooutputs
    natoutput = osim.TimeSeriesTable(natoutputs)
    exooutput = osim.TimeSeriesTable(exooutputs)
    # get the column labels
    natoutput_labels = natoutput.getColumnLabels()
    exooutput_labels = exooutput.getColumnLabels()
    # get the time
    nattime = natoutput.getIndependentColumn()
    exotime = exooutput.getIndependentColumn()
    
    # loop through the muscels and plot the force generating capacity for nat
    fig, axes = plt.subplots(ncols=1, nrows=int(len(muscle_names)/2), figsize=(4, len(muscle_names) * 2.5))
    for i, muscle_name in enumerate(muscle_names):
        if '_r' in muscle_name:
            ax = axes[i]
            mus = muscles.get(i)
            natMaxIsometricForce = mus.get_max_isometric_force()
            
            # print(mus.getName())
            # print(muscle_name)
            # Get the fiber  data for the current muscle
            nat_passivelengthmult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|passive_force_multiplier').to_numpy()
            nat_activelengthmult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|active_force_length_multiplier').to_numpy()
            nat_velocitymult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|force_velocity_multiplier').to_numpy()
            
            activity = np.ones(len(nat_velocitymult))
            nat_force_gen_capacity = natMaxIsometricForce * ((activity * nat_activelengthmult * nat_velocitymult) + nat_passivelengthmult)
            nat_activation = natoutput.getDependentColumn('/forceset/' + muscle_name + '|activation').to_numpy()
            nat_actual_force = natMaxIsometricForce * ((nat_activation * nat_activelengthmult * nat_velocitymult) + nat_passivelengthmult)
            
            nat_force_along_tendon = natoutput.getDependentColumn('/forceset/' + muscle_name + '|fiber_force_along_tendon').to_numpy()
            nat_active_fiber_force = natoutput.getDependentColumn('/forceset/' + muscle_name + '|active_fiber_force').to_numpy()
            nat_fiber_force = natoutput.getDependentColumn('/forceset/' + muscle_name + '|fiber_force').to_numpy()
            
            # Plot the fiber force generating capacity, optimal force, and the force along the tendon, and active force.
            ax.plot(nattime, nat_force_gen_capacity, label='Force Generating Capacity', color='blue')
            ax.plot(nattime, nat_actual_force, label='Actual Force', color='red')
            ax.plot(nattime, nat_fiber_force, label='Fiber Force', color='green', linestyle='--')
            
            # Set the title and labels
            ax.set_title(f'{muscle_name} Forces')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)')
            # ax.axhline(0, color='grey', linestyle='--', linewidth=1.0)
            ax.plot(nattime, np.zeros(len(nattime)), color='grey', linestyle='--')
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    print('\nNat Fiber Forces')
    # plt.show()
    fig1_html = mpld3.fig_to_html(fig)
    plt.close()

    # loop through the muscels and plot the force generating capacity for nat - just plot multipliers
    fig, axes = plt.subplots(ncols=1, nrows=int(len(muscle_names)/2), figsize=(4, len(muscle_names) * 2.5))
    for i, muscle_name in enumerate(muscle_names):
        if '_r' in muscle_name:
            ax = axes[i]
            mus = muscles.get(i)
            natMaxIsometricForce = mus.get_max_isometric_force()
            
            # print(mus.getName())
            # print(muscle_name)
            # Get the fiber  data for the current muscle
            nat_passivelengthmult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|passive_force_multiplier').to_numpy()
            nat_activelengthmult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|active_force_length_multiplier').to_numpy()
            nat_velocitymult = natoutput.getDependentColumn('/forceset/' + muscle_name + '|force_velocity_multiplier').to_numpy()
            
            activity = np.ones(len(nat_velocitymult))
            nat_force_gen_capacity = natMaxIsometricForce * ((activity * nat_activelengthmult * nat_velocitymult) + nat_passivelengthmult)
            nat_activation = natoutput.getDependentColumn('/forceset/' + muscle_name + '|activation').to_numpy()
            nat_actual_force = natMaxIsometricForce * ((nat_activation * nat_activelengthmult * nat_velocitymult) + nat_passivelengthmult)
            
            nat_force_along_tendon = natoutput.getDependentColumn('/forceset/' + muscle_name + '|fiber_force_along_tendon').to_numpy()
            nat_active_fiber_force = natoutput.getDependentColumn('/forceset/' + muscle_name + '|active_fiber_force').to_numpy()
            nat_fiber_force = natoutput.getDependentColumn('/forceset/' + muscle_name + '|fiber_force').to_numpy()
            
            # Plot the fiber force generating capacity, optimal force, and the force along the tendon, and active force.
            ax.plot(nattime, nat_passivelengthmult, label='Passive length mult', color='blue')
            ax.plot(nattime, nat_activelengthmult, label='Active length mult', color='red')
            ax.plot(nattime, nat_velocitymult, label='velocity mult', color='green', linestyle='--')
            
            
            # Set the title and labels
            ax.set_title(f'{muscle_name} Multipliers')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Multiplier')
            # ax.axhline(0, color='grey', linestyle='--', linewidth=1.0)
            ax.plot(nattime, np.zeros(len(nattime)), color='grey', linestyle='--')
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    print('\nNat Fiber multipliers')
    # plt.show()
    fig2_html = mpld3.fig_to_html(fig)
    plt.close()

    return fig1_html, fig2_html

# wrapper function for the two above functions fiberPlotting and forceCapacityPlotting
def fiberPropertiesPlotting(modelfile, natfibers, exofibers, natoutputs, exooutputs, normornot):
    # get the fiber lengths and mtu lengths
    fiber_html = fiberPlotting(modelfile, natfibers, exofibers, normornot)
    # get the force generating capacity of the muscles
    force_html, mult_html = forceCapacityPlotting(modelfile, natoutputs, exooutputs)
    display_html(f"""
    <div style="display: flex; justify-content: space-around; background-color: #f4f4f4; padding: 5px; border-radius: 5px;">
        <div style="color: black;">{fiber_html}</div>
        <div style="color: black;">{force_html}</div>
        <div style="color: black;">{mult_html}</div>
    </div>
    """, raw=True)
    return 

# function for plotting the toe trajectories through different files. 
def toeTrajectoryPlotting(nattoefiler, nattoefilel, exotoefiler, exotoefilel):
    # load in the files
    nat_toe_r = osim.TimeSeriesTable(nattoefiler)
    nat_toe_l = osim.TimeSeriesTable(nattoefilel)
    exo_toe_r = osim.TimeSeriesTable(exotoefiler)
    exo_toe_l = osim.TimeSeriesTable(exotoefilel)
    # get the time
    nattime = nat_toe_r.getIndependentColumn()
    exotime = exo_toe_r.getIndependentColumn()

    # Extract the toe trajectory data
    nat_toe_l_x = nat_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_1').to_numpy()
    nat_toe_l_y = nat_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_2').to_numpy()
    nat_toe_l_z = nat_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_3').to_numpy()
    exo_toe_l_x = exo_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_1').to_numpy()           
    exo_toe_l_y = exo_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_2').to_numpy()
    exo_toe_l_z = exo_toe_l.getDependentColumn('/bodyset/toes_l/bigToeL|position_3').to_numpy()

    nat_toe_r_x = nat_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_1').to_numpy()
    nat_toe_r_y = nat_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_2').to_numpy()
    nat_toe_r_z = nat_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_3').to_numpy()
    exo_toe_r_x = exo_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_1').to_numpy()
    exo_toe_r_y = exo_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_2').to_numpy()
    exo_toe_r_z = exo_toe_r.getDependentColumn('/bodyset/toes_r/bigToeR|position_3').to_numpy()

    # create a figure with 3 subplots that has 3 rows, one for each of the toe position trajectories 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    axes[0].plot(nattime, nat_toe_l_x, label='Natural Left Toe X', color='blue')
    axes[0].plot(exotime, exo_toe_l_x, label='Exo Left Toe X', color='red')
    axes[0].set_title('Toe X Position')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('X Position (m)')
    axes[0].legend()

    # Plot Y position
    axes[1].plot(nattime, nat_toe_l_y, label='Natural Left Toe Y', color='blue')
    axes[1].plot(exotime, exo_toe_l_y, label='Exo Left Toe Y', color='red')
    axes[1].set_title('Toe Y Position')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()

    # Plot Z position
    axes[2].plot(nattime, nat_toe_l_z, label='Natural Left Toe Z', color='blue')
    axes[2].plot(exotime, exo_toe_l_z, label='Exo Left Toe Z', color='red')
    axes[2].set_title('Toe Z Position')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Z Position (m)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # now do the same for the right side
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    axes[0].plot(nattime, nat_toe_r_x, label='Natural Right Toe X', color='blue')
    axes[0].plot(exotime, exo_toe_r_x, label='Exo Right Toe X', color='red')
    axes[0].set_title('Toe X Position')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('X Position (m)')
    axes[0].legend()

    # Plot Y position
    axes[1].plot(nattime, nat_toe_r_y, label='Natural Right Toe Y', color='blue')
    axes[1].plot(exotime, exo_toe_r_y, label='Exo Right Toe Y', color='red')
    axes[1].set_title('Toe Y Position')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()

    # Plot Z position
    axes[2].plot(nattime, nat_toe_r_z, label='Natural Right Toe Z', color='blue')
    axes[2].plot(exotime, exo_toe_r_z, label='Exo Right Toe Z', color='red')
    axes[2].set_title('Toe Z Position')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Z Position (m)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    
    
    return

###### - also generate a muscle activity plot between the conditions and save
# TODO: add muscle activity plotting here
# TODO: save or output the activity stuff I need for power analysis
# TODO: actually just copy structure to a new method... 

# function for plotting the muscle activity across different conditions
def muscleActivityPlotting(natoutputs, exooutputs):
    # load the natoutputs and exooutputs
    natoutput = osim.TimeSeriesTable(natoutputs)
    exooutput = osim.TimeSeriesTable(exooutputs)
    # get the column labels
    natoutput_labels = natoutput.getColumnLabels()
    exooutput_labels = exooutput.getColumnLabels()
    nat_activation_labels = [label for label in natoutput_labels if '_r|activation' in label]
    exo_activation_labels = [label for label in exooutput_labels if '_r|activation' in label]
    
    # allocate the vectors for the muscle activation peaks and stats
    nat_activation_peaks = np.zeros(len(nat_activation_labels))
    exo_activation_peaks = np.zeros(len(exo_activation_labels))

    nat_simple = {}
    exo_simple = {}
    muscle_isos = {
        '/forceset/bfsh_r|activation': 1.0651e3,
        '/forceset/gasmed_r|activation': 2.558810662649832e+03,
        '/forceset/hipextensors|activation': 8.708971513480666e+02,
        '/forceset/hipflexors|activation': 1.120310648038673e+03,
        '/forceset/recfem_r|activation': 1.738712659840884e+03,
        '/forceset/semimem_r|activation': 4.811580119041698e+02,
        '/forceset/soleus_r|activation': 5.112942567856007e+03,
        '/forceset/dorsiflexors|activation': 1.009219972555261e+03,
        '/forceset/vasint_r|activation': 3.935155836924216e+03
    }

    # get the time
    nattime = natoutput.getIndependentColumn()
    exotime = exooutput.getIndependentColumn()

    # loop through the muscels and plot the force generating capacity for nat
    fig, axes = plt.subplots(ncols=1, nrows=int(len(nat_activation_labels)), figsize=(4, len(nat_activation_labels) * 2.5))
    for i, muscle_name in enumerate(nat_activation_labels):
            ax = axes[i]
            nat_activation = natoutput.getDependentColumn(muscle_name).to_numpy()
            exo_activation = exooutput.getDependentColumn(muscle_name).to_numpy()

            # store activation data
            if 'tibant' in muscle_name:
                nat_activation_peaks[i] = np.max(nat_activation[len(nat_activation)//2:])
                exo_activation_peaks[i] = np.max(exo_activation[len(exo_activation)//2:])
            else: 
                nat_activation_peaks[i] = np.max(nat_activation)
                exo_activation_peaks[i] = np.max(exo_activation)
            print(f'{muscle_name} - nat: {nat_activation_peaks[i]} - exo: {exo_activation_peaks[i]}')
            print(f'{muscle_name} - difference: {nat_activation_peaks[i] - exo_activation_peaks[i]}')
            print('\n')
            # plotting 
            ax.plot(nattime, nat_activation, label='Natural Activation', color='orange')
            ax.plot(exotime, exo_activation, label='Exo Activation', color='purple')
            # Set the title and labels
            ax.set_title(f'{muscle_name}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Activation')
            # ax.axhline(0, color='grey', linestyle='--', linewidth=1.0)
            ax.plot(nattime, np.zeros(len(nattime)), color='grey', linestyle='--')
            ax.legend()
            ax.set_ylim(0, 1)
    
    diff_peaks = exo_activation_peaks - nat_activation_peaks
    print(f'Peak differences: {diff_peaks}')

    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    # fig1_html = mpld3.fig_to_html(fig)
    # plt.close()

    nat_simple['quads'] = (natoutput.getDependentColumn('/forceset/vasint_r|activation').to_numpy() * muscle_isos['/forceset/vasint_r|activation'] + 
                           natoutput.getDependentColumn('/forceset/recfem_r|activation').to_numpy() * muscle_isos['/forceset/recfem_r|activation']) / (muscle_isos['/forceset/vasint_r|activation'] + muscle_isos['/forceset/recfem_r|activation'])
    nat_simple['hamstrings'] = (natoutput.getDependentColumn('/forceset/bfsh_r|activation').to_numpy() * muscle_isos['/forceset/bfsh_r|activation'] +
                            natoutput.getDependentColumn('/forceset/semimem_r|activation').to_numpy() * muscle_isos['/forceset/semimem_r|activation']) / (muscle_isos['/forceset/bfsh_r|activation'] + muscle_isos['/forceset/semimem_r|activation'])
    nat_simple['hipflexors'] = natoutput.getDependentColumn('/forceset/psoas_r|activation').to_numpy()
    nat_simple['hipextensors'] = natoutput.getDependentColumn('/forceset/glmax2_r|activation').to_numpy()
    nat_simple['plantarflexors'] = (natoutput.getDependentColumn('/forceset/soleus_r|activation').to_numpy() * muscle_isos['/forceset/soleus_r|activation'] +
                            natoutput.getDependentColumn('/forceset/gasmed_r|activation').to_numpy() * muscle_isos['/forceset/gasmed_r|activation']) / (muscle_isos['/forceset/soleus_r|activation'] + muscle_isos['/forceset/gasmed_r|activation'])
    nat_simple['dorsiflexors'] = natoutput.getDependentColumn('/forceset/tibant_r|activation').to_numpy()

    exo_simple['quads'] = (exooutput.getDependentColumn('/forceset/vasint_r|activation').to_numpy() * muscle_isos['/forceset/vasint_r|activation'] + 
                           exooutput.getDependentColumn('/forceset/recfem_r|activation').to_numpy() * muscle_isos['/forceset/recfem_r|activation']) / (muscle_isos['/forceset/vasint_r|activation'] + muscle_isos['/forceset/recfem_r|activation'])
    exo_simple['hamstrings'] = (exooutput.getDependentColumn('/forceset/bfsh_r|activation').to_numpy() * muscle_isos['/forceset/bfsh_r|activation'] +
                            exooutput.getDependentColumn('/forceset/semimem_r|activation').to_numpy() * muscle_isos['/forceset/semimem_r|activation']) / (muscle_isos['/forceset/bfsh_r|activation'] + muscle_isos['/forceset/semimem_r|activation'])
    exo_simple['hipflexors'] = exooutput.getDependentColumn('/forceset/psoas_r|activation').to_numpy()
    exo_simple['hipextensors'] = exooutput.getDependentColumn('/forceset/glmax2_r|activation').to_numpy()
    exo_simple['plantarflexors'] = (exooutput.getDependentColumn('/forceset/soleus_r|activation').to_numpy() * muscle_isos['/forceset/soleus_r|activation'] +
                            exooutput.getDependentColumn('/forceset/gasmed_r|activation').to_numpy() * muscle_isos['/forceset/gasmed_r|activation']) / (muscle_isos['/forceset/soleus_r|activation'] + muscle_isos['/forceset/gasmed_r|activation'])
    exo_simple['dorsiflexors'] = exooutput.getDependentColumn('/forceset/tibant_r|activation').to_numpy()
    
    
    fig, axes = plt.subplots(nrows=len(nat_simple), ncols=1, figsize=(8, len(nat_simple) * 3))
    for i, (muscle_group, nat_activation) in enumerate(nat_simple.items()):
        exo_activation = exo_simple[muscle_group]
        ax = axes[i]
        ax.plot(nattime, nat_activation, label=f'Natural {muscle_group}', color='orange')
        ax.plot(exotime, exo_activation, label=f'Exo {muscle_group}', color='purple')
        ax.set_title(f'{muscle_group.capitalize()} Activation', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Activation', fontsize=12)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    # plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_muscle_activation_comparison.png')
    plt.show()

    muscle_diffs = {}
    for muscle_group in nat_simple.keys():
        muscle_diffs[muscle_group] = np.max(exo_simple[muscle_group]) - np.max(nat_simple[muscle_group])
        print(f'{muscle_group} - difference: {muscle_diffs[muscle_group]}')
        


    return

# create a figure for plotting the exotendon forces across different conditions. 
def exotendonTensionPlotting(modelfile, resultsdir, customresultsdirs=None):
    import matplotlib as mpl
    import pandas as pd
    import os
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        # Configure mathtext to use Arial where possible (custom mapping)
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'
    })
    
    wantdots = False
    if wantdots: 
        styles = [(0, (1, 1)), '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2, 2, 2))]  # Six distinct styles
    else: 
        styles = ['-', '-', '-', '-', '-', '-']
    testmap = {}
    testmap['1'] = '#d01c8b'
    testmap['2'] = '#eeffee'
    testmap['3'] = '#e4ffe4'
    testmap['4'] = '#d9ffd9'
    testmap['5'] = '#cffff4'
    testmap['6'] = '#c4ffc4'
    testmap['7'] = '#baffa3'
    testmap['8'] = '#b0ffb0'
    testmap['9'] = '#a5fba5'
    testmap['10'] = '#9af19a'
    testmap['11'] = '#8fe78f'
    testmap['12'] = '#84dd84'
    testmap['13'] = '#79d379'
    testmap['14'] = '#6ec96e'
    testmap['15'] = '#63bf63'
    testmap['16'] = '#58b558'
    testmap['17'] = '#4dab4d'
    testmap['18'] = '#42a042'
    testmap['19'] = '#379637'
    testmap['20'] = '#2c8c2c'
    testmap['21'] = '#218221'
    testmap['22'] = '#167916'
    testmap['23'] = '#0b6f0b'
    testmap['24'] = '#006600'
    testmap['25'] = '#004400'
    data = np.linspace(1,25,25)
    # Create a bar plot

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the data
    bars = ax.bar(data, data, color=[testmap[str(int(d))] for d in data])

    # Set the labels and title
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Values')
    ax.set_title('Bar Plot with Custom Colors')

    # Show the plot
    # plt.show()
    
    # Create a color scale descriptor
    fig, ax = plt.subplots(figsize=(5, 1), dpi=500)
    # Create a color bar with the same color map
    cmap = mcolors.ListedColormap([testmap[str(int(d))] for d in data])
    norm = mcolors.BoundaryNorm(data, cmap.N)
    cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    # Set custom tick labels
    cb.set_ticks([data[0], data[-1]])
    cb.set_ticklabels(['Increasing cost', 'Saving cost'], fontsize=14)
    norm = mcolors.BoundaryNorm(data, cmap.N)
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_exotendonTensions_40_colorscale.png')
    # plt.show()

    
    
    # create a color map dict for all the exotendon conditions
    exocolormap = {}

    # ### these are the colors for the fixed duration strides
    # exocolormap['l = 0.0718 | k = 240'] = '#d01c8b'#'#ff6666'
    # exocolormap['l = 0.1435 | k = 240'] = '#f1b6da'#'#ff9999'
    # exocolormap['l = 0.0718 | k = 180'] = '#eeffee'
    # exocolormap['l = 0.5741 | k = 30'] = '#e4ffe4'
    # exocolormap['l = 0.4305 | k = 30'] = '#d9ffd9'
    # exocolormap['l = 0.5741 | k = 60'] = '#cffff4'
    # exocolormap['l = 0.2870 | k = 30'] = '#c4ffc4'
    # exocolormap['l = 0.1435 | k = 30'] = '#baffa3'
    # exocolormap['l = 0.4305 | k = 60'] = '#b0ffb0'
    # exocolormap['l = 0.0718 | k = 30'] = '#a5fba5'
    # exocolormap['l = 0.1435 | k = 180'] = '#9af19a'
    # exocolormap['l = 0.2870 | k = 240'] = '#8fe78f'
    # exocolormap['l = 0.5741 | k = 120'] = '#84dd84'
    # exocolormap['l = 0.2870 | k = 60'] = '#79d379'
    # exocolormap['l = 0.5741 | k = 180'] = '#6ec96e'
    # exocolormap['l = 0.1435 | k = 60'] = '#63bf63'
    # exocolormap['l = 0.4305 | k = 120'] = '#58b558'
    # exocolormap['l = 0.0718 | k = 60'] = '#4dab4d'
    # exocolormap['l = 0.2870 | k = 120'] = '#42a042'
    # exocolormap['l = 0.2870 | k = 180'] = '#379637'
    # exocolormap['l = 0.0718 | k = 120'] = '#2c8c2c'
    # exocolormap['l = 0.5741 | k = 240'] = '#218221'
    # exocolormap['l = 0.4305 | k = 240'] = '#167916'
    # exocolormap['l = 0.4305 | k = 180'] = '#0b6f0b'
    # exocolormap['l = 0.1435 | k = 120'] = '#006600'
    
    ### these are the colors for the minimums of the bowls - actual mins
    exocolormap['l = 0.0718 | k = 240'] = '#d01c8b'
    exocolormap['l = 0.5741 | k = 30'] = '#eeffee'
    exocolormap['l = 0.4305 | k = 30'] = '#e4ffe4'
    exocolormap['l = 0.5741 | k = 60'] = '#d9ffd9'
    exocolormap['l = 0.2870 | k = 30'] = '#cffff4'
    exocolormap['l = 0.1435 | k = 30'] = '#c4ffc4'
    exocolormap['l = 0.4305 | k = 60'] = '#baffa3'
    exocolormap['l = 0.0718 | k = 30'] = '#b0ffb0'
    exocolormap['l = 0.1435 | k = 240'] = '#a5fba5'
    exocolormap['l = 0.5741 | k = 120'] = '#9af19a'
    exocolormap['l = 0.2870 | k = 60'] = '#8fe78f'
    exocolormap['l = 0.5741 | k = 180'] = '#84dd84'
    exocolormap['l = 0.0718 | k = 180'] = '#79d379'
    exocolormap['l = 0.1435 | k = 60'] = '#6ec96e'
    exocolormap['l = 0.4305 | k = 120'] = '#63bf63'
    exocolormap['l = 0.0718 | k = 60'] = '#58b558'
    exocolormap['l = 0.2870 | k = 120'] = '#4dab4d'
    exocolormap['l = 0.1435 | k = 180'] = '#42a042'
    exocolormap['l = 0.5741 | k = 240'] = '#379637'
    exocolormap['l = 0.0718 | k = 120'] = '#2c8c2c'
    exocolormap['l = 0.2870 | k = 240'] = '#218221'
    exocolormap['l = 0.4305 | k = 180'] = '#167916'
    exocolormap['l = 0.1435 | k = 120'] = '#0b6f0b'
    exocolormap['l = 0.2870 | k = 180'] = '#006600'
    exocolormap['l = 0.4305 | k = 240'] = '#004400'
    

    # create a filler for all the max tension values. 
    maxtension = {}
    top5tension = {}
    selectedtension = {}

    if customresultsdirs==None: 
        # first establish the results dir. 
        resultsfiles = os.listdir(resultsdir)
        # get the exotendon tension files
        exotendon_files = [file for file in resultsfiles if 'HOBL' in file]; # print(exotendon_files)
    else: 
        # Assuming the dictionary is named 'my_dict'
        exotendon_files = []
        for key in customresultsdirs:
            exotendon_files.append(customresultsdirs[key])

    print(exotendon_files)

    datacompare = {}

    # create a figure for all of the exo forces. 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), dpi=500)
    # loop through each of the files
    for f in exotendon_files: 
        # print(f)
        if '3exoActMet' not in f and '4exoActMet' not in f:
            # get the parameters of the exotendon based on the file name
            exotendon_params = f.split('_')
            print(exotendon_params)
            # get the stiffness of the exotendon
            if 'compliant' in exotendon_params:
                exostiff = 'k = 60'
            elif 'stiff' in exotendon_params:
                exostiff = 'k = 180'
            elif 'stiff2' in exotendon_params:
                exostiff = 'k = 240'
            elif 'compliant2' in exotendon_params:
                exostiff = 'k = 30'
            else:
                exostiff = 'k = 120'
            # get the length of the exotendon
            if 'long' in exotendon_params:
                exolength = 'l = 0.4305'
            elif 'short' in exotendon_params:
                exolength = 'l = 0.1435'
            elif 'long2' in exotendon_params:
                exolength = 'l = 0.5741'
            elif 'short2' in exotendon_params:
                exolength = 'l = 0.0718'
            else:
                exolength = 'l = 0.2870'
            
            # print(exostiff)
            # print(exolength)
            # load in the exotendon data
            exodata = osim.TimeSeriesTable(os.path.join(resultsdir, f))
            # print(os.path.join(resultsdir,f))
            exotime = exodata.getIndependentColumn()
            tension = exodata.getDependentColumn('/forceset/HOBL|tension').to_numpy()
            # interpolate and resample 
            newtime = np.linspace(exotime[0], exotime[-1], 1000)
            tension_interp = np.interp(newtime, exotime, tension)

            # add the max tension to the max tension dict
            maxtension[exolength+' | '+exostiff] = np.max(tension_interp)

            # plot the data
            if exostiff == 'k = 120' and exolength == 'l = 0.2870':
                if not wantdots:
                    ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                else:
                    ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                print('k=120 | l=0.29')
                print(np.max(tension_interp))
            # also only grab the top five
            elif exostiff == 'k = 240' and exolength == 'l = 0.4305':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp) #styles[1]
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=240 | l=0.43')
                print(np.max(tension_interp))
            elif exostiff == 'k = 30' and exolength == 'l = 0.4305':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=30 | l=0.43')
                print(np.max(tension_interp))
            elif exostiff == 'k = 240' and exolength == 'l = 0.1435':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=240 | l=0.14')
                print(np.max(tension_interp))
            # elif exostiff == 'k = 30' and exolength == 'l = 0.2870':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='lightcoral')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=30 | l=0.29')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 60' and exolength == 'l = 0.4305':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='red')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=60 | l=0.43')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 60' and exolength == 'l = 0.5741':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkred')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=60 | l=0.57')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 180' and exolength == 'l = 0.2870':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='red')
            #     top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=180 | l=0.29')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 180' and exolength == 'l = 0.4305':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkred')
            #     top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=180 | l=0.43')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 240' and exolength == 'l = 0.2870':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkorange')
            #     top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=240 | l=0.29')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 120' and exolength == 'l = 0.1435':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='lightcoral')
            #     top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=120 | l=0.14')
            #     print(np.max(tension_interp))
            else:
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])

    # exocolormap['l = 0.2870 | k = 240'] = '#218221'
    # exocolormap['l = 0.1435 | k = 120'] = '#0b6f0b'
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # set the labels and title
    ax1.set_title('Exotendon Tension', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel('Tension (N)', fontsize=16, fontweight='bold')
    # ax.legend()
    
    # Hide the last subplot and use it to display the legend   
    ax2.axis('off')

    # Function to group legend labels
    def group_labels(labels, group_size):
        grouped_labels = []
        for i in range(0, len(labels), group_size):
            grouped_labels.append(' | '.join(labels[i:i + group_size]))
        return grouped_labels

    # get the legend labels from the previous subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    grouped_labels1 = group_labels(labels1, 1)  # Adjust the group size as needed

    # display the legend entries in the last subplot 
    ax2.legend(handles1, labels1, loc='center left', fontsize=14, handlelength=4, ncol=2)#, bbox_to_anchor=(1.05, 0.5))

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_exotendonTensions_40.png')
    plt.show()

    # create a figure for all of the exo forces. 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), dpi=500)
    # loop through each of the files
    for f in exotendon_files: 
        # print(f)
        if '3exoActMet' not in f and '4exoActMet' not in f:
            # get the parameters of the exotendon based on the file name
            exotendon_params = f.split('_')
            print(exotendon_params)
            # get the stiffness of the exotendon
            if 'compliant' in exotendon_params:
                exostiff = 'k = 60'
            elif 'stiff' in exotendon_params:
                exostiff = 'k = 180'
            elif 'stiff2' in exotendon_params:
                exostiff = 'k = 240'
            elif 'compliant2' in exotendon_params:
                exostiff = 'k = 30'
            else:
                exostiff = 'k = 120'
            # get the length of the exotendon
            if 'long' in exotendon_params:
                exolength = 'l = 0.4305'
            elif 'short' in exotendon_params:
                exolength = 'l = 0.1435'
            elif 'long2' in exotendon_params:
                exolength = 'l = 0.5741'
            elif 'short2' in exotendon_params:
                exolength = 'l = 0.0718'
            else:
                exolength = 'l = 0.2870'
            
            # print(exostiff)
            # print(exolength)
            # load in the exotendon data
            exodata = osim.TimeSeriesTable(os.path.join(resultsdir, f))
            # print(os.path.join(resultsdir,f))
            exotime = exodata.getIndependentColumn()
            tension = exodata.getDependentColumn('/forceset/HOBL|tension').to_numpy()
            # interpolate and resample 
            newtime = np.linspace(exotime[0], exotime[-1], 1000)
            tension_interp = np.interp(newtime, exotime, tension)

            # add the max tension to the max tension dict
            maxtension[exolength+' | '+exostiff] = np.max(tension_interp)

            # plot the data
            if exostiff == 'k = 120' and exolength == 'l = 0.2870':
                if not wantdots:
                    ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                else:
                    ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff])
                print('k=120 | l=0.29')
                print(np.max(tension_interp))
                datacompare[exolength+' | '+exostiff] = tension_interp
            # also only grab the top five
            elif exostiff == 'k = 240' and exolength == 'l = 0.4305':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff], alpha=0.9)
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp) #styles[1]
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=240 | l=0.43')
                print(np.max(tension_interp))
                datacompare[exolength+' | '+exostiff] = tension_interp
            elif exostiff == 'k = 30' and exolength == 'l = 0.4305':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff], alpha=0.9)#exocolormap[exolength+' | '+exostiff])
                # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=30 | l=0.43')
                print(np.max(tension_interp))
                datacompare[exolength+' | '+exostiff] = tension_interp
            elif exostiff == 'k = 240' and exolength == 'l = 0.1435':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color=exocolormap[exolength+' | '+exostiff], alpha=0.9)#exocolormap[exolength+' | '+exostiff])
                # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                selectedtension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=240 | l=0.14')
                print(np.max(tension_interp))
                datacompare[exolength+' | '+exostiff] = tension_interp
            # elif exostiff == 'k = 30' and exolength == 'l = 0.2870':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='lightcoral')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=30 | l=0.29')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 60' and exolength == 'l = 0.4305':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='red')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=60 | l=0.43')
            #     print(np.max(tension_interp))
            # elif exostiff == 'k = 60' and exolength == 'l = 0.5741':
            #     ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkred')
            #     # top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
            #     print('k=60 | l=0.57')
            #     print(np.max(tension_interp))
            elif exostiff == 'k = 180' and exolength == 'l = 0.2870':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='red')
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=180 | l=0.29')
                print(np.max(tension_interp))
            elif exostiff == 'k = 180' and exolength == 'l = 0.4305':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkred')
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=180 | l=0.43')
                print(np.max(tension_interp))
            elif exostiff == 'k = 240' and exolength == 'l = 0.2870':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='darkorange')
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=240 | l=0.29')
                print(np.max(tension_interp))
            elif exostiff == 'k = 120' and exolength == 'l = 0.1435':
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), linewidth=3, linestyle=':', color='lightcoral')
                top5tension[exolength+' | '+exostiff] = np.max(tension_interp)
                print('k=120 | l=0.14')
                print(np.max(tension_interp))
            else:
                ax1.plot(newtime, tension_interp, label=str(exostiff + ' ' + exolength), color="white", alpha=0.0)#exocolormap[exolength+' | '+exostiff])
    # set the labels and title
    ax1.set_title('Exotendon Tension', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel('Tension (N)', fontsize=16, fontweight='bold')
    # ax.legend()
    
    # Hide the last subplot and use it to display the legend   
    ax2.axis('off')

    # Function to group legend labels
    def group_labels(labels, group_size):
        grouped_labels = []
        for i in range(0, len(labels), group_size):
            grouped_labels.append(' | '.join(labels[i:i + group_size]))
        return grouped_labels

    # get the legend labels from the previous subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    grouped_labels1 = group_labels(labels1, 1)  # Adjust the group size as needed

    # display the legend entries in the last subplot 
    ax2.legend(handles1, labels1, loc='center left', fontsize=14, handlelength=4, ncol=2)#, bbox_to_anchor=(1.05, 0.5))

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.savefig('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\figure_exotendonTensions_40_single.png')
    plt.show()



    # find the average and standard deviation of the max tension values
    maxtensionvalues = np.array(list(maxtension.values()))
    
    # have to hand pick the top metabolic savings ones and then average them... 
    # exocolormap['l = 0.0718 | k = 120'] = '#2c8c2c'
    # exocolormap['l = 0.5741 | k = 240'] = '#218221'
    # exocolormap['l = 0.4305 | k = 240'] = '#167916'
    # exocolormap['l = 0.4305 | k = 180'] = '#0b6f0b'
    # exocolormap['l = 0.1435 | k = 120'] = '#006600'
    selectedTensionValues = np.array(list(selectedtension.values()))
    print('Selected tension values')
    print(selectedTensionValues)
    
    print('Max tension values')
    print(maxtensionvalues)
    print('Average tension')
    print(np.mean(maxtensionvalues))
    print('Standard deviation')
    print(np.std(maxtensionvalues))
    print('##################')
    # do stats on the top 5 
    top5tensionvalues = np.array(list(top5tension.values()))
    print('Top 5 tension values')
    print(top5tensionvalues)
    print('Average tension')
    print(np.mean(top5tensionvalues))
    print('Standard deviation')
    print(np.std(top5tensionvalues))

    # save the data compare dict to load other scripts. 
    import pickle
    with open('G:\\Shared drives\\Exotendon\\predictiveSim\\analysis\\validationfigures\\exotendonTensionDataCompare.pkl', 'wb') as f:
        pickle.dump(datacompare, f)


    return

# experimental data plotting of the exotendon tension 
def experimentalExotendonTensionPlotting():
    print('idk might do a jupyter to make it simple and fast')
#### cb# #al;sdf;lkj asdf

# end of functions