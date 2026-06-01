# PredictRunning_2d
Repository for code and data used to create predictive simulations of running with a focus on exotendon running across two speeds. 

### Language / API
This project was completed most entirely using python. Historically, there were some MATLAB as well, though much of that code has been removed (functionality here is not guarenteed). The python files in the repository were written in python 3.10.14. The OpenSim version built for this project was 4.5.1 [commit: cf3ef35](https://github.com/opensim-org/opensim-core/tree/cf3ef35d2e5e248276d535b71e5c6d9b1d880570).

# General
In this project we perform muscle-driven kinematic/ground reaction force tracking simulations. The simulations are performed with the use of the OpenSim API and OpenSim Moco (opensource direct collocation framework for biomechanics simulations). The simulations in this repository attempt to predict what running with an exotendon (assistive device) would be like for individuals. We perform simulations of running at various speeds (2.7, 4 m/s), as well as with various device parameters (stiffness and slack lengths). The largest contributors to the optimization cost function are muscle activations squared and metabolic cost. By only lightly tracking reference data, we allow the simulations to predict changes when we change the device, or duration of the gait cycle. These simulations are then evaluated based on rate of energy expenditure, activation patterns, gait patterns, etc. The purpose of this framework was to make predictions of how users would benefit from the exotendon at a new (not yet tested) running speed, from which we would select a subset of designs to evaluate in an experiment.  

# Data
- We utilize 3D motion capture data and full ground reaction force data in order to set the simulations up.
- For the slower speeds, we used data from a previous Exotendon study that can be found in the SimTK project: https://simtk.org/projects/exotendon_sims (we primarily used data from the 'welk005' subject) (Stingel et al., 2023).
- The higher speed running did not have any prior exotendon data. Therefore, we paired the model (above) with running from other subjects found in the following study: https://simtk.org/projects/nmbl_running/ (we primarily used data from Subject19) (Hamner & Delp, 2013). 
- A condensed, streamlined data repository specifically for this project can be found on SimTK here: (TODO)

# Running the code/repository
## Setup data
1. Be sure to have an up-to-date version of OpenSim installed, as well as a python environment set up (See: https://simtk-confluence.stanford.edu:8443/display/OpenSim/OpenSim+Documentation for general OpenSim installation, or find 4.X API at https://github.com/opensim-org/opensim-core/). The OpenSim version built for this project was 4.5.1 [commit: cf3ef35](https://github.com/opensim-org/opensim-core/tree/cf3ef35d2e5e248276d535b71e5c6d9b1d880570), though general functionality should work with any version 4.5 or newer. 
2. Fork and download/clone the repository to wherever you will be executing the code. 
3. You can download the needed data from the corresponding SimTK project here: (TBD1). This data will include any mocap, GRF, or other data that was required in scaling the initial model as well as those used in the tracking problem. This data folder should be placed within the main repository folder, as scripts were coded to reference this data (and may need slight tweaks based on the location that you have stored the repository/data).
4. There are several OpenSim models included in the repository. They are all scaled to the same subject. These different model files were generated with each iteration of the exotendon device implemented. Generally, each of the models should be called/loaded appropriately in the code, so that you don't have to keep track of each one.

## Running the code for 4 m/s simulations
- Since this is a predictive simulation project, we utilized a musculoskeletal model that was representative of the runners we'd previously collected data with. The inverese kinematics data is used as the tracking reference for each of the simulations, as well as the GRF data. We used a single gait cycle solution in order to provide an initial guess to each of the subsequent simulations. In order to generate it, the following scripts can be used:
- *** The following scripts can then be used in order to generate simulations similar to those in the results of the project. These simulations utilize inverse kineamtics and GRFs files for tracking in the cost function, activations squared as an effort term, a smoothed metabolics term (Bhargava, 2004; Falisse, 2019), as well as a term to keep implicit auxillary derivatives relatively small. In this project, we simulated different gait cycle durations. Within each of the simulations scripts the final time can be changed in order to implement a different duration for the gait cycle. Because there are symmetric constraints on the solution, simply changing the final timepoint, along with an endpoint speed constraint, allows us to change the gait cycle length (in time and distance).
    1. py_overground_4ActMet_CMA.py : natural running simulations
    2. py_overground_4exoActMet_CMA.py : exotendon running simulations (exotendon parameters from measured device)
    3. py_overground_4exoActMet_CMA_stiff.py : exotendon running simulations (simulation experiments with stiffer exotendon)
    4. py_overground_4exoActMet_CMA_compliant.py : exotendon running simulations (simulation experiments with a more compliant exotendon)
    5. py_overground_5exoActMet_zero.py : exotendon running simulations (simulation validation experiments where the exotendon stiffness was set to 1 N/m, meant to recreate natural outcomes)
    6. ... similar scripts for each iteration of exotendon design.
- Each of these scripts includes an initial guess. In order to generate the initial guess solutions, we used a combination of 1) prior simulation results of tight tracking inverse simulations (Stingel et al. 2023), as well as 2) loose convergence tolerances in order to get a solution. These were refined over iterations of simulations. IG files are included in the simTK repository. 

## Running the code for 2.7 m/s simulations. 
1. The structure for this speed is the same as the 4 m/s simulations, where we have scripts that will perform a simpler simulation that can then be used as the initial guess for the more detailed simulations that are used in analysis for this project.

## Other scripts/simulations.
- There are a number of other scripts that were included in the repository that were used at various points in the development of these simulations. For example, py_bilevel_CMA_parallel.py and bilevelTools.py builds and runs a bilevel optimization where the inner loop is the same predictive simulations as above, but the outer loop utilizes CMA-ES as a means to tune the weights in the lower loop cost function. This was done early in development in order to determine relative weights for the objectives in predictive simulations.

## Plotting and results.
- exotendon_simulation_results.ipynb is the main python notebook that was used in order to visualize and explore the results of the simulations. The notebook pairs with helperOsimFunctions.py for much of the plotting functionality.
- experimentalExotendonTensionPlotting.ipynb was used to visualize some of the experimental validation data, specifically, the estimated tension in the exotendons that was seen in the experiments in comparison to simulations.
- simulatedMetabolicsPlotting.ipynb was used to visualize the changes in metabolic cost across exotendon designs for the various stride durations simulated.
