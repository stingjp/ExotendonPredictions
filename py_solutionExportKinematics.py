import opensim as osim
import pdb


solution = osim.MocoTrajectory('./goodresults/27ms_mk12/015_3ActMet_6835_tight_clever/3ActMet_2D3D_OG_muscles_Tracking_solution_FullStride.sto')

statesTable = solution.exportToStatesTable()
statesNames = statesTable.getColumnLabels()

# loop and get rid of anything without kinematics values. 
for each in statesNames:
    if 'value' not in each:
        statesTable.removeColumn(each)
    # if 'jointset' not in each:
        # statesTable.removeColumn(each)


newStatesNames = statesTable.getColumnLabels()
# print(newStatesNames)
# pdb.set_trace()

osim.STOFileAdapter.write(statesTable, './goodresults/27ms_mk12/015_3ActMet_6835_tight_clever/3ActMet_kinematicsValues_solution.sto')