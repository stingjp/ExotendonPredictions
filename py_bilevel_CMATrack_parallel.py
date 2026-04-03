import os
# os.add_dll_directory("C:/OpenSim 4.4/bin")
# os.add_dll_directory("C:/Users/jonstingel/opensim4.4update/opensim-core-install/bin")
# os.add_dll_directory("C:/Users/jonstingel/mocobuilds/opensim-core-install/bin")
# os.add_dll_directory('C:/Users/jonstingel/opensim/opensim-core-4.5-2024-05-15-a1a2282/bin')
os.add_dll_directory('C:/Users/jonstingel/opensim-core-4.5.1-2024-08-23-cf3ef35/bin')



import opensim as osim
import numpy as np
import pdb
import helperOsimFunctions
import time
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.optimize import minimize
# from multiprocessing import Pool
from multiprocessing import Pool
import cma
import bilevelTools
import time
from datetime import date

def cmaesAlgorithmTrack(num_workers, outlogfile):
    # setup the algorithm
    # define the cma parameters
    sigma0 = [1, 1, 1, 1] #, 0.8]
    # sigma0 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 
    fillsigma = 1

    # 5.1739458   1.35650315  5.45546111 15.48640291  5.47417995
    # 5.35730665  4.80445649 7.3886564  8.34994985 6.30152856
    stateIG = 5e-5 # -4.5 # 0.006
    # activIG = 8 # 0.9 # 53.6 
    # metabolicIG = 3e-4 # -3.2 # 0.000149  
    contactIG = 7 # 0.06 # 0.054
    # kneeIG = # 4.0
    # knee2IG = # 2.7
    headIG = 1e2
    tyIG = 1e2

    # pelvIG = 20
    # stiffIG = 8.39382533 # 7.302499244712357 # 6.30152856

    # x0 = [stateIG, metabolicIG, contactIG, kneeIG, knee2IG] # stateIG, contactIG, headIG, pelvIG] #,] metabolicIG, activIG
    x0 = [stateIG,contactIG,headIG,tyIG] # stateIG, contactIG, headIG, pelvIG] #,] metabolicIG, activIG
    
    # x0 = np.ones((14,))
    # x0 = [16.4642132, 26.17749431, 2.40395633, 1.40346929, 5.81808907, 9.45995736, 9.9811225, 19.10767123, 0.17034188, 6.38006386, 7.76837126, 4.61191839, 8.47651309, 2.5277297]

    # set bounds for the variables
    statelb = 0
    stateub = 1e-2
    activationlb = 0
    activationub = 100
    metaboliclb = 0
    metabolicub = 0.03
    contactlb = 0
    contactub = 100
    headlb = 0
    headub = 1e6
    tylb = 0
    tyub = 1e6
    # headacclb = -2
    # headaccub = 30
    # pelvacclb = -2
    # pelvaccub = 30
    # flipped for stiffness
    # stifflb = 2
    # stiffub = 15
    kneelb = -0.5
    kneeub = 6.5

    # lb = np.array([statelb, metaboliclb, contactlb, kneelb, kneelb]) #,]) metaboliclb, , activationlb
    # ub = np.array([stateub, metabolicub, contactub, kneeub, kneeub]) #stateub, contactub, headaccub , pelvaccub]) #,]) metabolicub, , activationub
    lb = np.array([statelb, contactlb, headlb, tylb]) #,]) metaboliclb, , activationlb
    ub = np.array([stateub, contactub, headub, tyub])# lb = np.zeros((14,))
    # ub = 500 * np.ones((14,))
    bounds = [lb, ub]


    # pdb.set_trace()
    # todo figure out scaling of the variables and of their bounds/stddev
    opts = cma.CMAOptions()
    opts.set('popsize', 16) # 40
    opts.set('bounds', bounds)
    opts.set('maxiter', 50)
    opts.set('verb_disp', 1)
    opts.set('CMA_stds', sigma0)


    # create a pool of worker processses for parallel fitness evaluation
    pool = Pool(num_workers)

    # create the cmaes optimizer
    optimizer = cma.CMAEvolutionStrategy(x0, fillsigma, opts)

    # run the optimization loop
    while not optimizer.stop():
        # generate a new population of candidates
        solutions = optimizer.ask()
        outlog = open(outlogfile, 'a')
        outlog.write('\n\nPopulation:')
        outlog.write(str(solutions))
        outlog.close()


        print('\n\ntesting:')
        print(solutions)
        tic1 = time.time()
        #evaluate the fitness of each in parallel
        # fitness_scores = pool.map(bilevelTools.objective_bilevel_CMATrack, solutions)

        # swing for the fences and optimize both at the same time. 
        fitness_scores = pool.map(bilevelTools.objective_bilevel_CMATrackTight, solutions)

        # this version only optimizes on the natural simulations...
        # fitness_scores = pool.map(bilevelTools.objective_bilevel_CMATrackTightnat, solutions)
                

        # update the cmaes optimizer with the fitness scores
        optimizer.tell(solutions, fitness_scores)
        iterations = optimizer.countiter
        iterfile = open("iteration_.txt", 'w')
        iterfile.write(str(iterations+1))
        iterfile.close()

        tic2 = time.time()

        # output and print some of the stats of each of the iterations and their performance
        outlog = open(outlogfile, 'a')
        outlog.write('\n\n#################################\n\n')
        print('\n\n#############################################################\n\n')
        outlog.write('\n\nIteration: %f' % float(iterations))
        outlog.write(f"Iteration: {iterations}\n")
        outlog.write(f"Stopping criteria: {optimizer.stop()}\n")        
        print('\niterations')
        print(iterations)
        print(f"Iteration: {iterations}")
        print(f"Stopping criteria: {optimizer.stop()}")

        outlog.write('\nTime in interation [min]: %f' % ((tic2-tic1)/60))
        print('\ntime it took for this generation (min):')
        print((tic2-tic1)/60)

        outlog.write('\n\nStdev:\n')
        outlog.write(str(optimizer.result[6]))

        outlog.write('\n\nPotential solutions:\n')
        outlog.write(str(solutions))
        print('\nsolutions')
        print(solutions)
        
        outlog.write('\n\nFitness scores:\n')
        outlog.write(str(fitness_scores))
        print('\nfitness_scores')
        print(fitness_scores)
        
        outlog.write('\n\n#############################################################\n\n')
        outlog.close()
        print('\n\n')
        print('\n\n#############################################################\n\n')


    # return the best individual and its fitness score
    # best_index = np.argmin(fitness_scores)
        # pdb.set_trace()

    return optimizer.result.xbest, optimizer.result.fbest, optimizer.result[6]



if __name__ == '__main__':
    tic = time.time()
    today = date.today()
    # create log of bilevel performance
    outlogfile = "outlog_bilevelCMATRack_"+str(today)+'T'+str(tic)+'.txt' 
    # check if the file exists - shouldn't ever happen since using time in title
    if not os.path.isfile(outlogfile):
        # create file for this call
        outlog = open(outlogfile, 'x')

    iterfile = open("iteration_.txt", 'w')#+str(today)+'T'+str(tic)+'.txt'
    iterfile.write(str(0))
    iterfile.close()

    CMAlog = open('CMATrack_logfile.txt', 'w')
    CMAlog.write(outlogfile)
    CMAlog.close()


    # open up the file to modify it
    outlog = open(outlogfile, 'a')
    outlog.write('\nBilevel Optimization CMA performance.')
    outlog.close()


    # call
    best_individual, best_fitness, newopts = cmaesAlgorithmTrack(num_workers=8, outlogfile=outlogfile)
    toc = time.time()

    # stop the timer
    print(tic)
    print(toc)

    # run the best one again just to have it be a saved result
    finalbest = bilevelTools.objective_bilevel_CMATrack(best_individual)
    # finalbest = bilevelTools.objective_bilevel_CMATrackTightnat(best_individual)
    
    # output the best solutions to the file
    outlog = open(outlogfile, 'a')
    outlog.write('\n\n#############################################################')
    outlog.write('\n#############################################################\n\n')
    outlog.write('\nFinal Result:')
    outlog.write('\nTotal time [hours]: %f' % ((toc-tic)/3600))
    outlog.write('\n\nBest Individual:\n')
    outlog.write(str(best_individual))
    outlog.write('\nBest Fitness:\n')
    outlog.write(str(best_fitness))
    outlog.write('\nBest outer cost: %f' % finalbest)
    outlog.write('\nStd dev:\n')
    outlog.write(str(newopts))
    outlog.close()

    

    # print to command line the results
    print('\n\ndone')
    print('best individual')
    print(best_individual)
    print('\nbest fitness')
    print(best_fitness)
    print('\nbest outer cost')
    print(finalbest)
    print('\nstd dev.')
    print(newopts)



    # synchronize the data to G
    destDir = 'G:\\Shared drives\\Exotendon\\predictiveSim\\testingMaterials\\PredictRunning_2d';
    localDir = 'C:\\Users\\jonstingel\\code\\predictiveSim\\testingMaterials\\PredictRunning_2d';
    helperOsimFunctions.syncDrives(localDir, destDir)


    print('\nTotal time [hours]')
    print((toc-tic)/3600)
    pdb.set_trace()

    

