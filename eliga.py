# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:26:02 2015

@author: mgerard
"""

##===========================================


def ELIGA(settings_file):
    
    #----------------------------
    # IMPORTING MODULES
    import numpy as np
    
    import matplotlib.pyplot as plt
    
    import json, csv
    
    import multiprocessing
    
    import timeit
    
    import libELIGA

    from sklearn import preprocessing

    #    import matplotlib.pyplot as plt
    
    #----------------------------
    # INITIALIZING
    tic = timeit.default_timer
    start_initialization = tic()
    
    
    #----------------------------
    # LOAD SETTINGS
    print("\nLoading SETTINGS...")
    settings = libELIGA.Settings(settings_file)
    
    #-------------------------------------------
    # BUILD STRUCTURE TO SAVE EXPERIMENT DATA
    #-------------------------------------------
    #HISTORY = libELIGA.History(Gmax=int(settings['maxIterations']))
    
    #==========================
    # MEASURES INITIALIZATION
    #==========================
    MEASURES = libELIGA.Measures()
    MEASURES.add('features')
    MEASURES.add('uar')
    MEASURES.add('fitness')
    
    
    #==========================
    # STARTING PARALLEL SERVER
    #==========================
#    ppservers = ()
#    jobs_server = pp.Server(ncpus=ncpus, ppservers=ppservers)
    if settings['NCORES'] == 1:
        jobs_server = ''
    else:
        jobs_server = multiprocessing.Pool(settings['NCORES']) # Multiprocessing or PATHOS


    time_settings = tic()
    loadig_parameters = time_settings - start_initialization
    print("Elapsed time: " + str(loadig_parameters) + " seconds\n")
    
    
    #=====================
    # LOAD TRAINING DATA
    #=====================
    print("Loading TRAINING DATA...")
    name,ext = settings['TrainData'].split('/')[-1].split('.')
    
    X_train = []
    Y_train = []
    with open(settings['TrainData'], 'r') as fp:
        for row in csv.reader(fp):
            
            X_train.append(row[:-1])
            Y_train.append(row[-1])
    
    Nfeatures = len(X_train[0])
    
    X_train = np.array([X_train],dtype=np.float64)
    Y_train = np.array([Y_train],dtype=np.float64)


    time_train = tic()
    loading_train = time_train - time_settings
    print("Elapsed time: " + str(loading_train) + " seconds\n")
    

    #=====================
    # LOAD TEST DATA
    #=====================
    print("Loading TEST DATA...")
    name,ext = settings['TestData'].split('/')[-1].split('.')
    
    X_test = []
    Y_test = []
    with open(settings['TestData'], 'r') as fp:
        for row in csv.reader(fp):
            
            X_test.append(row[:-1])
            Y_test.append(row[-1])
    
    X_test = np.array([X_test],dtype=np.float64)
    Y_test = np.array([Y_test],dtype=np.float64)
    
    time_test = tic()
    loading_test = time_test - time_train
    print("Elapsed time: " + str(loading_test) + " seconds\n")
    
    
    
    #=====================
    # LOAD VALIDATION DATA
    #=====================
    print("Loading VALIDATION DATA...")
    
    name,ext = settings['ValidationData'].split('/')[-1].split('.')
    
    X_validation = []
    Y_validation = []
    with open(settings['ValidationData'], 'r') as fp:
        for row in csv.reader(fp):
            
            X_validation.append(row[:-1])
            Y_validation.append(row[-1])
    
    X_validation = np.array([X_validation],dtype=np.float64)
    Y_validation = np.array([Y_validation],dtype=np.float64)
    
    time_test = tic()
    loading_test = time_test - time_train
    print("Elapsed time: " + str(loading_test) + " seconds\n")

    #=====================
    # STANDARIZING DATA
    #=====================

    # conviene escalar los datos completos una unica vez previo a la evolucion \
    # de manera evitar este calculo en cada evaluacion de fitness
    for j in range(X_train.shape[0]):
        scaler = preprocessing.StandardScaler().fit(X_train[j])
        X_train[j] = scaler.transform(X_train[j])
        X_test[j] = scaler.transform(X_test[j])
        X_validation[j] = scaler.transform(X_validation[j])

    
    #=====================
    # BUILDING POPULATIONS
    #=====================
    print('Construyendo población...\n')
    population = libELIGA.Population(Nindiv=settings['Nindiv'],
                                     alpha=float(settings['alpha']),
                                     Nfeatures=Nfeatures,
                                     px=settings['px'],
                                     pm=settings['pm'],
                                     pF=settings['pF'],
                                     gamma_initial=settings['gamma_initial'],
                                     gamma_final=settings['gamma_final']
                                    )
    
    
    if settings['Nsubpop'] > 0:
        subpopulation = population.clone()
        subpopulation.clear()



    
    
    #---------------------------------------------------
    # INITIALIZING ANTHILL - N CPUs ---> MULTIPROCESSING
    print('Inicializando...\n')
    #population.initialize(method='random')
    population.initialize(method='relief', features=X_train, labels=Y_train)
    #---------------------------------------------------
    
    
    
    #---------------------------------------------------
    # SAVE INITIALIZATION TIME
    #---------------------------------------------------
    end_initialization = tic()
    MEASURES.time['initialization'] = end_initialization - start_initialization
    
    
    #---------------------------------------------------
    # EVALUATE POPULATION ---> PODRIA PARALELIZARSE??
    print('Evaluando población...\n')
    fitness = population.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=True)
    
    
    MEASURES.update(key='features', values=fitness['features'], idx_elite=population.elite_idx)
    MEASURES.update(key='uar', values=fitness['uar'], idx_elite=population.elite_idx)
    MEASURES.update(key='fitness', values=fitness['total'], idx_elite=population.elite_idx)
        
    #---------------------------------------------------
    
    
    start_search = tic()
    
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = =
    # PLOTS
    #= = = = = = = = = = = = = = = = = = = = = = = = = = =
    if settings['Plot_measures']:
        plt.ion()
        fig = plt.figure(1,figsize=(17, 9))
        p11 = fig.add_subplot(2,3,1)
        p12 = fig.add_subplot(2,3,2)
        p13 = fig.add_subplot(2,3,3)
        p21 = fig.add_subplot(2,3,4)
        p22 = fig.add_subplot(2,3,5)
        p23 = fig.add_subplot(2,3,6)
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    
    fitness = population.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=True)
    
    MEASURES.G = 1
    
    counter = 0
    bestfitness = 0.0
    
    while (float(1.0) - population.elite_fitness > 1E-5) and \
          (MEASURES.G < settings['maxIterations']) and \
          (counter < float(settings['maxIterations'])/1):
        
        if settings['verbose']:
            print("Generación: " + str(MEASURES.G))
            print("Contador: " + str(counter))
        
        
        
        
        #===============================
        # LOCAL IMPROVING STEP
        #===============================
        if (settings['Nsubpop'] > 0) and (MEASURES.G % settings['Subpop_Gwait'] == 0):
            
            SUBPOPULATIONS = []
            
            ELITES, idxs = population.get_elite(N=settings['Nsubpop'])

            for N,ELITE in enumerate(ELITES):
                
                SUBPOP_MEASURES = libELIGA.Measures()
                SUBPOP_MEASURES.add('features')
                SUBPOP_MEASURES.add('uar')
                SUBPOP_MEASURES.add('fitness')
                
                
                # BUILD POPULATION
                subpopulation = libELIGA.Population(Nindiv=settings['Subpop_Nindiv'],
                                                    px=settings['px'],
                                                    alpha=settings['alpha'],
                                                    Nfeatures=ELITE.get_selected_features(),
                                                    pm=settings['pm'],
                                                    pF=settings['pF'],
                                                    gamma_initial=settings['gamma_initial'],
                                                    gamma_final=settings['gamma_final'])
                
                
                # INITIALIZE SUBPOPULATION
                subpopulation.initialize_SP(method='random')
                subpopulation[0].chromosome = [1] * len(subpopulation[0].chromosome)
                
                # EVALUATE SUBPOPULATION
                fitness = subpopulation.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=True)
                
                
                SUBPOP_MEASURES.G = 1
                subpop_counter = 0
                subpop_bestfitness = 0.0
                
                while (float(1.0) - population.elite_fitness > 1E-5) \
                and (SUBPOP_MEASURES.G < settings['Subpop_Gmax']) \
                and (subpop_counter < float(settings['Subpop_Gmax'])/1):
                    
                    print('Subpopulation %d - Generation: %d' % (N+1,SUBPOP_MEASURES.G) )
                    
                    #---------------------------------------------------
                    # EXTRACT ELITE INDIVIDUAL
                    #---------------------------------------------------
                    subpop_elite, null = subpopulation.get_elite(N=1)
                    subpop_elite = subpop_elite[0]

                    if subpop_elite.fitness['total'] == subpop_bestfitness:
                        subpop_counter += 1
                    else:
                        subpop_bestfitness = subpop_elite.fitness['total']
                        subpop_counter = 0
                    #---------------------------------------------------
                    
                    
                    subpop_gap, subpop_offspring = subpopulation.build_offspring(settings['Subpop_Nchild'],
                                                                                 settings['Subpop_Ngap'],
                                                                                 G=SUBPOP_MEASURES.G,
                                                                                 Gmax=settings['Subpop_Gmax'])
                    
                    
                    #---------------------------------------------------
                    # BUILD NEW POPULATION
                    #---------------------------------------------------
                    subpopulation.clear()
                    subpopulation.elite_fitness = 0.0
                    
                    subpopulation.append(subpop_elite.clone())
                    
                    # COMENTAR PARA ELIMINAR BRECHA GENERACIONAL
                    subpopulation.extend(subpop_gap[:])
                    #population.extend(new_parents[:])
                    
                    subpopulation.extend(subpop_offspring[:])
                    
                    
                    #---------------------------------------------------
                    # EVALUATE NEW POPULATION
                    #---------------------------------------------------
                    if settings['verbose']:
                        subpop_fitness = subpopulation.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=True)
                    else:
                        subpop_fitness = subpopulation.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=False)
                    
                    
                    SUBPOP_MEASURES.update(key='features',
                                           values=subpop_fitness['features'],
                                           idx_elite=subpopulation.elite_idx)
                    
                    SUBPOP_MEASURES.update(key='uar',
                                           values=subpop_fitness['uar'],
                                           idx_elite=subpopulation.elite_idx)
                    
                    SUBPOP_MEASURES.update(key='fitness',
                                           values=subpop_fitness['total'],
                                           idx_elite=subpopulation.elite_idx)
                    
                    
                    
                    
                    
                    #=======================
                    # UPDATE PLOT
                    #=======================
                    if settings['Plot_measures']:
                        
                        ################
                        # CURRENT
                        ##########
                        
                        #==================================================================
                        # FITNESS
                        #==========            
                        p21.cla()
                        
                        # HISTOGRAM
                        n = len(SUBPOP_MEASURES.history['fitness']['mean'])
                        delta = n/settings['Subpop_Nindiv']
                        x = np.arange(0,settings['Subpop_Nindiv']) * delta
                        p21.bar(x, SUBPOP_MEASURES.current['fitness'], width=0.5*delta, color='magenta', alpha=0.50)
                        
                        # PLOTS
                        p21.plot(SUBPOP_MEASURES.history['fitness']['mean'],'r-')
                        p21.plot(SUBPOP_MEASURES.elite['fitness'],'b-')
                        p21.set_xlabel('Generations', fontsize=14)
                        p21.set_ylabel('Fitness', fontsize=14)
                        p21.set_ylim([0, 1])
                        p21.legend(['mean','elite'],loc=4)
                        
                        p21.grid(True)
                        
                        
                        
                        
                        #==================================================================
                        
                        #==================================================================
                        # FEATURES
                        #==========
                        p22.cla()
                        
                        # HISTOGRAM
                        n = len(SUBPOP_MEASURES.history['features']['mean'])
                        delta = n/settings['Subpop_Nindiv']
                        x = np.arange(0,settings['Subpop_Nindiv']) * delta
                        p22.bar(x, SUBPOP_MEASURES.current['features'], width=0.5*delta, color='cyan', alpha=0.50)
                        
                        # PLOTS
                        p22.plot(SUBPOP_MEASURES.history['features']['mean'],'r-')
                        p22.plot(SUBPOP_MEASURES.elite['features'],'b-')
                        p22.set_xlabel('Generations', fontsize=14)
                        p22.set_ylabel('Features', fontsize=14)
                        p22.legend(['mean','elite'],loc=4)
                        
                        p22.grid(True)
                        #==================================================================
                        
                        #==================================================================
                        # UAR
                        #==========
                        p23.cla()
                        
                        # HISTOGRAM
                        n = len(SUBPOP_MEASURES.history['uar']['mean'])
                        delta = n/settings['Subpop_Nindiv']
                        x = np.arange(0,settings['Subpop_Nindiv']) * delta
                        p23.bar(x, SUBPOP_MEASURES.current['uar'], width=0.5*delta, color='yellow', alpha=0.50)
                        
                        # PLOTS
                        p23.plot(SUBPOP_MEASURES.history['uar']['mean'],'r-')
                        p23.plot(SUBPOP_MEASURES.elite['uar'],'b-')
                        p23.set_xlabel('Generations', fontsize=14)
                        p23.set_ylabel('UAR', fontsize=14)
                        p23.set_ylim([0, 1])
                        p23.legend(['mean','elite'], loc=4)
                        
                        p23.grid(True)
                        #==================================================================
                        
                        fig.tight_layout()
                        plt.pause(0.01)
                    
                    

                    
                    #---------------------------------------------------
                    # UPDATE GENERATION COUNTER
                    #---------------------------------------------------
                    SUBPOP_MEASURES.G += 1
                    
                
                #--------------------------------------------------------------
                if settings['Replacement'] == 'Reemplazo Padre':
                    
                    subpop_elite, null = subpopulation.get_elite(N=1)
                    subpop_elite = subpop_elite[0]

                    subpop_elite.reshape_chromosome(ELITE.size())
                    
                    if ELITE.fitness['total'] <= subpop_elite.fitness['total']:
                        ELITES[N] = subpop_elite.clone()
                        N += 1

                #--------------------------------------------------------------
                elif settings['Replacement'] == 'Reemplazo Completo':
                    
                    SUBPOPULATIONS.extend([indiv.reshape_chromosome(ELITE.size()) for indiv in subpopulation])
                
                
                #--------------------------------------------------------------
                else:
                    print('Unknown replacement strategy')
            
        
        
            #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            if settings['Replacement'] == 'Reemplazo Padre':
                for i,j in enumerate(idxs):
                    population[j] = ELITES[i].clone()

            #--------------------------------------------------------------
            elif settings['Replacement'] == 'Reemplazo Completo':

                population.extend(SUBPOPULATIONS)
                population = population.sort_pop()

            else:
                pass
        
        
        
        
        
        
        
        #=======================
        # UPDATE PLOT
        #=======================
        if settings['Plot_measures']:
            
            ################
            # HISTORY
            ##########
            
            #==================================================================
            # FITNESS
            #==========            
            p11.cla()
            
            # HISTOGRAM
            n = len(MEASURES.history['fitness']['mean'])
            delta = n/settings['Nindiv']
            x = np.arange(0,settings['Nindiv']) * delta
            p11.bar(x, MEASURES.current['fitness'], width=0.5*delta, color='red', alpha=0.50)
            
            # PLOTS
            p11.plot(MEASURES.history['fitness']['mean'],'r-')
            p11.plot(MEASURES.elite['fitness'],'b-')
            p11.set_xlabel('Generations', fontsize=14)
            p11.set_ylabel('Fitness', fontsize=14)
            p11.set_ylim([0, 1])
            p11.legend(['mean','elite'],loc=4)
            
            p11.grid(True)
            
            
            
            
            #==================================================================
            
            #==================================================================
            # FEATURES
            #==========
            p12.cla()
            
            # HISTOGRAM
            n = len(MEASURES.history['features']['mean'])
            delta = n/settings['Nindiv']
            x = np.arange(0,settings['Nindiv']) * delta
            p12.bar(x, MEASURES.current['features'], width=0.5*delta, color='blue', alpha=0.50)
            
            # PLOTS
            p12.plot(MEASURES.history['features']['mean'],'r-')
            p12.plot(MEASURES.elite['features'],'b-')
            p12.set_xlabel('Generations', fontsize=14)
            p12.set_ylabel('Features', fontsize=14)
            p12.legend(['mean','elite'],loc=4)
            
            p12.grid(True)
            #==================================================================
            
            #==================================================================
            # UAR
            #==========
            p13.cla()
            
            # HISTOGRAM
            n = len(MEASURES.history['uar']['mean'])
            delta = n/settings['Nindiv']
            x = np.arange(0,settings['Nindiv']) * delta
            p13.bar(x, MEASURES.current['uar'], width=0.5*delta, color='green', alpha=0.50)
            
            # PLOTS
            p13.plot(MEASURES.history['uar']['mean'],'r-')
            p13.plot(MEASURES.elite['uar'],'b-')
            p13.set_xlabel('Generations', fontsize=14)
            p13.set_ylabel('UAR', fontsize=14)
            p13.set_ylim([0, 1])
            p13.legend(['mean','elite'], loc=4)
            
            p13.grid(True)
            #==================================================================
            
            
            ################
            # CURRENT
            ##########
            
            #==================================================================
            # FITNESS
            #==========
            p21.cla()
#            p21.bar(range(settings['Nindiv']), MEASURES.current['fitness'], width=0.8, color='magenta', alpha=0.75)
#            p21.set_xlabel('Individuals', fontsize=14)
#            p21.set_ylabel('Fitness', fontsize=14)
#            p21.set_xlim([-0.5, settings['Nindiv']-0.5])
            p21.set_ylim([0, 1.0])
#            p21.legend(['mean','elite'])
            p21.grid(True)
            #==================================================================
            
            #==================================================================
            # FEATURES
            #==========
            p22.cla()
#            p22.bar(range(settings['Nindiv']), MEASURES.current['features'], width=0.8, color='cyan', alpha=0.75)
#            p22.set_xlabel('Individuals', fontsize=14)
#            p22.set_ylabel('Fitness', fontsize=14)
#            p22.set_xlim([-0.5, settings['Nindiv']-0.5])
#            p22.set_ylim([0, 1.0])
            p22.grid(True)
            #==================================================================
            
            #==================================================================
            # UAR
            #==========
            p23.cla()
#            p23.bar(range(settings['Nindiv']), MEASURES.current['uar'], width=0.8, color='yellow', alpha=0.75)
#            p23.set_xlabel('Individuals', fontsize=14)
#            p23.set_ylabel('UAR', fontsize=14)
#            p23.set_xlim([-0.5, settings['Nindiv']-0.5])
#            p23.set_ylim([0, 1.0])
            p23.grid(True)
            #==================================================================
            
            fig.tight_layout()
            plt.pause(0.01)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #---------------------------------------------------
        # EXTRACT ELITE INDIVIDUAL
        #---------------------------------------------------
#        print 'Extrayendo individuo elite...'
        ELITES, null = population.get_elite(N=1)
        elite = ELITES[0]
        
        if elite.fitness['total'] == bestfitness:
            counter += 1
        else:
            bestfitness = elite.fitness['total']
            counter = 0
        #---------------------------------------------------
        
        
        #---------------------------------------------------
        # BUILDING GENERATIONAL GAP
        #---------------------------------------------------
#        print 'Seleccionando padres...'
        #gap = population.tournament_selection(settings['Ngap'], K=settings['K'])
        #---------------------------------------------------


        #---------------------------------------------------
        # SELECT NEW PARENTS
        #--------------------------------------------------- 
#        print 'Seleccionando padres...'
        new_parents = population.tournament_selection(settings['Np2x'], K=settings['K'])
        
#        new_parents = population.tournament_selection(settings['Nchild'], K=settings['K'])
#        new_parents.mutate(jobs_server, method='exponential', G=MEASURES.G, Gmax=settings['maxIterations'])
        #---------------------------------------------------
        
        
        #---------------------------------------------------
        # BUILD OFFSPRING
        #---------------------------------------------------
#        print 'Generando descendencia...'
        offspring = new_parents.crossover(settings['Nchild'])
        #---------------------------------------------------
        
        
        #---------------------------------------------------
        # APPLY MUTATIONS - N CPUs ---> MULTIPROCESSING
        #---------------------------------------------------
#        print 'Aplicando mutaciones...'
        offspring.mutate(jobs_server, method='constant')
        #offspring.mutate(jobs_server, method='exponential',G=MEASURES.G, Gmax=settings['maxIterations'])
        #---------------------------------------------------
        
        
        # gap,offspring = population.build_offspring(settings['Nchild'], settings['Ngap'], G=MEASURES.G, Gmax=settings['maxIterations'])
        
        gap = population.tournament_selection(settings['Ngap'], K=settings['K'])
        
        #---------------------------------------------------
        # BUILD NEW POPULATION
        #---------------------------------------------------
#        print 'Construyendo nueva población...'
        population.clear()
        population.elite_fitness = 0.0
        
        for elite in ELITES:
            population.append(elite.clone())
        
        # COMENTAR PARA ELIMINAR BRECHA GENERACIONAL
        population.extend(gap[:])
        #population.extend(new_parents[:])
        
        population.extend(offspring[:])
        
        
        #---------------------------------------------------
        # EVALUATE NEW POPULATION
        #---------------------------------------------------
        if settings['verbose']:
            fitness = population.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=True)
        else:
            fitness = population.evaluate(X_train, Y_train, X_test, Y_test, jobs_server, show=False)
        
        
        MEASURES.update(key='features', values=fitness['features'], idx_elite=population.elite_idx)
        MEASURES.update(key='uar', values=fitness['uar'], idx_elite=population.elite_idx)
        MEASURES.update(key='fitness', values=fitness['total'], idx_elite=population.elite_idx)
        
        
        #---------------------------------------------------
        # UPDATE GENERATION COUNTER
        #---------------------------------------------------
        MEASURES.G += 1
        

        
#==============================================================================
    
    MEASURES.G -= 1 # CORRECTION
    
    #------------------------
    # SAVE SEARCHING TIME
    #------------------------
    end_search = tic()
    MEASURES.time['execution'] = end_search - start_search
    print("Execution time: " + str(MEASURES.time['execution']) + " seconds\n")
    
    
    
    #--------------------
    # SAVE TOTAL TIME
    #--------------------
    MEASURES.time['total'] = end_search-start_initialization
    
    
    
    #------------------------
    # SAVING EXPERIMENT DATA
    #------------------------
    DBname = settings['TrainData'].split('/')[1].split('.')[0]
    FILENAME = 'ELIGA_indexes__' + DBname + '_' + timeit.time.strftime("__%Y%m%d-%H%M%S")
    
    
    
    # PLOT
    if settings['Plot_measures']:
        plt.savefig('plot_' + FILENAME + '.pdf')
    
    
    
    
    
    RESULTS = dict()
    
    RESULTS['SETTINGS'] = settings
    
    RESULTS['TIME'] = MEASURES.export(key='time')
    RESULTS['GENERATIONS'] = MEASURES.G
    
    #------------------------------------------
    
    RESULTS['HISTORY'] = MEASURES.export(key='history')
    
    #------------------------------------------
    
    
    #--------------------
    # VALIDATION
    #--------------------
    X_train = np.array([np.vstack([X_train[0],X_test[0]])])
    Y_train = np.hstack([Y_train,Y_test])
    
    elite, null = population.get_elite(N=1)
    elite = elite[0]
    elite.modified = True
    elite.evaluate(X_train, Y_train, X_validation, Y_validation)
    
    print('-----------------------------')
    print('MEDIDAS --- VALIDACION')
    print('-----------------------------')
    print('FEATURES:\t%d\n' % (elite.fitness['features']) )
    print('UAR:\t\t%0.4f\n' % (elite.fitness['uar']) )
    print('FITNESS:\t%0.4f\n' % (elite.fitness['total']) )
    print('\n\n')
    
    #========================================================

    RESULTS['ELITE'] = dict()
    RESULTS['ELITE']['train'] = MEASURES.export(key='elite')
    
    RESULTS['ELITE']['test'] = dict()
    RESULTS['ELITE']['test']['chromosome'] = [idx+1 for idx in range(elite.size()) if elite.chromosome[idx] == 1]
    RESULTS['ELITE']['test']['features'] = elite.chromosome.count(1)
    RESULTS['ELITE']['test']['uar'] = (elite.fitness['uar'])
    RESULTS['ELITE']['test']['fitness'] = (elite.fitness['total'])
    #------------------------------------------
    
    with open( FILENAME + '.json', 'w' ) as fp:
        json.dump(RESULTS, fp, sort_keys=True)
    #------------------------



#==============================================================================
if __name__ == "__main__":
#==============================================================================
    '''
    Example:
        
        python3 eliga.py -settings SETTINGS_eliga.yaml
    '''
    
    import sys
    
    # settings = 'SETTINGS.yaml'
    settings = 'SETTINGS_eliga_leuk.yaml'
    
    for ii in range(1,len(sys.argv),2):
        
        if sys.argv[ii] == '-settings':
            settings = sys.argv[ii+1]
        
        else:
           print('Parámetro desconocido.')

    ELIGA(settings)
