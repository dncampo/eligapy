#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:12:14 2015

@author: mgerard
"""


# LIBRARIES
import numpy as np
import copy

import json
import yaml
import dill

import re

#from multiprocessing import Pool


from sklearn import svm
#from sklearn import ensemble
#from sklearn.tree import DecisionTreeClassifier


from sklearn import metrics
from sklearn import preprocessing

from svmutil import *

import skrebate



#----------------
# INITIALIZATION
#----------------
def initialize(arguments):
    '''
    Executes the "initialize" method implemented in ants.
    '''
    # np.random.seed()
    
    individual,method = arguments
    
    if method == 'random':
        individual.initialize_random()
    else:
        pass
    
    return individual


def initialize_SP(arguments):
    '''
    Executes the "initialize" method implemented in ants.
    '''
    # np.random.seed()

    individual, method = arguments

    if method == 'random':
        individual.initialize_random_SP()
    else:
        pass

    return individual

#----------------
# EVALUATION
#----------------
def evaluate(arguments):
    '''
    Executes the "initialize" method implemented in ants.
    '''
    
    # np.random.seed()
    
    individual, X_train, Y_train, X_test, Y_test = arguments
    
    individual.evaluate(X_train, Y_train, X_test, Y_test)
    
    return individual


#---------------------
# MUTATIONS
#---------------------
def mutate(arguments):
    '''
    Executes the "search" method implemented in ants.
    '''
    
    # np.random.seed()
    
    individual,method,kwargs = arguments
    
    if method == 'constant':
        individual.mutation_constant()
    
    elif method == 'lineal':
        individual.mutation_lineal(G=kwargs['G'], Gmax=kwargs['Gmax'])
    
    elif method == 'exponential':
        individual.mutation_exponential(G=kwargs['G'], Gmax=kwargs['Gmax'])
        
    else:
        pass
    
    return individual





 
###############################################################################
class Individual(object):
    
    '''
    Definir
    '''
    
    #================================================
    # BUILDING
    #=================
    def __init__(self, identifier, Nfeatures, alpha, pm, pF, gamma_initial, gamma_final):
        '''
        Nfeatures: Total number of posible features.
        '''
        
        # TOTAL NUMBER OF FEATURES
        self.identifier = identifier
        self.hash = 0
        
        self.modified = True
        
        # TOTAL NUMBER OF FEATURES
        self.features = dict()
        self.features['selected'] = 0
        self.features['total'] = Nfeatures # INT (max features) or []
        
        
        # CHROMOSOME
        self.chromosome = []
        
        
        # FITNESS
        self.fitness = dict()
        self.fitness['total'] = 0.0
        self.fitness['features'] = 0.0
        self.fitness['uar'] = 0.0
        
        
        # PARAMETERS
        self.params = dict()
        self.params['alpha'] = alpha   # PESO RELATIVO DEL UAR EN EL FITNESS
        self.params['pm'] = pm         # PROBABILIDAD DE QUE UNA FEATURE SEA ELEGIDA DURANTE LA INICIALIZACION
        self.params['pF'] = pF         # PROBABILIDAD DE QUE UNA FEATURE SEA ELEGIDA DURANTE LA INICIALIZACION
        self.params['gamma_initial'] = gamma_initial
        self.params['gamma_final'] = gamma_final
        
    
    
    
    #================================================
    # INITIALIZATION - RANDOM
    #=========================
    def initialize_random(self):
        '''
        Definir
        '''
        
        np.random.seed()
        
        SIZE = self.features['total'] if isinstance(self.features['total'],int) else len(self.features['total'])
        
        
        #------------------------------------------
        # PROBABILISTIC INITIALIZATION
        #-------------------------------
        #RANDOM = np.random.rand(SIZE)
        #self.chromosome = [1 if RANDOM[x] < self.params['pF'] else 0 for x in range(SIZE)]
        
        NFactivated = int(round(self.params['pF'] * SIZE))
        self.chromosome = [0] * (SIZE-NFactivated) + [1] * NFactivated
        self.chromosome = np.random.permutation(self.chromosome).tolist()
        
        
        self.modified = True
        
        self.hash = np.sum(np.power(self.chromosome*np.arange(self.size()-1,-1,-1),2))
        
        return self


    # ================================================
    # INITIALIZATION - RANDOM - SUBOPOPs
    # =========================
    def initialize_random_SP(self):
        '''
        Inicialización empleada en las subpoblaciones.
        '''

        np.random.seed()

        SIZE = self.features['total'] if isinstance(self.features['total'], int) else len(self.features['total'])

        # ------------------------------------------
        # PROBABILISTIC INITIALIZATION
        # -------------------------------
        RANDOM = np.random.rand(SIZE)
        self.chromosome = [1 if RANDOM[x] < 0.5 else 0 for x in range(SIZE)]

        self.modified = True

        self.hash = np.sum(np.power(self.chromosome * np.arange(self.size() - 1, -1, -1), 2))

        return self


    
    #================================================
    # INITIALIZATION - RELIEF
    #=========================
    def initialize_relief(self, pF):
        '''
        Definir
        '''
        np.random.seed()
        
        N = int(round(self.params['pF'] * self.features['total']))
        
        F = np.random.choice(self.features['total'],N,replace=False,p=pF)
        
        self.chromosome = [1 if idx in F else 0 for idx in range(self.features['total'])]
        
        self.modified = True

        self.hash = np.sum(np.power(self.chromosome * np.arange(self.size() - 1, -1, -1), 2))

        return self
    

    #================================================
    # MUTATION - CONSTANT
    #=====================
    def mutation_constant(self):
        '''
        Definir
        '''
        np.random.seed()
        
        chromosome = [np.abs(x-1) if np.random.rand() < self.params['pm']/len(self.chromosome) else x for x in self.chromosome]
        # chromosome = self.chromosome[:]
        # for jj in range(len(chromosome)):
        #     if np.random.rand() < (self.params['pm']/len(chromosome)):
        #         chromosome[jj] = np.abs(chromosome[jj] - 1)

        if chromosome != self.chromosome:
            
            self.modified = True
            
            self.chromosome = chromosome[:]
            self.features['selected'] = self.chromosome.count(1)
            
            self.hash = np.sum(np.power(self.chromosome*np.arange(self.size()-1,-1,-1),2))
        
        return self
        
    
    #================================================
    # MUTATION - LINEAL
    #===================
    def mutation_lineal(self, G, Gmax):
        '''
        '''
        np.random.seed()
        pass
#        self.modified = True
        
        
        
    
    
    #================================================
    # MUTATION - EXPONENTIAL
    #========================
    def mutation_exponential(self, G, Gmax):
        '''
        G: Current generation.
        '''
        np.random.seed()

        ww = self.size()
        qq = len(self.chromosome)

        gamma1 = self.params['gamma_initial']/self.size()
        gamma2 = self.params['gamma_final']/self.size()
        # gamma1 = self.params['gamma_initial'] / len(self.chromosome)
        # gamma2 = self.params['gamma_final'] / len(self.chromosome)

        pm = gamma1 * np.power((gamma2/gamma1), (G - 1)/ float(Gmax))
        
        # chromosome = [int(np.abs(x-1)) if np.random.rand() < pm else x for x in self.chromosome]
        chromosome = self.chromosome[:]
        for jj in range(len(chromosome)):
            if np.random.rand() < pm:
                chromosome[jj] = np.abs(chromosome[jj] - 1)
        
        if chromosome != self.chromosome:
            
            self.modified = True
            
            self.chromosome = chromosome[:]
            self.features['selected'] = chromosome.count(1)

            self.hash = np.sum(np.power(self.chromosome*np.arange(self.size()-1,-1,-1),2))
            
        
        else:
            print('Unchanged chromosome...')
        
        return self
    
    
    #================================================
    # EVALUATE
    #=================
    def evaluate(self, X_train, Y_train, X_test, Y_test):
        '''
        Evaluate chromosome.
        
        X_train = [
                    [], <---- Partición de entrenamiento
                    [],
                    [],
                    ...
                  ]
        
        '''
        np.random.seed()
        
        if self.modified:

            # EXTRACTING SELECTED FEATURES FROM PATTERNS
            features = self.get_selected_features()
            len_full_crom = X_train.shape[2]
            UAR = []
            for x_train, y_train, x_test, y_test in zip(X_train, Y_train, X_test, Y_test):
                
                x_train = x_train[:,features]
                x_test = x_test[:,features]
                
                # STANDARIZE DATA
                # scaler = preprocessing.StandardScaler().fit(x_train)
                # conviene escalar los datos completos una unica vez previo a la evolucion \
                # de manera evitar este calculo en cada evaluacion de fitness
                # x_train = scaler.transform(x_train)
                # x_test = scaler.transform(x_test)
                
                #=====================
                # SET UP CLASSIFIER
                #=====================
                #clf = svm.SVC(kernel='linear', C=1, cache_size=500)
                #clf = svm.SVC(C=10.0, kernel='poly', degree=1, gamma=0.1, coef0=1.0, shrinking=True, probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
                
                # configs_trn=" -s 0 -t 1 -c 10 -g 0.1 -r 1 -d 1 -q "
                #clf = svm.SVC(kernel='linear', C=1, cache_size=500)
                
                # clf = Classifier(layers=[Layer("Sigmoid", units=1200),Layer("Sigmoid", units=800),Layer("Sigmoid", units=400),Layer("Sigmoid", units=200),Layer("Softmax")],learning_rate=0.05, learning_momentum=0.9, regularize=None, n_iter=1550, debug=False, verbose=False)
                
                # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
                #clf = DecisionTreeClassifier(random_state=0)

                #======================
                # TRAIN
                #======================
                #clf.fit(x_train,y_train)
                
                #======================
                # SVM PREDICTION
                #======================
                #y_pred = clf.predict(x_test)

                # ======================
                # libSVM
                # ======================
                prob = svm_problem(y_train.tolist(), x_train.tolist())
                param = svm_parameter('-s 0 -t 1 -c 10 -g 0.1 -r 1 -d 1 -q')
                model = svm_train(prob, param)
                p_labels, p_acc, p_val = svm_predict(y_test.tolist(), x_test.tolist(), model, '-q')

                #===============================
                # CONFUSSION MATRIX CALCULATION
                #===============================
                # M = np.float64(metrics.confusion_matrix(y_test,y_pred))#,labels=Train_data['class']))
                M = np.float64(metrics.confusion_matrix(y_test, np.asarray(p_labels)))

                uar = np.mean(np.diag(M)/np.sum(M,1))

                UAR.append(uar)
            
            
            #===============================
            # FITNESS CALCULATION
            #===============================
            self.fitness['uar'] = np.mean(UAR)
            
            self.fitness['features'] = self.chromosome.count(1)
            
            # self.fitness['total'] = self.params['alpha'] * self.fitness['uar'] + \
            #                                                (1.0-self.params['alpha']) * \
            #                                                (1.0 - float(self.fitness['features']/self.size()))
            self.fitness['total'] = self.params['alpha'] * self.fitness['uar'] + \
                                    (1.0 - self.params['alpha']) * \
                                    (1.0 - float(self.fitness['features'] / len_full_crom))


            self.modified = False
        
        
        return self
        
    
    #================================================
    # GET SELECTED FEATURES
    #=======================
    def get_selected_features(self):
        '''
        Shows a summary of the individual.
        '''
        
        if isinstance(self.features['total'],int):
            
            features = [idx for idx in range(self.size()) if self.chromosome[idx] == 1]
            
        else:
            
            features = [self.features['total'][idx] for idx in range(self.size()) if self.chromosome[idx] == 1]
        
        return features
    
    
    
    #================================================
    # SUMMARY
    #=================
    def summary(self):
        '''
        Shows a summary of the individual.
        '''
        print('============================')
        print('IDENTIFIER: %d' % (self.identifier))
        print('============================')
        
        print('NFEATURES: %d' % (self.features['total']))
        print('SELECTED FEATURES: %d' % ( self.chromosome.count(1) ))
        
        print('CHROMOSOME: %s' % (self.chromosome))
        
        print('----------------------------')
        print('FITNESS: %0.4f' % (self.fitness['total']))
        print('----------------------------')
        print('FEATURES: %0.4f' % (self.fitness['features']))
        print('UAR: %0.4f' % (self.fitness['uar']))
        print('============================')
    
    
    
    #================================================
    # CLONE
    #=================
    def clone(self):
        '''
        Definir
        '''
        
        return copy.deepcopy(self)    
    
    #================================================
    # SIZE
    #=================
    def size(self):
        '''
        Return the number of active features in the chromosome.
        '''
        
        return len(self.chromosome)
    
    
    
    
    #================================================
    # UPDATE
    #=================
    def update_hash(self):
        '''
        '''
        
        hash_value = np.sum(np.power(self.chromosome*np.arange(self.size()-1,-1,-1),2))
        
        if self.hash != hash_value:
            
            self.hash = hash_value
            self.modified = True
        
        return self
    
    
    #================================================
    # UPDATE
    #=================
    def reshape_chromosome(self, Nfeatures):
        '''
        '''
        
        selected_features = self.get_selected_features()
        
        chromosome = [1 if idx in selected_features else 0 for idx in range(Nfeatures)]
        
        self.chromosome = chromosome[:]
        self.features['total'] = self.size()
        self.features['selected'] = self.chromosome.count(1)
        
        return self

###############################################################################




###############################################################################
class Population(list):
    '''
    EXPLICACION
    '''
    
    
    #===========================================
    def __init__(self, Nindiv=100, px=0.8, alpha=0.90, Nfeatures=100, pm=0.01, pF=0.1, gamma_initial=10, gamma_final=0.01):
        '''
        Nindiv=100
        px=0.8
        pm=0.05
        alpha=0.65
        Nfeatures=100
        '''
        self.fitness = np.zeros(Nindiv,dtype=np.float64)
        
        # ELITE
        self.elite_idx = 0
        self.elite_fitness = 0.0
        
        self.params = dict()
        self.params['px'] = px
        self.params['Nindiv'] = Nindiv
        
        self.extend([Individual(identifier=idx,
                                Nfeatures=Nfeatures,
                                alpha=alpha,
                                pm=pm,
                                pF=pF,
                                gamma_initial=gamma_initial,
                                gamma_final=gamma_final
                               ) for idx in range(0,Nindiv)])
    
    
    
    #===========================================
    def build_offspring(self, Nchild, Ngap, G, Gmax):
        '''
        '''
        
        Offspring = self.clone()
        Offspring.clear()
        Offspring.Nindiv = Nchild
        
        Gap = self.clone()
        Gap.clear()
        Gap.Nindiv = Ngap
        #================================
        
        for ii in range(Nchild):
            
            idxs = np.random.permutation(self.size()).tolist()
            idx1 = idxs.pop()
            idx2 = idxs.pop()
            
#            while idxs and (self[idx1].hash == self[idx2].hash):
#                idx2 = idxs.pop()
            
            
            P1 = self[idx1].clone()
            P2 = self[idx2].clone()
            
            #-----------------------------------------------
            # BUILD GAP
            if Gap.size() < Gap.Nindiv:
                
                # ---------------------
                # INSERTO PRIMER PADRE
                # ---------------------
                Gap.append(P1.clone())
            #-----------------------------------------------
            
            
            # Apply mutation
            P1.mutation_exponential(G, Gmax)
            P2.mutation_exponential(G, Gmax)
            
            O1 = P1.clone()
            O2 = P2.clone()
            
            # SI HAY CRUZA > # CRUZO LOS PADRES
            if np.random.rand() < self.params['px']:
                
                if P1.size()>1:
                    idx = np.random.randint(1,P1.size())
                else:
                    idx = 0   
                
                # -----------------
                # FIRST CHILD
                # -----------------
                O1.chromosome = []
                O1.chromosome.extend(P1.chromosome[:idx])
                O1.chromosome.extend(P2.chromosome[idx:])
                
                
                # -----------------
                # SECOND CHILD
                # -----------------
                O2.chromosome = []
                O2.chromosome.extend(P2.chromosome[:idx])
                O2.chromosome.extend(P1.chromosome[idx:])
                O2.update_hash()
                
            
            if Offspring.size() < Offspring.Nindiv:
                
                # ---------------------
                # INSERTO PRIMER HIJO
                # ---------------------
                O1.update_hash()
                Offspring.append(O1)
            
            if Offspring.size() < Offspring.Nindiv:
                
                # ---------------------
                # INSERTO SEGUNDO HIJO
                # ---------------------
                O2.update_hash()
                Offspring.append(O2)
                
        
        return (Gap,Offspring)
    
    
    #===========================================
    def roulette_selection(self, N):
        
        P2X = self.clone()
        P2X.clear()
        
        f = self.fitness[:]
        p = f/np.sum(f)
        
        idx = np.random.choice(range(self.Nindiv), size=N, p=p)
        
        for ii in idx:
            P2X.append(self[ii].clone())
        
        return P2X

    # ===========================================
    def sort_pop(self):

        pop_temp = self.clone()
        self.clear()

        FITNESS = [indiv.fitness['total'] for indiv in pop_temp]

        idxs = [i[0] for i in sorted(enumerate(FITNESS), key=lambda x: x[1], reverse=True)]

        idxs = idxs[:self.params['Nindiv']]

        for ii,idx in enumerate(idxs):
            self.append(pop_temp[idx].clone())
            self.fitness[ii] = FITNESS[idx]

        #self = [self[i] for i in idxs]
        #self.fitness = [self.fitness[i] for i in idxs]

        return self
    
    #===========================================
    def clear(self):
        '''
        '''
        self[:] = []
        self.fitness.fill(0.0)
#        while self:
#            self.pop()
    
    
    
    #===========================================
    def evaluate(self, X_train, Y_train, X_test, Y_test, jobs_server='', show=False):
        '''
        Realiza una iteración de búsqueda (cada hormiga realiza la búsqueda una
        vez.
        
        --- MULTIPROCESSING ---
        jobs_server = multiprocessing.Pool(ncpus)
        
        '''
        
        FITNESS = dict()
        FITNESS['features'] = []
        FITNESS['uar'] = []
        FITNESS['total'] = []
        
        if jobs_server == '':
            
            
            for idx in range(0,self.size()):
                
                #=======================================
                self[idx].evaluate(X_train, Y_train, X_test, Y_test)
                self.fitness[idx] = self[idx].fitness['total']
                
                FITNESS['total'].append(self[idx].fitness['total'])
                FITNESS['uar'].append(self[idx].fitness['uar'])
                FITNESS['features'].append(self[idx].fitness['features'])
                #=======================================
                
                if self.fitness[idx] > self.elite_fitness:
                    self.elite_idx = idx
                    self.elite_fitness = self.fitness[idx]
            
        
        else:
            
            # MULTIPROCESSING
            INDIVIDUOS = jobs_server.map(evaluate,
                                         [(individual, X_train, Y_train, X_test, Y_test) for individual in self]
                                         )
            
            self[:] = []
            self.extend(INDIVIDUOS)

            self.fitness.fill(0.0)
            
            for idx in range(0,self.size()):
            
                #=======================================
                self.fitness[idx] = self[idx].fitness['total']
                
                FITNESS['total'].append(self[idx].fitness['total'])
                FITNESS['uar'].append(self[idx].fitness['uar'])
                FITNESS['features'].append(self[idx].fitness['features'])
                #=======================================
                
                if self.fitness[idx] > self.elite_fitness:
                    self.elite_idx = idx
                    self.elite_fitness = np.float64(self.fitness[idx])
            
#            print(np.mean(FITNESS['features']))
        
        if show:
            print('-----------------------------')
            print('MEDIDAS --- GENERALES')
            print('-----------------------------')
            print('FEATURES:\t%0.2f (%0.2f) +/- %0.2f\n' % (np.mean(FITNESS['features']),np.median(FITNESS['features']),np.std(FITNESS['features'])))
            print('UAR:\t\t%0.4f (%0.4f) +/- %0.4f\n' % (np.mean(FITNESS['uar']),np.median(FITNESS['uar']),np.std(FITNESS['uar'])))
            print('FITNESS:\t%0.4f (%0.4f) +/- %0.4f\n' % (np.mean(FITNESS['total']),np.median(FITNESS['total']),np.std(FITNESS['total'])))
            
            print('-----------------------------')
            print('MEDIDAS --- ELITE')
            print('-----------------------------')
            print('FEATURES:\t%0.2f\n' % (FITNESS['features'][self.elite_idx]))
            print('UAR:\t\t%0.4f\n' % (FITNESS['uar'][self.elite_idx]))
            print('FITNESS:\t%0.4f\n' % (FITNESS['total'][self.elite_idx]))
            print('\n\n')
            
        return FITNESS
    
    
    
        
    #===========================================
    def size(self):
        return len(self)
    
    
    #===========================================
    def clone(self):
        
        return copy.deepcopy(self)
    
    
    #===========================================
    def crossover(self, Nchild):
        '''
        One point crossover.
        '''
        
        Offspring = self.clone()
        Offspring.clear()
        Offspring.Nindiv = Nchild
        
        #================================
        
        for ii in range(Nchild):
            
            idxs = np.random.permutation(self.size()).tolist()
            idx1 = idxs.pop()
            idx2 = idxs.pop()
            
            while idxs and (self[idx1].hash == self[idx2].hash):
                idx2 = idxs.pop()
            
            
            O1 = self[idx1].clone()
            O2 = self[idx2].clone()
            O1.modified = True
            O2.modified = True
            
            # SI HAY CRUZA > # CRUZO LOS PADRES
            if np.random.rand() < self.params['px']:
                
                # SELECCIONO LOS PADRES
#                P1 = self[idx[0]].clone()
#                P2 = self[idx[1]].clone()
                
                P1 = self[idx1].clone()
                P2 = self[idx2].clone()
                
                IDX = np.random.randint(1,P1.size())

                # -----------------
                # FIRST CHILD
                # -----------------
                O1.chromosome = []
                O1.chromosome.extend(P1.chromosome[:IDX])
                O1.chromosome.extend(P2.chromosome[IDX:])
                
                # -----------------
                # SECOND CHILD
                # -----------------
                O2.chromosome = []
                O2.chromosome.extend(P2.chromosome[:IDX])
                O2.chromosome.extend(P1.chromosome[IDX:])
            
            if Offspring.size() < Offspring.Nindiv:
                
                # ---------------------
                # INSERTO PRIMER HIJO
                # ---------------------
                Offspring.append(O1)
            
            if Offspring.size() < Offspring.Nindiv:
                
                # ---------------------
                # INSERTO SEGUNDO HIJO
                # ---------------------
                Offspring.append(O2)
                
        
        return Offspring
    
    
    #===========================================
    def roulette_selection(self, N):
        
        P2X = self.clone()
        P2X.clear()
        
        f = self.fitness[:]
        p = f/np.sum(f)
        
        idx = np.random.choice(range(self.Nindiv), size=N, p=p)
        
        for ii in idx:
            P2X.append(self[ii].clone())
        
        return P2X
    
    
    #===========================================
    def tournament_selection(self, N, K=5):
        
        P2X = self.clone()
        P2X.clear()
        
        
        for ii in range(N):
            
            idx = np.random.choice(range(self.size()), size=K, replace=False)
            
            # ii = np.argmax(self.fitness[idx])
            # P2X.append(self[idx[ii]].clone())

            j = np.argmax(self.fitness[idx])
            P2X.append(self[idx[j]].clone())
        
        return P2X
    
    
    #===========================================
    def initialize(self, jobs_server='', method='random', features=None, labels=None):
        '''
        Inicializa el hormiguero
        '''
        
        #-----------------------------------
        # SINGLE PROCESSING
        #-----------------------------------
        if jobs_server == '':
            
            if method == 'random':
                self = [individual.initialize_random() for individual in self]
            
            if method == 'relief':
                
                RF = skrebate.ReliefF(n_features_to_select=2, n_neighbors=100).fit(features[0],labels[0])
                
                y = np.cumsum(RF.feature_importances_ - min(RF.feature_importances_))
                pF = y/max(y)
                
                self = [individual.initialize_relief(pF=pF/sum(pF)) for individual in self]
                
            else:
                pass
            
        
        
        #-----------------------------------
        # MULTIPROCESSING
        #-----------------------------------
        else:
            
            self = jobs_server.map(initialize,[(individual,method) for individual in self])

    # ===========================================
    def initialize_SP(self, jobs_server='', method='random'):
        '''
        Inicializa el hormiguero
        '''

        # -----------------------------------
        # SINGLE PROCESSING
        # -----------------------------------
        if jobs_server == '':

            if method == 'random':
                self = [individual.initialize_random_SP() for individual in self]
            else:
                pass



        # -----------------------------------
        # MULTIPROCESSING
        # -----------------------------------
        else:

            self = jobs_server.map(initialize_SP, [(individual, method) for individual in self])

    #===========================================
    def mutate(self, pool, jobs_server='', method='exponential', **kwargs):
        '''
        Realiza una iteración de búsqueda (cada hormiga realiza la búsqueda una
        vez.
        
        --- MULTIPROCESSING ---
        jobs_server = multiprocessing.Pool(ncpus)
        '''
        
        #-----------------------------------
        # SINGLE PROCESSING
        #-----------------------------------
        if jobs_server == '':
            
            if method == 'constant':
                self = [individual.mutation_constant() for individual in self]
                
            elif method == 'lineal':
                self = [individual.mutation_lineal(G=kwargs['G'], Gmax=kwargs['Gmax']) for individual in self]
                
            elif method == 'exponential':
                self = [individual.mutation_exponential(G=kwargs['G'], Gmax=kwargs['Gmax']) for individual in self]
                
            else:
                pass
        
        #-----------------------------------
        # MULTIPROCESSING
        #-----------------------------------
        else:
            
            # MULTIPROCESSING
            self = jobs_server.map(mutate,[(individual,method,kwargs) for individual in self])
    
    
    
    #======================
    def get_elite(self, N=1):
        '''
        '''
        
        Elites = []
        Indexes = []
        Hashes = []
        
        indiv = copy.deepcopy(self[self.elite_idx])
        Elites.append(indiv)
        Hashes.append(indiv.hash)
        
        if N > 1:
            
            idxs = sorted(enumerate(self.fitness), key=lambda x:x[1], reverse=True)
            
            while len(Elites) < N:
                
                idx,fitness = idxs.pop(0)
                indiv = copy.deepcopy(self[idx])
                
                if indiv.hash not in Hashes:
                    
                    Elites.append(indiv)
                    Hashes.append(indiv.hash)
                    Indexes.append(idx)
                    
        
        return (Elites, Indexes)


###############################################################################


###############################################################################
class History(object):
    
    '''
    Definir
    '''
    
    #================================================
    # BUILDING
    #=================
    def __init__(self, Gmax=1000):
        '''
        Definir
        '''
        
        # TOTAL NUMBER OF FEATURES
        self.Gmax = Gmax
        
        self.measures = dict()
        self.measures['fitness'] = dict()
        self.measures['features'] = dict()
        self.measures['uar'] = dict()
        
        self.elite = dict()
    
    
    
    
    
    #================================================
    # SAVING HISTORY
    #=================
    def save(self, filename='prueba.json'):
        '''
        '''
        
        #------------------------------------------
        # CONVERTING OBJECT TO DICTIONARY
        #---------------------------------
        History = dict()
        
        History['Gmax'] = self.Gmax
        
        History['measures'] = self.measures
        
        History['elite'] = self.elite
        #------------------------------------------
        
        
        with open(filename, 'w') as fp:
            json.dump(History, fp)
#==============================================================================



#==============================================================================
class Settings(dict):
    """EXPLICACION"""
    
    def __init__(self, filename):
        """EXPLICACION"""
        
        #-----------------------------
        # OPEN SETTINGS FILE
        #-----------------------------
        with open(filename, 'r') as f:
            settings = yaml.load(f)
        #f.close()
        
        
        for key,value in settings.items():
            self[key] = value
#==============================================================================






#==============================================================================
class LoadData(object):
    '''
    This object allows to manage the dataset for Traning, Validation and Test.
    '''
    
    
    #===========================================
    def __init__(self, train_filename, validation_filename, test_filename):
        '''
        Definir
        '''
        self.train = dict()
        self.validation = dict()
        self.test = dict()
    
    
    
    
    #===========================================
    def summary(self):
        '''
        Imprime en pantalla un resumen de los datos cargados.
        '''
        pass
    
    
    
    
    
    
    #===========================================
    def export(self):
        '''
        Exporta los datos seleccionados como una estructura para alimentar un clasificador.
        '''
        pass
    
    
    
    #===========================================
    def save(self):
        '''
        Guarda los datos seleccionados (estructura para alimentar un clasificador) a un archivo JSON.
        '''
        pass



#==============================================================================




#==============================================================================
class ARFF_Object(dict):
    '''
    Maneja los datos del archivo ARFF.
    '''
    
    def __init__(self,arff_filename):
        '''
        '''
        
        # STRUCTURE DATA RETURNED
        self['attributes'] = list()
        self['attribute_types'] = list()
        self['class'] = list()
        self['data'] = list()
        self['target'] = list()
        self['relation'] = ''
        
        self.__name = arff_filename.split('/')[-1].split('.')[0]
        
        
        # OPEN FILE
        with open(arff_filename, 'r') as fp:
            FILE = fp.read()
        
        # SPLIT DATA INTO HEADER AND DATA
        if '@DATA' in FILE:
            header,data = FILE.split('@DATA')
        elif '@data' in FILE:
            header,data = FILE.split('@data')
        else:
            print('No es posible encontrar las etiquetas "@DATA" o "@data" en el archivo ARFF')
        
        # --- HEADER PROCESSING ---
        self['relation'] = re.findall('@RELATION (.{1,100})\n', header)
        
        L = re.findall('@ATTRIBUTE (.{1,100}) (.{1,100})\n', header)
        
        for l in L:
            if l[0] != 'class':
                self['attributes'].append(l[0])
                self['attribute_types'].append(l[1][0:])
        
        
        
        # --- DATA PROCESSING ---
        if '\r\n' in data:
            patterns = data.split('\r\n')
            del patterns[0]
            del patterns[-1]
        elif '\n' in data:
            patterns = data.split('\n')
            del patterns[0]
        
        for pattern in patterns:
            
            X = list()
            
            # SPLIT INTO ATTRIBUTES
            attributes = pattern.split(',')
            
            for attribute in attributes[:-1]:
                
                X.append(float(attribute))
            
            self['data'].append(X)
            self['target'].append(attributes[-1])
        
        #self['data'] = np.array(self['data'])
        #self['target'] = np.array(self['target'])
        
        self['class'] = re.findall('@ATTRIBUTE class \{(.{1,1000})\}',FILE)[0].split(',')
        #self['class'] = list(set(self['target']))
    
      
    #===========================================
    def size(self):
        return ( len(self['target']) , len(self['data'][0]) )
    
    
    #===========================================
    def get_patterns(self, idxs=[], features=[]):
        '''
        Devuelve el listado de patrones para los patrones indicados en la lista de índices "idx".
        '''
        
        patterns = self['data'][:]
        
        if idxs:
            patterns = patterns[idxs,:][:]
        
        if features:
            patterns = patterns[:,features][:]
        
        return patterns
    
    
    #===========================================
    def get_class(self, idxs=[]):
        '''
        Devuelve el listado de clases para los patrones indicados en la lista de índices "idxs".
        '''
        
        if idxs == []:
            return self['class']
            
        else:
            return [self['class'][idx] for idx in idxs]
    
    
    
    #===========================================
    def class_summary(self,show=False):
        '''
        Devuelve un vector conteniendo el número de patrones por clase.
        '''
        values = [ self['target'].count(x) for x in self['class'] ]
        
        if show == True:
            for ii in range(0,len(self['class'])):
                print(self['class'][ii] + ': ' + str(values[ii]))
        
        return values
    
        
    #===========================================
    def save(self, format_file='dill'):#,_format='json'
        '''
        Guarda el objeto en un archivo serializado con extensión PKL (pickle).
        '''
        
        #------------------------------------------------------------
        if format_file == 'dill':
            with open(self.__name + '.dill', "wb" ) as fp:
                dill.dump( self, fp )
        
        
        #------------------------------------------------------------        
        elif format_file == 'json':
            
            DATA = dict()
            DATA['attributes'] = self['attributes']
            DATA['attribute_types'] = self['attribute_types']
            DATA['class'] = self['class']
            DATA['data'] = self['data']
            DATA['target'] = self['target']
            DATA['relation'] = self['relation']
            
            with open(self.__name + '.json', "w" ) as fp:
                json.dump( DATA, fp )
        
        
        #------------------------------------------------------------
        elif format_file == 'csv':
            
            DATA = []
            for X,Y in zip(self['data'],self['target']):
                
                X = ','.join([str(x) for x in X])
                DATA.append(X + ',' +  str(self['class'].index(Y)+1) + '\n')
            
            with open(self.__name + '.csv', "w" ) as fp:
                fp.writelines(DATA)
        
        
        #------------------------------------------------------------
        else:
            print('Unknow format type.')
#==============================================================================




#==============================================================================
class Measures(object):
    '''
    This object allows to manage the dataset for Traning, Validation and Test.
    '''
    
    
    #===========================================
    def __init__(self):
        '''
        Definir
        '''
        
        self.G = 0
        self.time = dict()
        self.time['initialization'] = 0.0
        self.time['execution'] = 0.0
        self.time['total'] = 0.0
        
        self.current = dict()
        self.elite = dict()
        self.history = dict()
    
    
    #===========================================
    def _mad(self,values):
        '''
        '''
        
        return np.median( np.absolute( values - np.median(values) ) )
        
        
    
    #===========================================
    def add(self, key):
        '''
        '''
        
        self.current[key] = []
        
        self.elite[key] = []
        
        self.history[key] = dict()
        self.history[key]['mean'] = []
        self.history[key]['std'] = []
        self.history[key]['median'] = []
        self.history[key]['mad'] = []
        self.history[key]['max'] = []
        self.history[key]['min'] = []
    
    
    #===========================================
#    def get_values(self, key, measure='mean', generation=None):
#        '''
#        '''
#        if generation == None:
#            
#            return self.history[key][measure]
#        
#        else:
#            
#            return self.history[key][measure][generation]
    
        
    #===========================================
    def mad(self, key):
        '''
        '''
        
        measure = self.history[key]
        
        return self._mad(measure)
    
    
    #===========================================
    def maximum(self, key):
        '''
        '''
        return np.max(self.history[key])
    
    
    #===========================================
    def mean(self, key):
        '''
        '''
        return np.mean(self.history[key])
    
    
    #===========================================
    def median(self, key):
        '''
        '''
        return np.median(self.history[key])
    
    
    #===========================================
    def minimum(self, key):
        '''
        '''
        return np.min(self.history[key])
        
    
    #================================================
    # SAVING HISTORY
    #=================
    def export(self, key):
        '''
        '''
        
        if key == 'time':
            return self.time
        
        elif key == 'current':
            return self.current
        
        elif key == 'elite':
            return self.elite
            
        elif key == 'history':
            return self.history
        
        else:
            print('Available keys are: "time", "current", "elite", and "history".')
            X = dict()
            return X
    
    
    #===========================================
    def std(self, key):
        '''
        '''
        return np.std(self.history[key])
    
    
    #===========================================
    def summary(self):
        '''
        '''
        pass
    
    
    #===========================================
    def update(self, key, values, idx_elite):
        '''
        Update current "values" for the specified "key", and then update the
        historical values associated to this key.
        '''
        
        #---------------
        # CURRENT
        #----------
        self.current[key] = values
        
        
        #---------------
        # ELITE
        #----------
        self.elite[key].append(values[idx_elite])
        
        
        #---------------
        # HISTORY
        #----------
        self.history[key]['mean'].append(float(np.mean(values)))
        self.history[key]['std'].append(float(np.std(values)))
        self.history[key]['median'].append(float(np.median(values)))
        self.history[key]['mad'].append(self._mad(values))
        self.history[key]['max'].append(float(np.max(values)))
        self.history[key]['min'].append(float(np.min(values)))


#==============================================================================
