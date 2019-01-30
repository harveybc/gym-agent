# agent_test_v2:Load a dataset generated by q-datagen for reading the reward per 
#               action  and a dataset generated from MT4 CSV-export.mq4 for simulating
#               the trade strategy. It does NOT  use a model to predict the reward  
#               but takes the training signals directly from que q-datagen dataset.
#               the objective is to test if the reward function and the strategy used 
#               for trding with them is correct in an ideal scenario.

import gym
import gym.wrappers
import gym_forex
from gym.envs.registration import register
import sys
import neat
import os
from joblib import load
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import operator
from numpy import genfromtxt
import csv
from sklearn import svm
from operator import add
from joblib import dump, load
from sklearn import preprocessing
            
## \class QAgent
## \brief Q-Learning agent that uses an OpenAI gym environment for fx trading 
## estimating for each tick, the optimal SL, TP, and Volume.
class QAgent():    
    ## init method
    ## Loads the validation dataset, loads the pre-trained models
    #  initialize forex environment.
    def __init__(self):
        # First argument is the validation dataset, including headers indicating maximum and minimum per feature
        self.vs_f = sys.argv[1]
        # Second argument is the prefix (including path) for the dcn pre-trained models 
        # for the actions, all modes are files with .svm extention and the prefix is
        # concatenated with a number indicating the action:
        # 0 = Buy/CloseSell/nopCloseBuy
        # 1 = Sell/CloseBuy/nopCloseSell
        # 2 = No Open Buy
        # 3 = No Open Sell
        self.model_prefix = sys.argv[2]
        # third argument is the path of the datasset to be used in the gym environment (not q-datagen generated, without headers) 
        self.env_f = sys.argv[3]
        # initialize gym-forex env (version 4)
        self.test_episodes = []
        self.generation = 0
        self.min_reward = -15
        self.max_reward = 15
        self.episode_score = []
        self.episode_length = []
        self.svr_rbf = svm.SVR(kernel='rbf')
        self.num_s = 19
        self.model = [self.svr_rbf] * self.num_s 
        self.raw_action = 0
        self.max_index = 0
        self.vs_data = []
        self.vs_num_ticks = 0
        self.vs_num_columns = 0
        self.obsticks = 30
        # TODO: obtener min y max de actions from q-datagen dataset headers
        self.min_TP = 100
        self.max_TP = 3000
        self.min_SL = 100
        self.max_SL = 3000
        self.min_volume = 0.0
        self.max_volume = 0.1
        self.security_margin = 0.1
        # load pre-processing settings
        self.pt = preprocessing.PowerTransformer()
        print("loading pre-processing.PowerTransformer() settings for the generated dataset")
        self.pt = load(self.vs_f+'.powertransformer')
        # load feature-selection mask
        print("loading pre-processing feature selection mask")
        self.mask = load(self.vs_f+'.feature_selection_mask')
        
        
        
        # register the gym-forex openai gym environment
        # TODO: extraer obs_ticks como el window_size, desde los headers de  salida de q-datagen
        register(
            id='ForexValidationSet-v1',
            entry_point='gym_forex.envs:ForexEnv6',
            kwargs={'dataset': self.env_f ,'max_volume':self.max_volume, 'max_sl':self.max_SL, 
                    'max_tp':self.max_TP, 'min_sl':self.min_SL,
                    'min_tp':self.min_TP,'obsticks':self.obsticks, 
            'capital':800, 'leverage':100, 'num_features': 17}
        )
        # make openai gym environments
        self.env_v = gym.make('ForexValidationSet-v1')
        # Shows the action and observation space from the forex_env, its observation space is
        # bidimentional, so it has to be converted to an array with nn_format() for direct ANN feed. (Not if evaluating with external DQN)
        print("action space: {0!r}".format(self.env_v.action_space))
        print("observation space: {0!r}".format(self.env_v.observation_space))
        # read normalization maximum and minimum per feature
        # n_data_full = genfromtxt(self.vs_f, delimiter=',',dtype=str,skip_header=0)    
        with open(self.vs_f, newline='') as f:
            reader = csv.reader(f)
            n_data = next(reader)  # gets the first line
        # read header from vs_f
        #n_data = n_data_full[0].tolist()
        self.num_columns = len(n_data)
        print("vs_f num_columns = ", self.num_columns)
        # minimum and maximum per feature for normalization before evaluation in pretrained models
        self.max = [None] * self.num_columns
        self.min = [None] * self.num_columns
        for i in range(0, self.num_columns-self.num_s):
            header_cell = n_data[i]
            #print("header_cell = ", header_cell, "type = " ,type(header_cell))
            data = header_cell.split("_")
            num_parts = len(data)
            self.max[i] = float(data[num_parts-1])
            self.min[i] = float(data[num_parts-2])
            # data was mormalized as: my_data_n[0, i] = (2.0 * (my_data[0, i] - min[i]) / (max[i] - min[i])) - 1
        
    ## the action model is the same q-datagen generated dataset
    def load_action_models(self):
        self.vs_data = genfromtxt(self.vs_f, delimiter=',')
        # get the number of observations
        self.vs_num_ticks = len(self.vs_data)
        self.vs_num_columns = len(self.vs_data[0])

    ## For an observation for each tick, returns 0 if the slope of the future(10) MACD signal (output 16 zero-based) is negative, 1 if its positive. 
    def decide_next_action(self, normalized_observation):
        # evaluate all models with the observation data window 
        self.action = []
        self.max_index = 0 
        action_list = [0,0,0]
        vs = np.array(normalized_observation)
        # read the normalized_observation skipping (num_features-1) and sum the values to compare with
        # the sum of the same sum from the validation set.
        a_pattern = 0
        num_features = (self.vs_num_columns-self.num_s)//self.obsticks
        n_p = -1
        for i in range(0, num_features):
            n_p = n_p * -1
            # print("num_features= ",num_features ," len(obs)=",len(normalized_observation), "i=",i)
            if n_p == 1:            
                a_pattern = a_pattern + normalized_observation[self.obsticks * i]
            else:
                #print("len(normalized_observation)=",len(normalized_observation)," i=",i)
                a_pattern = a_pattern * normalized_observation[self.obsticks * i]
        #  for each row of the validation set(output of q-datagen), do the sum and compare with the observation sum
        index = 0
        
        for i in range(1, self.vs_num_ticks):
            a_search = 0
            n_p = -1
            # do the sum of the values per feature to compare with the q-datagen dataset output
            for j in range(0, num_features):
                n_p = n_p * -1
                if n_p == 1:
                    a_search = a_search + self.vs_data[i, self.obsticks * j]
                else:
                    a_search = a_search * self.vs_data[i, self.obsticks * j]
            # Return all values from the action signals
            if a_pattern == a_search:
                action_list_n = self.vs_data[i, self.vs_num_columns-3 : self.vs_num_columns].copy()
                action_list = action_list_n.tolist()
                break
        #print("normalized_observation=", normalized_observation)
        #print("a_pattern=", a_pattern, " a_search=", a_search, " index=", index)
        # VOILA!
        self.action = action_list.copy()
        return self.action

    ## normalize the observation matrix, converts it to a list feedable to a pretrained DcN
    # oldest data is first in dataset and also in observation matrix
    # input obs_matrix, prev obs_matrix, output:row
    def normalize_observation(self, observation, observation_prev):
        # observation is a list with size num_features of numpy.deque of size 30 (time window) 
        n_obs = []
        num_columns_o = len(observation)
        #print("num_columns_o = ", num_columns_o)
        # compose list from observation matrix similar to a row of the training set output from q-datagen (tick contiguous per feature)
        for i in range (0, num_columns_o):
            l_obs = list(observation[i])   
            for j in l_obs:
                n_obs.append(j)
        # append list of the returned values 
        # TODO: Cambiar a recorrido de l_obs restando el anterior y solo usar l_obs_prev para el primer elemento
        for i in range (0, num_columns_o):
            l_obs = list(observation[i])   
            l_obs_prev = list(observation_prev[i])   
            # l_dif = l_obs - l_obs_prev
            l_dif = list( map(add, l_obs, l_obs_prev) )
            for l in l_obs:
                n_obs.append(l)
        
        # append 10 columns used to preprocess test training signals TODO: QUITAR DE AQUI Y DEL DATASET
        for i in range(0,10):
            n_obs.append(0) 
        #apply pre-processing
        n_obs = self.pt.transform(np.array(n_obs).reshape(1,-1))
        n_o = n_obs[0].tolist()
        #print("n_o=",n_o)
        
        # append 10 columns used to preprocess test training signals TODO: QUITAR DE AQUI Y DEL DATASET
        for i in range(0,9):
            n_o.append(0)
        #apply feature selection.
        #print("n_o=",n_o)
        #print("mask=",self.mask)
        #print("len(n_o)=",len(n_o))
        #print("len(mask)", len(self.mask))
        n_obs=np.array(n_o)
        
        n_obs = n_obs[self.mask]
    
        return n_obs
    
    ## Function transform_action: convert the output of the raw_action into the
    ## denormalized values to be used in the simulation environment.
    ## increase the SL in the sec_margin% and decrease the TP in the same %margin, volume is also reduced in the %margin  
    def transform_action(self, order_status):
        # order_status:  0 nop, -1=sell,1=buy
        # the variable self.raw_action contains the output of decide_next_action, which is an array of 3 values, MACD signal return, RSI return and MACD main - signal >0?
        # the output actions are: 0=TP,1=SL,2=volume(dInv). 
        # if there is no opened order
        act = []
        # initialize values for next order , dir: 1=buy, -1=sell, 0=nop
        dir = 0
        tp = 0.1
        sl = 1.0
        vol  = 1.0
        # if there is no opened order
        if order_status == 0:
            # si el action[0] > 0, compra, sino vende
            if (self.raw_action[0] > 0):
                # opens buy order  
                dir = 1
            else:
                dir = -1
        
        # if there is an existing buy order
        if order_status == 1:
            # si action[0] == 0 cierra orden de buy 
            if (self.raw_action[0] == 0):
                # closes buy order  
                dir = -1
               
        # if there is an existing sell order               
        if order_status == -1:
            # if action[0]>0, closes the sell order
            if (self.raw_action[0] > 0):
                # closes sell order  
                dir = 1
             
        # Create the action list output [tp, sl, vol, dir]
        act.append(tp)
        act.append(sl)
        act.append(vol)  
        act.append(dir)
        return act
    
    ## Evaluate all the steps on the simulation choosing in each step the best 
    ## action, given the observations per tick. 
    ## \returns the final balance and the cummulative reward
    # Posssible actions: 
    # 0 = Buy/CloseSell/nopCloseBuy
    # 1 = Sell/CloseBuy/nopCloseSell
    # 2 = No Open Buy
    # 3 = No Open Sell
    def evaluate(self):
        # calculate the validation set score
        hist_scores = []
        observation = self.env_v.reset()
        #print("observation = ", observation)
        normalized_observation = agent.normalize_observation(observation, observation) 
        #print("normalized_observation = ", normalized_observation)
        score = 0.0
        step = 0
        order_status=0
        while 1:
            step += 1
            self.raw_action = self.decide_next_action(normalized_observation)
            action = self.transform_action(order_status)
            # print("raw_action=", raw_action, " action=", action,)
            # TODO: verificar que datos usados en training sean inguales a los usados en evaluate()
            #       verificar primera fila de pretrainer ts y primera fila que se envía a svm en evaluate()
            #       comparar que ambas predicciones den los mismos valores para las self.num_s acciones
            # TODO: probar con DCN
            # TODO: exportar plots de pre-trainer como imagenes
            # TODO: verificar que fórmulas para cada action reward son correctas, haciendo 
            #       modelo pre-entrenado que retorna para cada lecctura los valores exáctos de 
            #       reward de cada acción basado en tabla de training apra simular mejor caso
            #if step > 1:
            #    print("a=", action, " order_status=",info['order_status'], " num_closes=", info['num_closes']," balance=",info['balance'], " equity=", info['equity'])
            observation_prev = observation.copy()
            observation, reward, done, info = self.env_v.step(action)
            order_status=info['order_status']
            normalized_observation = self.normalize_observation(observation, observation_prev)
            score += reward
            #env_v.render()
            if done:
                break
        hist_scores.append(score)
        avg_score = sum(hist_scores) / len(hist_scores)
        print("Validation Set Score = ", avg_score)
        print("*********************************************************")
        return avg_score     

    def show_results(self):
        test=0

# main function 
if __name__ == '__main__':
    agent = QAgent()
    agent.load_action_models()
    agent.evaluate()