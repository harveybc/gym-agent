# agent_test_dcn:uses a pre-trained dcn to trade in a historic timeseries.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import gym.wrappers
import gym_forex
from gym.envs.registration import register
import sys
import neat
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import operator
from numpy import genfromtxt
import csv
from sklearn import svm
from operator import add,sub
from joblib import dump, load
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD, Adamax
            
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
        self.window_size = self.obsticks
        # TODO: obtener min y max de actions from q-datagen dataset headers
        self.min_TP = 100
        self.max_TP = 10000
        self.min_SL = 100
        self.max_SL = 10000
        self.min_volume = 0.0
        self.max_volume = 0.1
        self.security_margin = 0.1
        self.test_action = 0
        self.num_features = 0
        self.learning_rate = 0.0002
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
            'capital':800, 'leverage':100, 'num_features': 13}
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

    def set_dcn_model(self):
        # Deep Convolutional Neural Network for Regression
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # first set of CONV => RELU => POOL
        # mejor result 0.1 con dropout de 0.4 en 400 epochs con learning rate 0.0002 en config  521,64,32,16, en h4 2018 con indicator_period=70
        # 0.2,0.1,lr=0.0002 1200 eva: 0.117
        # 0.4,eva = 0.108
        model.add(Dropout(0.4,input_shape=(self.num_features,self.window_size)))
        model.add(Conv1D(512, 3))
        model.add(Activation('sigmoid'))
        # Sin batch_normalization daba: 0.204
        # Con batch normalization: e=0.168
        model.add(BatchNormalization())
        # Con dropout = 0.1, e=0.168
        # con dropout = 0.2, e=0.121
        # con dropout = 0.4, e= 0.114
        model.add(Dropout(0.4))
        #sin capa de LSTM50,  e=0.107
        #con capa de LSTM50, e= 0.191
        #model.add(LSTM(units = 50, return_sequences = True))
        
        #model.add(Dropout(0.2))
        # mejor config so far: D0.4-512,D0.2-64,d0.1-32,16d64 error_vs=0.1 con 400 epochs y lr=0.0002
        # sin capa de 64, eva = 0.114
        # on capa de 128, eva = 0.125
        # on capa de 32,  eva = 0.107
        # on capa de 16,  eva = 0.114
        model.add(Conv1D(32, 3))
        model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())

        # con otra capa de 32, eva5 = 0.126
        # sin otra capa de 32, eva5 = 0.107, sin minmax normalization
        # sin otra capa de 32, eva5 = 0.124 , con minmax normalization antes de power transform
        #model.add(Conv1D(32, 3))
        #model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        
        # con capa de 16 da   eva5= 107
        model.add(Conv1D(16, 3))
        model.add(Activation('sigmoid'))
        model.add(BatchNormalization())

        #sin capa de LSTM50, eva3=0.104 probar con 400 epochs
        #con capa de LSTM50, eva3= 0.212
        #model.add(LSTM(units = 50, return_sequences = True))
        
        #model.add(MaxPooling1D(pool_size=2, strides=2))
        # second set of CONV => RELU => POOL
       # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
       # con d=0.1 daba 0.11 con loss=0.08
       # con d=0.2 daba 0.22 con loss=0.06
        model.add(Dense(64, activation='sigmoid', kernel_initializer='glorot_uniform')) # valor óptimo:64 @400k
       # model.add(Activation ('sigmoid'))
        #model.add(BatchNormalization())

        # output layer
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(1, activation = 'sigmoid'))
        # multi-GPU support
        #model = to_multi_gpu(model)
        #self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-4)
        # use SGD optimizer
        opt = Adamax(lr=self.learning_rate)
        #opt = SGD(lr=self.learning_rate, momentum=0.9)
        #paralell_model = multi_gpu_model(model, gpus=2)
        paralell_model = model
        paralell_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        #model.compile(loss="binary_crossentropy", optimizer="adamax", metrics=["accuracy"])
        #model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
        return paralell_model 

    ## Generate DCN  input matrix
    # data is an observation row
    
    def dcn_input(self, data):
        #obs_matrix = np.array([np.array([0.0] * self.num_features)]*len(data), dtype=object)
        obs_matrix = []
        # print ("len(data[0])=",len(data[0]))
        self.num_features =  len(data[0])//self.window_size
        obs = np.array([np.array([0.0] * self.window_size)] * self.num_features)
        # for each observation
        data_p = np.array(data)
        for i, ob in enumerate(data):
            # for each feature, add an array of window_size elements
            for j in range(0,self.num_features):
                #print("obs=",obs)
                #print("data_p=",data_p[i, j * self.window_size : (j+1) * self.window_size])
                obs[j] = data_p[i, j * self.window_size : (j+1) * self.window_size]
                #obs[j] = ob[0]
            obs_matrix.append(obs.copy())
        return np.array(obs_matrix)

    ## the action model is the same q-datagen generated dataset
    def load_action_model(self, signal):
        self.svr_rbf=load_model(self.model_prefix + str(signal)+'.dcn') 
        
    def decide_next_action(self, normalized_observation):
        # evaluate all models with the observation data window 
        self.action = []
        self.max_index = 0 
        action_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        vs = np.array(normalized_observation)
        # evaluate all models with the observation data window 
        self.action = []
        vs = np.array(normalized_observation)
        vs_r = np.reshape(vs, (1, -1))
        #print ("vs_r = ",vs_r)
        
        print ("normalized_observation = ",normalized_observation)
        input("Press Enter to continue...")
        action_list[0] = self.svr_rbf.predict(self.dcn_input(vs_r))
        self.action = action_list.copy()
        #print("action=",self.action)
        return self.action

# TODO LAST: BUSCAR ERROR DE DIFERENCIA ENTRE OUT DATASET DE Q-DATAGEN y PRIMERA OBSERVATION NORMALIZADA
    
    # add return of each feature as in the q-datagen genreated dataset,
    # applies power transform and feature selection mask to the observation,
    # to generate a row similar to the q-datagen generated, from the observation matrix of size: (num_features_prefs, window_size)
    # returns a roa with the pre-processed values
    def normalize_observation(self, observation, observation_prev):
        # observation is a list with size num_features of numpy.deque of size 30 (time window) 
        n_obs = []
        num_columns_o = len(observation)
        # print("num_columns_o = ", num_columns_o)
        # compose list from observation matrix similar to a row of the training set output from q-datagen (tick contiguous per feature)
        for i in range (0, num_columns_o):
            l_obs = list(observation[i])   
            for j in l_obs:
                n_obs.append(j)
        # append list of the returned values 
        for i in range (0, num_columns_o):
            l_obs = list(observation[i])   
            l_obs_prev = list(observation_prev[i])   
            # l_dif = l_obs - l_obs_prev
            l_dif = list( map(sub, l_obs, l_obs_prev) )
            for l in l_obs:
                n_obs.append(l)
        
        #apply pre-processing
        n_obs = self.pt.transform(np.array(n_obs).reshape(1,-1))
        n_o = n_obs[0].tolist()
        #print("n_o=",n_o)
        
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
        dire = 0.0
        tp = 1.0
        sl = 1.0
        vol  = 1.0
        # if there is no opened order
        if order_status == 0:
            # si el action[0] > 0, compra, sino vende
            if (self.raw_action[self.test_action] > 0.5):
                # opens buy order  
                dire = 1.0
            else:
                dire = -1.0
        
        # if there is an existing buy order
        if order_status == 1:
            # si action[0] == 0 cierra orden de buy 
            if (self.raw_action[self.test_action] <= 0.5):
                # closes buy order  
                dire = -1.0
               
        # if there is an existing sell order               
        if order_status == -1:
            # if action[0]>0, closes the sell order
            if (self.raw_action[self.test_action] > 0.5):
                # closes sell order  
                dire = 1.0
             
        # Create the action list output [tp, sl, vol, dir]
        act.append(tp)
        act.append(sl)
        act.append(vol)  
        act.append(dire)
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
        print("observation = ", observation)
        normalized_observation = agent.normalize_observation(observation, observation) 
        #print("normalized_observation = ", normalized_observation)
        score = 0.0
        step = 0
        order_status=0
        while 1:
            step += 1
            self.raw_action = self.decide_next_action(normalized_observation)
            action = self.transform_action(order_status)
            #print("raw_action=", self.raw_action, " action=", action)
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
        return info['balance'], avg_score     

    def show_results(self):
        test=0

# main function 
if __name__ == '__main__':
    agent = QAgent()
    scores = []
    balances = []
    for i in range(0, 1):
        print("Testing signal ",10+i)
        agent.test_action = i
        agent.num_f = agent.num_columns - agent.num_s
        agent.num_features = agent.num_f // agent.window_size
        agent.svr_rbf = agent.set_dcn_model()
        agent.load_action_model(10+i)
        balance,score = agent.evaluate()
        scores.append(score)
        balances.append(balance)
    print("Results:")
    for i in range(0, 1):
        print("Signal ", 10+i, " balance=",balances[i], " score=",scores[i])
        