import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, CuDNNLSTM, Dropout
from LayerNormalization import LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.backend import clear_session

class Force_regressor:
    def __init__(self, num_iemg, num_forces, iemg_force_correspondance, LSTM_size, dropout, learning_rate, reg_const):
        self.num_iemg = num_iemg
        self.num_forces = num_forces
        
        self.iemg_force_correspondance = iemg_force_correspondance
        
        self.LSTM_size = LSTM_size
        self.dropout = dropout
        
        self.learning_rate = learning_rate
        self.reg_const = reg_const
        
        self._build()

class One_to_one(Force_regressor):
    def _build(self):
        self.models = []
        for iemg_channel in range(self.num_iemg):
            model = Sequential()
            model.add(Dropout(self.dropout))
            model.add(Conv1D(filters=64,
                         kernel_size=21,
                         strides=1,
                         padding='same',
                         activation='relu',
                         kernel_regularizer=l2(self.reg_const)))
            model.add(LayerNormalization(axis=-1,
                                     center=False,
                                     scale=False))
            model.add(Dropout(self.dropout))
            
            model.add(CuDNNLSTM(self.LSTM_size,
                                return_sequences=False,#(i+1)<len(self.LSTM_sizes),
                                kernel_regularizer=l2(self.reg_const),
                                recurrent_regularizer=l2(self.reg_const)))
            model.add(LayerNormalization(axis=-1,
                                 center=False,
                                 scale=False))
            model.add(Dropout(self.dropout))
            
            
            model.add(Dense(32,
                            activation='relu',
                            kernel_regularizer=l2(self.reg_const)))
            model.add(LayerNormalization(axis=-1,
                                     center=False,
                                     scale=False))
            model.add(Dropout(self.dropout))
            
            model.add(Dense(1,
                            activation='linear'))
            
            opt = Adam(lr=self.learning_rate)
            model.compile(loss='mean_squared_error',
                          optimizer=opt,
                          metrics=[])
            
            self.models.append(model)
            
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, val_patience):
        for iemg_channel, model in enumerate(self.models):
            force_index = self.iemg_force_correspondance[iemg_channel]
            
            X_train_o2o = X_train[:, :, iemg_channel]
            X_train_o2o = np.expand_dims(X_train_o2o, axis=-1)
            y_train_o2o = y_train[:, force_index]
            
            X_val_o2o = X_val[:, :, iemg_channel]
            X_val_o2o = np.expand_dims(X_val_o2o, axis=-1)
            y_val_o2o = y_val[:, force_index]
                
            history = model.fit(X_train_o2o,
                                y_train_o2o,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data = (X_val_o2o, y_val_o2o),
                                shuffle=True,
                                verbose=0,
                                callbacks=[EarlyStopping(patience=val_patience,
                                                         mode='min',
                                                         verbose=0)])
    
    def predict(self, X):
        y_pred_o2o = np.zeros((X.shape[0], self.num_iemg))
        for iemg_channel, model in enumerate(self.models):
            X_test_o2o = X[:, :, iemg_channel]
            X_test_o2o = np.expand_dims(X_test_o2o, axis=-1)
            
            y_out_o2o = np.squeeze(model.predict(X_test_o2o, batch_size=100))
            
            y_pred_o2o[:, iemg_channel] = y_out_o2o
        return y_pred_o2o
    
    
class All_to_one(Force_regressor):
    def _build(self):
        self.models = []
        for force_index in range(self.num_forces):
            model = Sequential()
            model.add(Dropout(self.dropout))
            model.add(Conv1D(filters=64,
                         kernel_size=21,
                         strides=1,
                         padding='same',
                         activation='relu',
                         kernel_regularizer=l2(1e-6)))
            model.add(LayerNormalization(axis=-1,
                                     center=False,
                                     scale=False))
            model.add(Dropout(self.dropout))
            
            model.add(CuDNNLSTM(self.LSTM_size,
                                return_sequences=False,#(i+1)<len(self.LSTM_sizes),
                                kernel_regularizer=l2(self.reg_const),
                                recurrent_regularizer=l2(self.reg_const)))
            model.add(LayerNormalization(axis=-1,
                                 center=False,
                                 scale=False))
            model.add(Dropout(self.dropout))
            
            model.add(Dense(32,
                            activation='relu',
                            kernel_regularizer=l2(self.reg_const)))
            model.add(Dropout(self.dropout))
            model.add(LayerNormalization(axis=-1,
                                     center=False,
                                     scale=False))
            
            model.add(Dense(1,
                            activation='linear'))
            
            opt = Adam(lr=self.learning_rate)
            model.compile(loss='mean_squared_error',
                          optimizer=opt,
                          metrics=[])
            
            self.models.append(model)
        
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, val_patience):
        for force_index, model in enumerate(self.models):
            y_train_a2o = y_train[:, force_index]
            history = model.fit(X_train,
                                y_train_a2o,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data = (X_val, y_val[:, force_index]),
                                shuffle=True,
                                verbose=0,
                                callbacks=[EarlyStopping(patience=val_patience,
                                                 mode='min',
                                                 verbose=0)])
    def predict(self, X):
        y_out_a2o = np.zeros((np.shape(X)[0], self.num_forces))
        for force_index, model in enumerate(self.models):
            y_out_a2o[:, force_index] = np.squeeze(model.predict(X, batch_size=100))
        
        y_pred_a2o = np.zeros((np.shape(X)[0], self.num_iemg))
        for iemg_channel in range(self.num_iemg):
            y_pred_a2o[:, iemg_channel] = y_out_a2o[:, self.iemg_force_correspondance[iemg_channel]]
        
        return y_pred_a2o
    
class All_to_all(Force_regressor):
    def _build(self):
        self.model = Sequential()
        self.model.add(Dropout(self.dropout))
        self.model.add(Conv1D(filters=64,
                              kernel_size=21,
                              strides=1,
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(self.reg_const)))
        self.model.add(LayerNormalization(axis=-1,
                                          center=False,
                                          scale=False))
        self.model.add(Dropout(self.dropout))
        
        self.model.add(CuDNNLSTM(self.LSTM_size,
                                 return_sequences=False,#(i+1)<len(self.LSTM_sizes),
                                 kernel_regularizer=l2(self.reg_const),
                                 recurrent_regularizer=l2(self.reg_const)))
        self.model.add(LayerNormalization(axis=-1,
                                          center=False,
                                          scale=False))
        self.model.add(Dropout(self.dropout))
        
        self.model.add(Dense(32,
                             activation='relu',
                             kernel_regularizer=l2(self.reg_const)))
        self.model.add(Dropout(self.dropout))
        self.model.add(LayerNormalization(axis=-1,
                                          center=False,
                                          scale=False))
        
        self.model.add(Dense(self.num_forces,
                        activation='linear'))
        
        opt = Adam(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=[])
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, val_patience):
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data = (X_val, y_val),
                                 shuffle=True,
                                 verbose=0,
                                 callbacks=[EarlyStopping(patience=val_patience,
                                                          mode='min',
                                                          verbose=0)])
    
    def predict(self, X):
        y_pred_a2a = np.zeros((X.shape[0], self.num_iemg))
        
        y_out_a2a = self.model.predict(X, batch_size=100)
        
        for iemg_channel in range(self.num_iemg):
            y_pred_a2a[:, iemg_channel] = y_out_a2a[:, self.iemg_force_correspondance[iemg_channel]]
        
        return y_pred_a2a