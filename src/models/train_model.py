import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

from sklearn.metrics import classification_report
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import regularizers

import sys

class myMLPmodel():
    def __init__(self,NUM_OF_EPOCHS,test_size):
        self.NUM_OF_EPOCHS = NUM_OF_EPOCHS
        self.test_size = test_size

    def _load_the_data(self):
        
        '''
        load the data from 'data/final' and define X and y
        '''

        self.df = pd.read_csv('./data/final/final_dataframe.csv',sep=';')
        self.X = self.df.drop('target',axis=1).values
        self.y = self.df.target.values

        return self.X , self.y

    def _split_the_data(self):
        
        '''
        split the data in test and train in a stratified way and trasnform y_train and y_test in categorical features for keras models.
        '''

        self.x_train , self.x_test , self.y_train , self.y_test = train_test_split(self._load_the_data()[0],self._load_the_data()[1],test_size=float(self.test_size),random_state=42)
        self.y_train_cat = to_categorical(self.y_train,num_classes=3)
        self.y_test_cat = to_categorical(self.y_test,num_classes=3)

        return self.x_train,self.x_test,self.y_train_cat,self.y_test_cat,self.y_train,self.y_test

    def train(self):
        
        '''
        define, compile, train and save the MLP Model.
        '''

        self.x_train,self.x_test,self.y_train_cat,self.y_test_cat, _, _= self._split_the_data()

        self.model = Sequential()
        self.model.add(Dense(8,activation='relu',input_shape=(16,),kernel_regularizer=regularizers.l2(0.1)))
        self.model.add(Dense(8,activation='relu',input_shape=(8,)))
        self.model.add(Dense(8,activation='relu',input_shape=(8,),kernel_regularizer=regularizers.l2(0.1)))
        self.model.add(Dense(3,activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])      
        self.fitted = self.model.fit(self.x_train,self.y_train_cat,epochs=int(self.NUM_OF_EPOCHS),batch_size=32,verbose=1,validation_data=(self.x_test,self.y_test_cat))

        print('Model fitted.')

        time = datetime.now().strftime("%d%m%Y-%H%M%S")

        with open(f'./models/model-{time}.json','w') as file:
            file.write(self.model.to_json())

        print(f'Model saved as : model-{time}.json')

        self.model.save_weights(f'./models/model-{time}-weights.hd5')

        print(f'Weights saved as : model-{time}-weights.hd5')

    def generate_loss_curves(self):
        
        '''
        plot the loss and accuracy curves
        '''

        _, axs = plt.subplots(1,2,figsize=(15,9))

        axs[0].plot(list(range(1,self.NUM_OF_EPOCHS+1)),self.fitted.history['loss'],label='train_loss')
        axs[0].plot(list(range(1,self.NUM_OF_EPOCHS+1)),self.fitted.history['val_loss'],label='val_loss')
        axs[0].legend()

        axs[1].plot(list(range(1,self.NUM_OF_EPOCHS+1)),self.fitted.history['accuracy'],label='train_accuracy')
        axs[1].plot(list(range(1,self.NUM_OF_EPOCHS+1)),self.fitted.history['val_accuracy'],label='val_accuracy')
        axs[1].legend()

        plt.show()

    def model_plot(self):
        
        '''
        generate the model plot
        '''
        
        return plot_model(self.model,to_file='./reports/model.png',show_shapes=True,show_layer_names=True)

    def confusion_matrix(self):
        
        '''
        generate the confusion matrix for the model
        '''
        
        return classification_report(self.y_test,self.model.predict_classes(self.x_test,verbose=0),target_names=['low_quality_wine','medium_quality_wine','high_quality_wine'])

    def predict(self,data):

        return self.model.predict([[data]]),self.model.predict_classes([[data]])

# if __name__ == "__main__":
#     ls = sys.argv
#     model = myMLPmodel(NUM_OF_EPOCHS = ls[1], test_size= ls[2])
#     model.train()
#     print(model.confusion_matrix())


