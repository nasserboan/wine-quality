import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.compose import ColumnTransformer

def main():
    
    print(f'File found: intermediate_dataset.csv')
    print(f'Loading the dataset.')

    ## loading the dataframe
    df = pd.read_csv('./data/intermediate/intermediate_dataset.csv',sep=';')
    df.columns = [column_name.replace(' ','_') for column_name in df.columns]

    print('Initializing data preparation.')
    
    ## feature engineering
    df['rest_of_sulfur'] = df.total_sulfur_dioxide - df.free_sulfur_dioxide
    df['volatile_acid_types'] = pd.qcut(df.volatile_acidity,q=4,labels=['lowest','low','high','highest'])
    df['target'] = pd.qcut(df.quality,q=3,labels=[0,1,2])
    df.drop('quality',axis=1,inplace=True)
    
    ## definig X and Y
    X = df.drop('target',axis=1)
    y = df.target

    ## One Hot Encoder and Quantile Transformer
    ct = ColumnTransformer([('ohe',OneHotEncoder(sparse=False),['volatile_acid_types']),
                            ('quantile',QuantileTransformer(output_distribution='normal'),slice(0,12))],remainder='passthrough')
    
    ## defining the new X
    X = ct.fit_transform(X)

    ## creating the final dataframe
    final_dataframe = pd.DataFrame(X,columns=['volatile_acid_types_high', 'volatile_acid_types_highest', 'volatile_acid_types_low', 'volatile_acid_types_lowest',
                       'fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide',
                        'density','pH','sulphates','alcohol','rest_of_sulfur'])
    
    final_dataframe['target'] = y

    ## saving the final dataframe
    final_dataframe.to_csv('./data/final/final_dataframe.csv',sep=';',index=False)

    print(f'final_dataframe created with {final_dataframe.shape} shape.')

if __name__ == "__main__":
    main()