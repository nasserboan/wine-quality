import pandas as pd
from glob import glob
from tqdm import tqdm

def main():
    
    list_of_file_paths = glob('data/raw/*.csv')
    print(f'Files found: {list_of_file_paths}.')

    list_of_dataframes = []

    for path in tqdm(list_of_file_paths,dynamic_ncols=True):
        list_of_dataframes.append(pd.read_csv(path,sep=';'))

    print(f'Dataframes created: {len(list_of_dataframes)}.')

    df = pd.concat(list_of_dataframes).reset_index(drop=True)
    df.to_csv('data/intermediate/intermediate_dataset.csv',sep=';',index=False)
    print(f'Intermediate dataframe created: {df.shape[0]} rows e {df.shape[1]} columns.')

if __name__ == "__main__":
    main()