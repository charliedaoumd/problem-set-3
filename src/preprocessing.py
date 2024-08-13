'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Assuming the CSV files are named 'model_predictions.csv' and 'genres.csv'
    
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = genres_df['genre'].tolist()

    
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}

    # Process each row in the model predictions DataFrame
    for _, row in model_pred_df.iterrows():
        true_genres = eval(row['actual genres'])
        pred_genres = [row['predicted']]

        for genre in true_genres:
            if genre == '' or genre not in genre_true_counts:
                print(f"Unexpected or empty genre found: '{genre}'")
                continue  
            genre_true_counts[genre] += 1

        # Count true positives and false positives
        for genre in pred_genres:
            if genre == '' or genre not in genre_tp_counts:
                print(f"Unexpected or empty genre found: '{genre}'")
                continue  
            if genre in true_genres:
                genre_tp_counts[genre] += 1
            else:
                genre_fp_counts[genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
