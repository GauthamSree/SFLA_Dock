#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
import pickle


def tm_score(score, execdir):
    """Function which predict the tm_score using machine learning algorithm
    Input : Pandas DataFrame containing Scores 
    Output:
            A .csv file with contains all the score regarding your pdbs and
            the tm_score_prediction
    """
    #Load the machine learning algorithm thanks to picke
    final_model = pickle.load(open(execdir+"/utils/learningModel.pickle", "rb"))
    #prediction of the tm_score
    predictions = final_model.predict(score)
    #Adding interpretations:
    interpretation = []
    for tm in predictions:
        if float(tm) >= 0.8:
            interpretation.append('excellent')
        elif float(tm) < 0.8 and float(tm) >= 0.6:
            interpretation.append('moyen')
        elif float(tm) < 0.6 and float(tm) >= 0.4:
            interpretation.append('passable')
        elif float(tm) < 0.4:
            interpretation.append('mauvais')
    #Adding predictions to the initial Pandas DataFrame
    final = score.assign(tm_score_prediction=predictions)
    final = final.assign(interpretations=interpretation)
    #Sorting pdb regarding the predict tm_score
    final = final.sort_values(by=['tm_score_prediction'], ascending = False)
    return final