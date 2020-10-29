#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:05:14 2019

@author: malvika
"""

import pandas as pd
import os
import numpy as np


def findLabel(row):
    """
    Find label for the given data based on max probaility. Labels are assigned
    based on max probability.
    """
    if row["Max score"] == row["TSG"]:
        label = "TSG"
    elif row["Max score"] == row["OG"]:
        label = "OG"
    return label


def findCumLabel(row, cv=5):
    """
    Find label for the given data based on multiple cv models. Labels are
    assigned based on mode.
    """
    labels = [row["Label_{}".format(x)] for x in range(cv) if
              row["top_stat_{}".format(x)] == 1]
    countTSG = labels.count("TSG")
    countOG = labels.count("OG")
    if countTSG > countOG:
        return "TSG"
    elif countOG > countTSG:
        return "OG"
    else:
        return "Unlabelled"


def findCumProb(row, cv=5):
    """
    Find cumulative probability for the given data based on multiple cv
    models. p = 1 - ()
    """
    notLab_p = []
    allLab = [row["Label_{}".format(x)] for x in range(cv)]
    allProb = [row["Max score_{}".format(x)] for x in range(cv)]
    countTSG = allLab.count("TSG")
    countOG = allLab.count("OG")
    if countTSG > countOG:
        Label = "TSG"
    elif countOG > countTSG:
        Label = "OG"
    for lab, prob in zip(allLab, allProb):
        if lab == Label:
            notLab_p.append(1 - float(prob))
        else:
            notLab_p.append(float(prob))
#    if row["Label"] == "TSG":
#        
#    elif row["Label"] == "OG":
#        
#    notLab_p = [1 - float(row["Max score_{}".format(x)]) for x in range(cv)]
    return 1 - np.prod(notLab_p)


def findTop(row, treshold):
    """
    """
    if row["Max score"] >= treshold:
        return 1
    else:
        return 0


# TODO: Set path
PATH = "/set/absolute/path/to/IdentifyTSGOG"
os.chdir(PATH)

# Folder to save results
folderPath = "/output/RandomForest/CV"
os.makedirs(PATH + folderPath, exist_ok=True)

# initalise variables
cv = 5
percentile = 5
prob_df = pd.DataFrame()
for folder in range(cv):
    os.chdir("{}{}/{}".format(PATH, folderPath, folder))
    fname = "prob_allFeat.txt"
    temp = pd.read_csv(fname, sep="\t", index_col=0)
    # Find labels based on scores
    temp["Max score"] = (temp.apply(max, axis=1))
    temp["Label"] = (temp.apply(findLabel, axis=1))
    rank = int(len(temp) * percentile / 100)
    treshold = sorted(temp["Max score"], reverse=True)[rank]
    temp["top_stat"] = (temp.apply(findTop, axis=1, treshold=treshold))
    temp = temp.drop(["OG", "TSG"], axis=1)
    prob_df = prob_df.join(temp.add_suffix("_{}".format(folder)), how="outer")

prob_df["Label"] = prob_df.apply(findCumLabel, axis=1)
prob_df["Probability"] = prob_df.apply(findCumProb, axis=1)
os.chdir("{}{}".format(PATH, folderPath))
fname = "CVpredictions.txt"
prob_df.to_csv(fname, sep="\t", index_label="Gene name")
