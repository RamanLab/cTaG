#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:23:35 2019

@author: malvika
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
from scipy import stats

PATH = "/set/absolute/path/to/IdentifyTSGOG"
os.chdir(PATH)

# ----------- Load data and add or rename required columns ----------- #
# MutSigCV predictions load
os.chdir(PATH + "/output/MutSigCVComparisson/all_set")
fname = "all_output.sig_genes.xlsx"
data_mutsig = pd.read_excel(fname, sheet_name="all_output.sig_genes")
# rename columns
data_mutsig.rename(columns={"gene": "Gene", 'u x 10^6': "u"}, inplace=True)
# add columns
data_mutsig["Consensus"] = [5] * len(data_mutsig)
data_mutsig["Label"] = ["Driver"] * len(data_mutsig)
data_mutsig["Source"] = ["MutSigCV"] * len(data_mutsig)

# Our feature predictions load
os.chdir(PATH + "/output/RandomForest/CV")
fname = "CVpredictions.xlsx"
data_novel = pd.read_excel(fname, sheet_name="CVpredictions")
# rename columns
data_novel.rename(columns={"Gene name": "Gene"}, inplace=True)
# add columns
data_novel["Source"] = ["Our predictions"] * len(data_novel)

# Training data
fname = "Training_predictions.xlsx"
data_train = pd.read_excel(fname, sheet_name="ConsolidatedCV_results")
# rename columns
data_train.rename(columns={"Gene name": "Gene"}, inplace=True)
# add columns
data_train["Source"] = ["Training"] * len(data_train)

# ----------------- Data for plotting ---------------- ##
# Data requires following columns
# "Source" can be {MutSigCV, Training, Our predictions"}
cols = ["Gene", "u", "Consensus", "Label", "Source"]
data_all = pd.DataFrame(columns=cols)

# Filter data
# Our predictions
data_temp = data_novel[data_novel["Consensus"] >= 5]
data_temp = pd.merge(data_temp, data_mutsig[['Gene', 'u']], how="left",
                     on=["Gene"])
data_temp = data_temp[cols]
data_temp = data_temp.dropna()
data_all = pd.concat([data_all, data_temp])

# Training predictions
data_temp = data_train[data_train["Consensus"] >= 0]
data_temp = pd.merge(data_temp, data_mutsig[['Gene', 'u']], how="left",
                     on=["Gene"])
data_temp = data_temp[cols]
data_temp = data_temp.dropna()
data_all = pd.concat([data_all, data_temp])

# MutSigCV predictions
data_temp = data_mutsig[(data_mutsig["p"] <= 0.005) &
                        (data_mutsig["q"] <= 0.01)]
data_temp = data_temp[cols]
data_temp = data_temp.dropna()
data_all = pd.concat([data_all, data_temp])

data_plot = data_all

# ------------- Plot fraction of genes for varying mutation rates ------#
num_genes = 60
# Consensus defines how our model predictions are filtered
consensus = 5
# MutSigCV predictions ranked and filtered
cols = ["Gene", "u", "Consensus", "Label", "Source"]
data_plot = data_mutsig.sort_values(by=["q", "p"]).iloc[:num_genes, :]
data_plot = data_plot[cols]
# Data concatenated
data_plot = pd.concat([data_plot,
                       data_all[((data_all["Source"] == "Training") |
                                (data_all["Consensus"] >= consensus)) &
                                (data_all["Source"] != "MutSigCV")]])
data_plot["log_u"] = round(np.log(data_plot["u"]), 2)
# Populate fractions to be plotted
temp = []
for source, u in zip(data_plot["Source"], data_plot["log_u"]):
    tot = len(data_plot[(data_plot["Source"] == source)]["log_u"])
    fraction = len(data_plot[(data_plot["Source"] == source) &
                             (data_plot["log_u"] <= u)]["log_u"]) / tot
    temp.append(fraction)
data_plot["Fraction"] = temp

# plot
fig = pyplot.figure(figsize=(11, 8))
ax1 = fig.add_subplot(111)
for source, c, m in zip(list(set(data_plot["Source"])), ['c', 'b', 'r'],
                        ['x', 'o', 's']):
    temp = data_plot[data_plot["Source"] == source]
    ax1.scatter(temp["log_u"], temp["Fraction"], label=source,
         color=c, marker=m)
pyplot.xlabel('Log mutation rate', fontname='Calibri', fontsize =16)
pyplot.ylabel('Fraction of genes predicted below mutation rate',
              fontname='Calibri', fontsize =16)
handles, labels = ax1.get_legend_handles_labels()
pyplot.xticks(fontsize=12)
pyplot.xticks(fontsize=12)
lgd = ax1.legend(handles, labels, loc='upper left',
                 bbox_to_anchor=(0.01, 1), fontsize =16)
ax1.grid('on')
os.chdir(PATH + "/output/MutSigCVComparisson/all_set")
pyplot.savefig('log_fraction_scatter_cv60.jpg')
pyplot.close()

# --------------- Kolmogorov-Smirnov statistic -------------- ##
# Calculate statistic and pvalue between MutSigCV and (Our method + Training)
KS_stat, pval = stats.ks_2samp(data_plot[data_plot["Source"] ==
                                         "MutSigCV"]["u"],
                               data_plot[data_plot["Source"] !=
                                         "MutSigCV"]['u'])
print("KS statistic = {:0.3f}\np-value = {:0.3f}".format(KS_stat, pval))

# Compare training dirstribution to MutSigCv and our prediciton distributions
for source in ["MutSigCV", "Our predictions"]:
    KS_stat, pval = stats.ks_2samp(data_plot[data_plot["Source"] ==
                                             source]["u"],
                                   data_plot[data_plot["Source"] ==
                                             "Training"]['u'])
    print("Kolmogorov statistic for {} = {:0.3f}".format(source, KS_stat))
    print("P-value for {} = {:0.3f}".format(source, pval))


