#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:37:25 2018

@author: malvika
Loads feature matrix "feat_maxmul002.pkl". Uses model to classify TSG and OG.
Compares Old, New and all feature sets. Fit each feature set, estimates
paramters for different random seeds and uses mode of n_estimator and
corresponding parameters for model. Ranks the features.
"""
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
import numpy as np

# TODO: Set path
PATH = "/set/absolute/path/to/IdentifyTSGOG"
os.chdir(PATH)

# Folder to save results
folderPath = "/output/RandomForest/CV"
os.makedirs(PATH + folderPath, exist_ok=True)

# Random seed list to iterate over
# TODO: change the numRandIter to the number of random iterations requires
numRandIter = 10
RANDLIST = range(0, numRandIter)
N_EST = range(5, 31)
K_Folds = 5

# TODO: Load feature matrices
os.chdir(PATH + "/data/FeatureMat")
fname = "feat_keepv2_MM002.pkl"
with open(fname, 'rb') as f:
    features_cd = pickle.load(f)

# Drop rows where all entries are Nan
features_cd = features_cd[:-1].dropna(subset=list(features_cd.columns[0:-1]
                                     ))
# Split data
X, y = features_cd.iloc[:, 0:-1], features_cd.loc[:, "Label"]
# Get TSG and OG list
TSGlist = list(features_cd[features_cd["Label"] == "TSG"].index)
OGlist = list(features_cd[features_cd["Label"] == "OG"].index)
# Extract features only for TSG and OG and drop rows if all columns are Nan
X_tsgog = X.loc[TSGlist+OGlist].dropna(how='all')
y_tsgog = y[TSGlist+OGlist].dropna(how='all')
# Get data for unlabelled genes
Unlab = X.drop(TSGlist+OGlist)
Unlab = Unlab.dropna(how='all')

# Stratified k-fold
skf = StratifiedKFold(n_splits=K_Folds, random_state=3)
skf.get_n_splits(X_tsgog, y_tsgog)
for idx, (train_index, test_index) in enumerate(skf.split(X_tsgog, y_tsgog)):
    X_train, X_test = X_tsgog.iloc[train_index], X_tsgog.iloc[test_index]
    y_train, y_test = y_tsgog.iloc[train_index], y_tsgog.iloc[test_index]

    # Scaling
    sc = StandardScaler()
    sc.fit(X_train)
    # Save Standard scaling fit
    os.makedirs("{}{}/{}".format(PATH, folderPath, idx), exist_ok=True)
    os.chdir("{}{}/{}".format(PATH, folderPath, idx))
    scfname = "cosmicStdScale.pkl"
    with open(scfname, 'wb') as f:
        pickle.dump(sc, f)

    X_train = pd.DataFrame(sc.transform(X_train), index=X_train.index,
                               columns=X_train.columns)
    X_test = pd.DataFrame(sc.transform(X_test), index=X_test.index,
                              columns=X_test.columns)
    Unlab_std = pd.DataFrame(sc.transform(Unlab), index=Unlab.index,
                             columns=Unlab.columns)

    # get list of labels
    lab = list(set(y_tsgog))

    # features to be analysed
    newCols = ["Hifi/Lofi", "Hifi/benign", "Mifi/kb", "Nonstop/kb",
               "Inframe/kb", "Complex/kb", "Compound/benign",
               "Compound/kB", "Damaging/kb", "Damaging/benign",
               "Damaging/Lofi", "HiMisFreq", "FSEntr", "HiFSFreq",
               "SplicEntr", "HiSplicFreq", "NonsenseEntr",
               "HiNonsenseFreq", "TotMifi"]
    oldCols = [x for x in list(X_tsgog.columns) if x not in newCols]
    cols = [list(X_tsgog.columns), newCols, oldCols]

    files = ["allFeat", "newFeat", "oldFeat"]
    # Parameters for grid search
    param_rfc = {'max_features': ['sqrt', 'log2'], 'max_depth': [2, 3, 4],
                 'criterion': ['gini', 'entropy'], 'n_estimators': N_EST}
    os.chdir(PATH + folderPath)
    all_stats_cols = ["Features", "Num_feats", "n_estimator", "random_seed",
                      "Max_features", "Max_depth", "Criterion", "Accuracy",
                      ("F1_" + str(lab[0])), ("F1_" + str(lab[1])),
                      ("Precision_" + str(lab[0])),
                      ("Precision_" + str(lab[1])), ("Recall_" + str(lab[0])),
                      ("Recall_" + str(lab[1]))]
    all_stats = pd.DataFrame(columns=all_stats_cols)

    # Analyse for each feature set for different random seed
    for collist, featUsed in zip(cols, files):
        Acc_var_cols = ["random_seed", "n_estimator", "max_features", "max_depth",
                        "criterion", "Accuracy"]
        Acc_var = pd.DataFrame(columns=Acc_var_cols)
        for rand_seed in RANDLIST:
            # Find best features for first random seed using grid search
            rfc = RandomForestClassifier(random_state=rand_seed, n_jobs=-1)
            gs = GridSearchCV(estimator=rfc, param_grid=param_rfc,
                              scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
            gs = gs.fit(X_train[collist], y_train)
            # predict labels for training set
            tr_pred = gs.predict(X_train[collist])
            # calculate metrics
            acc_tr = accuracy_score(y_train, tr_pred)
            f1_tr = f1_score(y_train, tr_pred, average=None, labels=lab)
            p_tr = precision_score(y_train, tr_pred, average=None, labels=lab)
            r_tr = recall_score(y_train, tr_pred, average=None, labels=lab)

            # Save best params estimated for the given random seed
            nest = gs.best_params_['n_estimators']
            maxf = gs.best_params_['max_features']
            maxd = gs.best_params_['max_depth']
            crit = gs.best_params_['criterion']
            accv = gs.best_score_

            # Save stats
            all_stats.loc[len(all_stats)] = [featUsed, len(collist),
                                             nest, rand_seed, maxf,
                                             maxd, crit, acc_tr, f1_tr[0],
                                             f1_tr[1], p_tr[0], p_tr[1],
                                             r_tr[0], r_tr[1]]
            Acc_var.loc[len(Acc_var)] = [rand_seed, nest, maxf, maxd, crit,
                                         accv]
        # Get best n_estimator using min variance and other parameters
        max_counts = [list(Acc_var["n_estimator"]).count(x) for x in list(set(Acc_var["n_estimator"]))]
        max_est = [x for x in list(set(Acc_var["n_estimator"])) if list(Acc_var["n_estimator"]).count(x) == max(max_counts)]
        if len(max_est) == 1:
            best_nEst = max_est[0]
            acc = max(Acc_var[Acc_var["n_estimator"] == best_nEst]["Accuracy"])
        else:
            acc = 0
            for est in max_est:
                if acc < max(Acc_var[Acc_var["n_estimator"] == est]["Accuracy"]):
                    acc = max(Acc_var[Acc_var["n_estimator"] == est]["Accuracy"])
                    best_nEst = est
        maxf = list(Acc_var[(Acc_var["n_estimator"] == best_nEst) &
                            (Acc_var["Accuracy"] == acc)]["max_features"])[0]
        maxd = list(Acc_var[(Acc_var["n_estimator"] == best_nEst) &
                            (Acc_var["Accuracy"] == acc)]["max_depth"])[0]
        crit = list(Acc_var[(Acc_var["n_estimator"] == best_nEst) &
                            (Acc_var["Accuracy"] == acc)]["criterion"])[0]
        # Print to file for given feature matrix
        os.chdir("{}{}/{}".format(PATH, folderPath, idx))
        filename = "AccVar_MM{}_{}.txt".format(fname[11:14], featUsed)
        Acc_var.to_csv(filename, sep="\t", header=True, index=False)

        # Calculate feature ranking for the best n_estimator and features
        # with set random seed
        # set filename for output file write into file
        filename = "Best_{}MM{}.txt".format(featUsed, "002")
        # Classification
        rfc = RandomForestClassifier(random_state=3, n_jobs=-1,
                                     n_estimators=best_nEst,
                                     max_features=maxf,
                                     max_depth=maxd,
                                     criterion=crit)
        rfc.fit(X_train[collist], y_train)

        # Save model
        os.chdir("{}{}/{}".format(PATH, folderPath, idx))
        featFName = "randIterRFv5Model_{}.pkl".format(featUsed)
        with open(featFName, 'wb') as f:
            pickle.dump(rfc, f)

        # predict labels for training set and test set
        y_pred = rfc.predict(X_test[collist])
        tr_pred = rfc.predict(X_train[collist])

        # Print to file gene labels and prediction
        gene_pred = pd.DataFrame(index=list(X_train.index) +
                                 list(X_test.index))
        gene_pred["Label"] = (list(y_train) + list(y_test))
        gene_pred["Predictions"] = (list(tr_pred) + list(y_pred))
        gene_pred["Data"] = (["train"] * len(y_train)) + (["test"] * len(y_test))
        os.chdir("{}{}/{}".format(PATH, folderPath, idx))
        fname = "TrainingTest_predictions"
        gene_pred.to_csv(fname, sep="\t", index_label="Gene")
        # calculate metrics and print to o/p file
        f1_tr = f1_score(y_train, tr_pred, average=None, labels=lab)
        f1_ts = f1_score(y_test, y_pred, average=None, labels=lab)
        p_tr = precision_score(y_train, tr_pred, average=None, labels=lab)
        p_ts = precision_score(y_test, y_pred, average=None, labels=lab)
        r_tr = recall_score(y_train, tr_pred, average=None, labels=lab)
        r_ts = recall_score(y_test, y_pred, average=None, labels=lab)
        a_tr = accuracy_score(y_train, tr_pred)
        a_ts = accuracy_score(y_test, y_pred)
        # Feature ranking
        importances = rfc.feature_importances_
        indices = np.argsort(importances)[::-1]
        with open("Rank_"+filename, 'w') as f:
            f.write("# {}\n".format("\t".join(collist)))
            f.write("# Number of features: {:02d}\n".format(len(collist)))
            f.write("# Shape of training set : " +
                    "{}\n".format(X_train[collist].shape))
            f.write("# Shape of test set : " +
                    "{}\n".format(X_test[collist].shape))
            f.write("# Best features:\n")
            f.write("#\tn_estimator:{}\n".format(best_nEst))
            f.write("#\tmax_features:{}\n".format(maxf))
            f.write("#\tmax_depth:{}\n".format(maxd))
            f.write("#\tcriterion:{}\n\n".format(crit))
            f.write("\tTraining\t\tTest\t\n")
            f.write("\t{}\t{}\t{}\t".format(lab[0], lab[1], lab[0]) +
                    "{}\n".format(lab[1]))
            f.write("Accuracy\t{:1.4f}\t\t{:1.4f}\t\n".format(a_tr,
                                                              a_ts))
            f.write("F1 score\t{:1.4f}\t{:1.4f}\t{:1.4f}\t{:1.4f}\n".format(
                    f1_tr[0], f1_tr[1], f1_ts[0], f1_ts[1]))
            f.write("Precision\t{:1.4f}\t{:1.4f}\t{:1.4f}\t{:1.4f}\n".format(p_tr[0], p_tr[1], p_ts[0], p_ts[1]))
            f.write("Recall\t{:1.4f}\t{:1.4f}\t{:1.4f}\t{:1.4f}\n".format(
                    r_tr[0], r_tr[1], r_ts[0], r_ts[1]))
            f.write("\n\nFeature Ranking\n")
            for rank in range(X_train[collist].shape[1]):
                f.write("{:02d}\t{}\t{}\t{:1.4f}\n".format(rank + 1,
                        indices[rank],
                        X_train[collist].columns[indices[rank]],
                        importances[indices[rank]]))

        # Prediction of novel driver genes
        novel_pred = rfc.predict(Unlab_std[collist])
        novel_prob = pd.DataFrame(rfc.predict_proba(Unlab_std[collist]),
                                  index=Unlab_std.index,
                                  columns=rfc.classes_).sort_values(by=["TSG"],
                                                      ascending=False)
        novel_logp = pd.DataFrame(rfc.predict_log_proba(Unlab_std[collist]),
                                  index=Unlab_std.index,
                                  columns=rfc.classes_).sort_values(by=["TSG"],
                                                      ascending=False)
        # Print to file
        os.chdir("{}{}/{}".format(PATH, folderPath, idx))
        filename = "prob_{}.txt".format(featUsed)
        novel_prob.to_csv(filename, sep="\t", header=True, index=True)
        filename = "logp_{}.txt".format(featUsed)
        novel_logp.to_csv(filename, sep="\t", header=True, index=True)

    # Print to file for given feature matrix
    os.chdir("{}{}/{}".format(PATH, folderPath, idx))
    filename = "allStats_MM{}.txt".format(fname[11:14])
    all_stats.to_csv(filename, sep="\t", header=True, index=False)







