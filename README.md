cTaG
============================
cTaG (<ins>c</ins>lassify <ins>T</ins>SG <ins>a</ins>nd O<ins>G</ins>) is a tool used to identify tumour suppressor genes (TSGs) and oncogenes (OGs) using somatic mutation data.

## Table of Contents

- [Description](#description)
- [Overview of cTaG](#overview-of-ctag)
- [Data](#data)
- [Folder structure](#folder-structure)
- [Links](#links)

## Description

The model is built using somatic mutation data from COSMIC (v79) from differnt cancer types. We use ratio-metric and entropy features to classify genes as TSG or OG. Our model, unlike methods for identifying driver genes that use background mutation rate, is not biased towards genes with high frequency of mutations. The pan-cancer model is generated using random forest. Cancer Gene Census (CGC) is used to label genes as TSG or OG. The pan-cancer model is trained on these genes. We overcome overfitting due to the small set of genes to train by estimating stable hyper-parameter set using multiple random iterations. We also employ the pan-cancer model to identify tissue specific driver genes.

## Overview of cTaG
Random forest is used to build multiple classifiers for each fold (Block A) and overfitting is avoided by identification of stable hyper-parameters (Block B). Consensus across models is used for predicting new TSGs and OGs. Somatic mutation data used to train models is downloaded from COSMIC. For details see section [Data](#data). 

![fig3methods](https://user-images.githubusercontent.com/17045221/97172918-bf3e9080-17b5-11eb-8706-13f96a4c4fa2.jpg)

## Data
The data used for this analysis was downloaded from COSMIC (v9). Somatic coding mutation data from COSMIC was considered for samples less than 2000 mutations. The data was filtered. Mutation type annotations used were given by COSMIC. The genes were labelled as TSG or OG based on labels given by Cancer Gene Census (CGC), and the rest were marked Unlabelled. The list of TSGs and OGs used for training can be found [here](https://github.com/RamanLab/IdentifyTSGOG/tree/master/data/cgc_genes).The mutation data was used to generate feature matrix were the rows and columns correspond to genes/transcripts and 37 features respectively. The processed feature matrix can be found [here](https://github.com/RamanLab/IdentifyTSGOG/blob/master/data/FeatureMat/FeatureMat.tsv). The genes not identified as TSG or OG were used to make predictions of new driver genes.

The mutation data from COSMIC was also used for tissue-specific analysis. The samples were divided based on their primary tissue for origin. Tissues with greater than 1000 samples were used. These tissues are breast, central_nervous_system, cervix, endometrium, haematopoietic_and_lymphoid_tissue, kidney, large_intestine, liver, pancreas and prostate. Feature matrix for each tissue was generated and can be found [here](https://github.com/RamanLab/IdentifyTSGOG/tree/master/data/FeatureMat/tissues).

## Folder structure
The top-level directories contain code, data and output folders. 

### The directory layout

    .
    ├── code                        # All the code for analysis
    ├── data
    │   ├── FeatureMa               # pre-processed data, feature matrix
    │   │   └── tissues             # Tissue-specific feature matrices
    │   │   Functional_Analysis     # List of genes used for function analysis
    │   └── cgc_genes               # List of genes used for building model
    ├── output
    │   ├── evalRandIter            # Results for verying number of random iterations
    │   ├── FunctionalAnalysis      # Results for functional analysis
    │   ├── MutSigCVComparisson     # Results of MutSigCV and comparisson of mutation rates 
    │   ├── RandomForest            # Model for each CV and consensus of results
    │   └── TissueSpecificAnalyisis # Tissue specific predictions and consensus of results
    └── README.md

The code folder containes all the files used for building the feature matrix, building the models and and the tissue-specific analysis.

    .
    ├── ...
    ├── code                                # All the code for analysis
    │   ├── consolidateModelPredictions.py  # Consolidate results for pan-cancer analysis
    │   ├── evaluate_randiter.py            # Evaluates performance different number of random iterations
    │   ├── identifyTSGOG.py                # Module for building feature matrix
    │   ├── plotMutRate.py                  # Plots mutation rate for genes redicted by cTaG and MutSigCV
    │   ├── TSG_OG_Feat_final.py            # Loads feature matrices and build models
    │   └── TSG_OG_tissueAnalysis.py        # Creates tissue-specific feature matrices and makes predictions
    └── ...

## Links
[BioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.01.17.910075v1)
