

import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


#pacchetti statistici
import sklearn.metrics as metrics
from logproj.ml_dataCleaning import dummyColumns




# %%
def paretoDataframe(df, field):
    '''
    

    Parameters
    ----------
    df : TYPE pandas  dataframe
        DESCRIPTION. pandas dataframe with unsorted values
    field : TYPE string
        DESCRIPTION. column name to build the pareto

    Returns
    -------
    df : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe with cumulative and percentage columns

    '''
    df = df.dropna(subset=[field])
    df = df.sort_values(by = [field], ascending=False)
    df[f"{field}_PERC"] = df[field]/sum(df[field])
    df[f"{field}_CUM"] = df[f"{field}_PERC"].cumsum()
    
    return df

# %%

def paretoChart(df, barVariable, paretoVariable,titolo):

    #barVariable è la variabile di conteggio (istogramma) (itemcode)
    #paretoVariable è la variabile numerica (popularity)

    df = df.sort_values(by=paretoVariable,ascending=False)
    df["cumpercentage"] = df[paretoVariable].cumsum()/df[paretoVariable].sum()*100


    fig, ax = plt.subplots(figsize=(20,10))

    #asse principale
    ax.bar(np.linspace(0, 100, num=len(df)), df[paretoVariable], color="C0", width=0.5)
    #ax.bar(df[barVariable], df[paretoVariable], color="C0")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")

    #asse secondario
    ax2 = ax.twinx()
    ax2.plot(np.linspace(0, 100, num=len(df)), df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis="y", colors="C1")
    plt.title(titolo)
    plt.xlabel(str(barVariable))
    plt.ylabel('Percentage '+str(paretoVariable))
    plt.ylim([0 , 110])
    return fig




#costruisce scatterplot matrix
def scatterplotMatrix(X,dirResults):
    pal=sns.light_palette("orange", reverse=False)
    sns.set(style="ticks", color_codes=True)
    fig=sns.pairplot(X,diag_kind="kde", kind="reg",markers="+",palette=pal)
    fig.savefig(dirResults+'\\00_ScatterplotMatrix.png')
    return True

# %%
#costruisce la matrice di correlazione
def correlationMatrix(X,annotationCell=True):
    
    output_figures = {}
    d = X

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap=sns.light_palette("orange", reverse=False)

    # Draw the heatmap with the mask and correct aspect ratio
    figCorr=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, annot=annotationCell, square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=True, yticklabels=True)
    figure = figCorr.get_figure()
    output_figures['CorrelationMatrix'] = figure
    plt.close('all')
    return output_figures

# %%
#definisce un istogramma per ogni variabile
def histogramVariables(K,dirResults):
    for i in range(0,len(K.columns)):
        columnName=K.columns[i]
        plt.figure(figsize=(20,10))
        if(np.issubdtype(K.iloc[:,i].dtype, np.number)):
            plt.hist(K.iloc[:,i],color='orange')
            plt.title('Histogram var: '+str(columnName))
            plt.savefig(dirResults+'\\00_Hist_'+str(columnName)+'.png')
        else:
            sns.countplot(x=columnName, data=K,color='orange')
            plt.xticks(rotation=30)
            plt.title('Histogram var: '+str(columnName))
            plt.savefig(dirResults+'\\00_Hist_'+str(columnName)+'.png')
        plt.close('all')

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = rn.randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# randomly bootstrap values from a dataset X
def BootstrapValues(X, nboot):
    #bootstrap values;
    # X is the initial array
    # the funxtion returns an array X_bootstraped of the same length
    listBoots=[]
    for boot_i in range(nboot):
        boot_tr = np.random.choice(X, size=len(X), replace=True)
        listBoots.append(boot_tr)
    return listBoots

#stima MSE utilizzando bootstrap
def BootstrapLoop(nboot,model,X,y):


    X=dummyColumns(X) #rimuovo eventuali variabili categoriche



    scores_names = ["MSE"]
    scores_boot = np.zeros((nboot, len(scores_names)))
    #coefs_boot = np.zeros((nboot, X.shape[1]))
    orig_all = np.arange(X.shape[0])
    for boot_i in range(nboot):
        boot_tr = np.random.choice(orig_all, size=len(orig_all), replace=True)
        boot_te = np.setdiff1d(orig_all, boot_tr, assume_unique=False)
        Xtr, ytr = X.iloc[boot_tr, :], y[boot_tr]
        Xte, yte = X.iloc[boot_te, :], y[boot_te]
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte).ravel()
        scores_boot[boot_i, :] = metrics.mean_squared_error(yte, y_pred)
        #coefs_boot[boot_i, :] = model.coef_
    # Compute Mean, SE, CI
    scores_boot = pd.DataFrame(scores_boot, columns=scores_names)
    scores_stat = scores_boot.describe(percentiles=[.99, .95, .5, .1, .05, 0.01])
    #print("r-squared: Mean=%.2f, SE=%.2f, CI=(%.2f %.2f)" %\
    #      tuple(scores_stat.ix[["mean", "std", "5%", "95%"], "r2"]))
    #coefs_boot = pd.DataFrame(coefs_boot)
    #coefs_stat = coefs_boot.describe(percentiles=[.99, .95, .5, .1, .05, 0.01])
    #print("Coefficients distribution")
    #print(coefs_stat)
    return scores_stat
