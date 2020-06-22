# -*- coding: utf-8 -*-

imprt matplotlib.pyplot as plt
import seabors as sn


# %%
def buildLearningTablePickList(D_movements):
    D_learning = D_movements.groupby(['NODECODE','PICKINGLIST','INOUT']).agg(
        { 'QUANTITY':['sum','size'],
           'TIMESTAMP_IN':['max','min'],
           'RACK':['max','min'],
           'BAY':['max','min'],
           'LEVEL':['max','min'],
           'LOCCODEX':['max','min'],
           'LOCCODEY':['max','min'],
    
        }).reset_index()
    
    
    D_learning.columns = ['NODECODE','PICKINGLIST','INOUT',
                          'sum_QUANTITY','count_QUANTITY',
                          'max_TIMESTAMP_IN','min_TIMESTAMP_IN',
                          'max_RACK','min_RACK',
                          'max_BAY','min_BAY',
                          'max_LEVEL','min_LEVEL',
                          'max_LOCCODEX','min_LOCCODEX',
                          'max_LOCCODEY','min_LOCCODEY']
    
    # clean results
    D_learning['TIMESEC_SPAN'] = D_learning['max_TIMESTAMP_IN'] - D_learning['min_TIMESTAMP_IN']
    D_learning['RACK_SPAN'] = D_learning['max_RACK'] - D_learning['min_RACK']
    D_learning['BAY_SPAN'] = D_learning['max_BAY'] - D_learning['min_BAY']
    D_learning['LEVEL_SPAN'] = D_learning['max_LEVEL'] - D_learning['min_LEVEL']
    D_learning['LOCCODEX_SPAN'] = D_learning['max_LOCCODEX'] - D_learning['min_LOCCODEX']
    D_learning['LOCCODEY_SPAN'] = D_learning['max_LOCCODEY'] - D_learning['min_LOCCODEY']
    
    D_learning = D_learning.drop(columns=['max_TIMESTAMP_IN', 'min_TIMESTAMP_IN', 'max_RACK', 'min_RACK',
                                      'max_BAY', 'min_BAY', 'max_LEVEL','min_LEVEL', 'max_LOCCODEX',
                                      'min_LOCCODEX', 'max_LOCCODEY', 'min_LOCCODEY' ])
    
    D_learning['TIMESEC_SPAN'] = D_learning['TIMESEC_SPAN'].dt.seconds
    
    return D_learning


# %%

def histogramKeyVars(D_learning):
    output_figure={}
    
    
    
    columnToAnalyse = list(D_learning.columns)
    columnToAnalyse.remove('NODECODE')
    columnToAnalyse.remove('PICKINGLIST')
    columnToAnalyse.remove('INOUT')
    
    #split inbound and outbound
    D_learning_positive = D_learning[D_learning['INOUT']=='+']
    D_learning_negative = D_learning[D_learning['INOUT']=='-']
    
    for col in columnToAnalyse:
        
        #inbound
        fig = plt.figure()
        plt.hist(D_learning_positive[col])
        plt.title(f"Histogram: {col}, INBOUND")
        plt.xlabel(f"{col}")
        plt.ylabel("frequency")
        output_figure[f"{col}_inbound_histogram"]=fig
        
        #outbound
        fig = plt.figure()
        plt.hist(D_learning_negative[col])
        plt.title(f"Histogram: {col}, OUTBOUND")
        plt.xlabel(f"{col}")
        plt.ylabel("frequency")
        output_figure[f"{col}_outbound_histogram"]=fig
        
    return output_figure

# %%
def exploreKeyVars(D_learning):
    output_figures={}
    
    
    # pairplot
    fig = sn.pairplot(D_learning,hue='INOUT',diag_kind='hist')
    output_figures['pairplot']=fig
    
    
    
    D_learning_positive = D_learning[D_learning['INOUT']=='+']
    D_learning_negative = D_learning[D_learning['INOUT']=='-']
    
    #inbound_correlation
    df_corr = D_learning_positive.drop(columns=['NODECODE','PICKINGLIST','INOUT'])
    corr_matrix = df_corr.corr()
    plt.figure()
    fig = sn.heatmap(corr_matrix, annot=True)
    fig = fig.get_figure()
    output_figures['correlation_inbound'] = fig
    
    
    
    
    #outboud_correlation
    df_corr = D_learning_negative.drop(columns=['NODECODE','PICKINGLIST','INOUT'])
    corr_matrix = df_corr.corr()
    plt.figure()
    fig = sn.heatmap(corr_matrix, annot=True)
    fig = fig.get_figure()
    output_figures['correlation_outbound'] = fig
    
    return output_figures
