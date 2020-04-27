# -*- coding: utf-8 -*-


# %%





# %%
def convertContainerISOcodeToTEUFEU(D_hu,codeField='_id'):
    #la funzione considera i codici ISO sulle dimensioni dei container per stabilire se sono TEU o FEU
    
    # D_hu e' un dataframe con i container
    #codeField e' la stringa con il nome della colonna di D_hu contenente i codici dei container
    #codeFieldMov e' la stringa con il nome della colonna di D_mov contenente i codici dei container
    
    ctSize = D_hu[codeField].str[:1]
    TEU=ctSize=='2'
    FEU=ctSize=='4'
    L5GO=ctSize=='L'
    ctSize[TEU]='TEU'
    ctSize[FEU]='FEU'
    ctSize[L5GO]='L5GO'
    ctSize[~(TEU|FEU|L5GO)]='OTHER'
    D_hu['ContainerSize'] = ctSize
        
    return D_hu






