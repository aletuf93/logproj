#funzioni di utilita' (gestione cartelle, etc.


import os
# %% utilities

def creaCartella(currentDir,newFolderName):
        newDirPath = os.path.join(currentDir, newFolderName)
        try:
                os.mkdir(newDirPath)
        except OSError:
                print(f"Cartella {newFolderName} già esistente")
            
        
        return currentDir, newDirPath