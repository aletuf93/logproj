To install the package:


1) clone or fork the repository from github.

2) install python on your computer using miniconda https://docs.conda.io/en/latest/miniconda.html (remember to check the "ADD TO PATH" 
checkbox during the setup)

3) Install the virtual environments in the folder environment with the following steps:
-open the cmd
-use the cmd to go into the environments folder
-run the command "conda env create -f logproj_distribution.yml" to install the distribution environments (with road graph support)
-run the command "conda env create -f logproj.yml" to install the logproj environments

4) to run the code:
-open the cmd
-run the command "conda activate logproj_distribution" to activate the distribution environment
-OR run the command "conda activate logproj" to activate the logproj environment
-run the command "jupyter notebook" to run and edit the code