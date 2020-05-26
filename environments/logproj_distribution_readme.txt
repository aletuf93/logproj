If you have problem installing logproj_distribution.yml
run the following commands in your cmd


conda create -n logproj_distribution python=3.6
conda activate logproj_distribution 
conda install -c conda-forge notebook
conda install -c conda-forge/label/gcc7 osmnx
conda install -c plotly plotly
conda install -c anaconda seaborn
conda install -c anaconda scikit-learn
conda install -c conda-forge statsmodels
conda install -c anaconda openpyxl
conda install geopandas=0.6.3 -c conda-forge
conda install -c conda-forge geocoder
