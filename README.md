# rawPy #
Release notes  
**version Beta_0_0** 11/04/2020  
note: the name has been given by my wife Allie (supposed to be a joke for raw pie)  
**version Beta_0_1** 03/05/2020  
contributors:  
- [John Leeman](https://github.com/jrleeman) ([Leeman Geophysical](https://www.leemangeophysical.com/))  
- [Martijn Van Den Ende](https://github.com/martijnende) (Nice University)  

This verison includes:  
- Addition to the utilities of rawPy  
    - Remove average slope (derivative)  
    - Take slope and polynominal fit
    - Correction for geometrical thinning (rgt)
    - Smoothing data using a low pass filter
   
- Additon of **Rate- and State Friction tool** to model velocity steps

**Scope:**  
This program has been created to build a tool for reproducible data analysis of raw data collected with the biaxial rock deformation apparatus **BRAVA**.  
It allows to import the raw data and produce an output file with elaborated quantities.  
It takes inspiration from the program [**xlook**](https://github.com/PennStateRockandSedimentMechanics/xlook) developed by Chris Marone et al. for the Penn State Biax and the module [**biaxtools**](https://github.com/jrleeman/biaxtools) developed by John Leeman.  

### Installation ### update the instructions
In later version I will package everything and allow for installation on a conda channel. FOr now you have to follow these instructions.

- Download [Anaconda](https://www.anaconda.com/distribution/) package manager Python version 3.7
- create a directory called PythonModules
- download the rawPy folder in the PythonModules directory
- Add the PythonModules folder to the PYTHONPATH. This allows to import the rawPy as a module.  
    - To add the directory to the PYTHONPATH execute this command in the **anaconda prompt** for windows user (access it by going on search and type the name) or a **terminal** for mac users.  
`conda-develop /path/to/PythonModules`  
    - If this does not work open the **Spyder IDE** by typing in the anaconda prompt/terminal `spyder`.  
Open the **Python path manager** and add to the PYTHONPATH the the choosen directory.  

### Usage ###
- Open **Anaconda prompt**/**Terminal** and type **jupyter lab**.  
- A local server will be initialized and you will be brought to the jupyter lab environment in a new browser window.  
- On the left side, open the navigation panel and navigate to the directory where you have saved the file `rawPy_reduction.ipynb`.  
- Open the `rawPy_reduction.ipynb`. To use the functions provided with this tool you will notice at the beginning the call:  
```python 
from rawPy.rawPy import rawPy as rp 
```
- To analyze the data follow the instruction given in that notebook.  
p.s. as a first try please follow the example given with the data provided in the example directory.  

For routine operational usage you will save the file `rawPy_reduction.ipynb` in the same directory of the experimental data and experimental sheet. You may want to rename the file as `bxxx_r.ipynb`.  

All the details about the different functions can be found in the manual.ipynb.

## PyRSF  
This is a web based GUI (Graphical User Interface) that allow to perform inversion models of experimental velocity steps based on the structure used in xlook. It uses as base algorithm the Py-RSF developed by Martijn Van Den Ende. The GUI is designed to allow for data loading, plotting and inversion.  

#### Installation  
Some steps must be taken to install all the packages that allow to use the tool. I assume that you have installed Anaconda.  
- Install modules required for the inversion  
    - If you have not already add the forge channel to the anaconda : `conda config --add channels conda-forge`
    - install the package `conda install emcee`
    
- Install ipywidgets:  
    - Open anaconda prompt (a terminal for mac users) and copy and paste the commands highlighted below
    - install nodejs `conda install nodejs`
    - install ipympl `conda install ipympl`
    - install jupyter extensions:
        - `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
        - `jupyter labextension install jupyter-matplotlib`
        - `jupyter nbextension enable --py widgetsnbextension`  
        
If you find any problems please try to resolve yourself and don't give up. This is the only way to learn !!  
Some useful resources can be found at:  
    - [Official documentation](https://ipywidgets.readthedocs.io/en/latest/user_install.html)  
    - [Stackoverflow](https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab)  
    - [Stackoverflow](https://stackoverflow.com/questions/50149562/jupyterlab-interactive-plot)  
and if you don't find the answer here, internet is a beautiful place to find answers.  
- install [bqplot](https://github.com/bqplot/bqplot) as the widget plotting library  
    - `conda install -c conda-forge bqplot`
    - `jupyter labextension install bqplot`  
- install [voila](https://github.com/voila-dashboards/voila) that is used to render the web application  
    - `conda install -c conda-forge voila`
    - `jupyter labextension install @jupyter-voila/jupyterlab-preview`  

#### Usage
If everything is installed properly, open an anaconda prompt and navigate to directory where the inversion_rsf_master.ipynb is placed. Once there type `voila inversion_mater.ipynb` and a web page should open in you browser where you can use the RSF tool. 
If you have any problems:  
1. Open the notebook and execute the different cells and see what is the error  
2. Contact me at marco.scuderi@uniroma1.it
    
## Author comments ##  
There is still a very long to do list !!  
Some of the things I will add soonish are: 
- non linear elastic correction  
- package the installation (not in the near future) 
- More functions depending on users input  

Something that is unlikely to happen any time soon: 
- put toghether a GUI 

For any comment/complain/suggestion email me at: marco.scuderi@uniroma1.it
