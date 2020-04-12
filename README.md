# rawPy
Toll to analyze raw data from the biaxial apparatus BRAVA
Release notes  
**versio Beta_0_0** 11/04/2020  
note: the name has been given by my wife Allie (supposed to be a joke for raw pie)  
**Scope:**  
This program has been created to build a tool for reproducible data analysis of raw data collected with the biaxial rock deformation apparatus **BRAVA**.  
It allows to import the raw data and produce an output file with elaborated quantities.  
It takes inspiration from the program [**xlook**](https://github.com/PennStateRockandSedimentMechanics/xlook) developed by Chris Marone et al. for the Penn State Biax and the module [**biaxtools**](https://github.com/jrleeman/biaxtools) developed by John Leeman.  

### 1. Installation ###  
- Download [Anaconda](https://www.anaconda.com/distribution/) package manager Python version 3.7
- Place the rawPy in a directory of your choice and add it to the PYTHONPATH. This allows to import the rawPy as a module.  
    - To add the directory to the PYTHONPATH execute this command in the **anaconda prompt** for windows user (access it by going on search and type the name) or a **terminal** for mac users.  
`conda-develop /path/to/module/`  
    - If this does not work open the **Spyder IDE** by typing in the anaconda prompt/terminal `spyder`.  
Open the **Python path manager** and add to the PYTHONPATH the the choosen directory (see picture below).  

### 2.Usage ###
- Open **Anaconda prompt**/**Terminal** and type **jupyter lab**.  
- A local server will be initialized and you will be brought to the jupyter lab environment in a new browser window.  
- On the left side, open the navigation panel and navigate to the directory where you have saved the file `rawPy_reduction.ipynb`.  
- Open the `rawPy_reduction.ipynb`. To use the functions provided with this tool you will notice at the beginning the call:  
```python 
import rawPy as rp 
```
- To analyze the data follow the instruction given in that notebook.  
p.s. as a first try please follow the example given with the data provided in the example directory.  

For routine operational usage you will save the file `rawPy_reduction.ipynb` in the same directory of the experimental data and experimental sheet. You may want to rename the file as `bxxx_r.ipynb`  

## Author comments ##  
There is still a very long to do list !!  
Some of the things I will add soonish are: 
- Smoothing data 
- non linear elastic correction  
- running average slope (i.e. derivative of the data)
- RSF friction tool to perform inversion of experimental data  
- More functions depending on users input  

Something that is unlikely to happen any time soon: 
- put toghether a GUI 

For any comment/complain/suggestion email me at: marco.scuderi@uniroma1.it
