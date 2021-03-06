# Neural Networks Demystified

Supporting iPython notebooks for the YouTube Series Neural Networks Demystified. I've included formulas, code, and the text of the movies in the iPython notebooks, in addition to raw code in python scripts. 

iPython notebooks can be downloaded and run locally, or viewed using nbviewer: http://nbviewer.ipython.org/

## Using the iPython notebook
The iPython/Jupyter notebook is an incredible tool, but can be a little tricky to setup. I recommend the [anaconda] (https://store.continuum.io/cshop/anaconda/) distribution of python. I've written and tested this code with the the anaconda build of python 2 running on OSX. You will likely get a few warinings about contour plotting - if anyone has a fix for this, feel free to submit a pull request. 

## Using Torch
# Torch7 Version 
There's also a Torch version of the scripts, in Lua. You will need to install [torch](http://torch.ch/). 
Once you've installed it, to run the scripts just type: 
```
th -i <script_name>.lua
```
### About Torch
[Torch7](torch.ch) is "a scientific computing framework with wide support for machine learning algorithms". It's very popular among ML researchers. The torch folder contains all the equivalent scripts to run this tutorial using torch, with same results. Plotting is still done in python for now. iTorch notebooks will be added in future.  

