# Overcoming the Feature Selection Issue in the Pricing of American options - Source code

## Introduction
The aim is to use the Neural networks feature extraction ability to overcome the feature selection issue for the least squares Monte Carlo method (LSM).
The method is called feedforward neural network Monte Carlo (FNNMC).

## Technologies
<ul>
    <li> Python 3.8 </li>
    <li> Torch 1.8 </li>
</ul>

## Setup
First, install the required libraries in "requirements.txt" and create empty folders TrainedModels and data in the root.
To price one of the options with FNNMC, select "generateData.py" and uncommon the chosen option in this file. The script will create the paths for training and pricing. 
Hereafter, run "price{optionName}.py" to get a price and standard error.


## Sources
The paper is inspired by Glasserman's book "Monte Carlo Methods in Financial Engineering”.

## Author
You can see my other projects at [Peter Pommergård Lind]{https://mrppl.github.io/}
