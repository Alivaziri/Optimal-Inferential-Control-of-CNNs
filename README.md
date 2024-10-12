## Kalman-MPC
Official implementation of the `Kalman-MPC`; a GPU-compatible single pass filtering and smoothing algorithm that uses matrix variate distribution. The method is developed in [Optimal inferential Control of Convolutional Neural Networks].

## Highlights
* We propose to address optimal control of CNNs via probabilistic inference. This perspective centers around inferring or estimating  the best control actions from the control objectives and constraints, setting up the basis for the optimal inferential control (OIC) framework, opening  up the use of nonlinear state estimation of CNNs. 

* To computationally  implement  OIC for CNNs, we  propose  harnessing  the power of Monte Carlo sampling for inference and further combining it with the computational prowess  of GPUs.  To this end, we develop a sequential ensemble Kalman smoother (EnKS). This EnKS uniquely builds on matrix variate  distributions  and   can exploit GPUs' multi-dimensional array operations to accelerate computation considerably. 

## Requirements
Generally, there is no limit on the version of Pytorch or Python. It only depends on the requirements of the convolutional neural network model that is given to you or you trained. We have used
* Python 3.7 or higher(tested on 3.7 and 3.10).
* Pytorch (tested on 1.6 for Python 3.7 and 1.12 for Python 3.10).
* Other packages, such as Matplotlib, Numpy and Scipy, are also used.

## Data Generation
* For generating data, we have used the following repository:
 `https://github.com/isds-neu/PhyCRNet/tree/main/Datasets`

## 2D coupled Burgers PDE
* Navigate to the `Two_Dim_Burgers_Coupled` folder.
* The pretrained CNN model is given in `\Two_Dim_Burgers_Coupled\main\128by128\models\checkpoint.pt`
* The code for training a CNN model is located in `\Two_Dim_Burgers_Coupled\main\128by128\models\training\Burger2dUnet. py`. The grid size and CNN structure can be changed in this file. The physics informed techniques are used to train the neural network.
* To reproduce the results in the paper, run the code `main.py` in the `\Two_Dim_Burgers_Coupled\` folder. This will do the pixel wise control of the PDE's velocity field. Reynolds number is set to 200 in this problem.

## 2D uncoupled Burgers PDE
* Navigate to the 'Two_Dim_Burgers_Uncoupled' folder.
* The pretrained model is given in `\Two_Dim_Burgers_Uncoupled\trained_models\checkpoint.pt`. Reynolds number is set to 200 in this problem.
* The code for training a CNN model is located in `\Two_Dim_Burgers_Uncoupled\Training\Burgers_CNN.py`. The grid size and CNN structure can be changed in this file. The physics informed techniques are used to train the neural network.
* To reproduce the results in the paper, run the code 
    * `main_BoundaryControl.py` for applying the force to the top and bottom boundaries of the PDE (finite dimension action space).
    * `main_PixelwiseControl.py` for applying the force to each pixel of the PDE (infinite dimension action space in theory, high-dimension in the simulations).
* `\Two_Dim_Burgers_Uncoupled\MPPI\mppi_burgers.py` will generate the comparison results when the model predictive path integral controller is used for this task. You can change the covariance noises, temperature parameter Lambda, and number of samples as well as the prediction horizon in the code.

## Citing
* Please consider citing us if you find our research helpful:
