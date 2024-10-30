## Optimal Inferential Control of Convolutional Neural Networks
Official implementation of the `Kalman-MPC`; a GPU-compatible single pass filtering and smoothing algorithm that uses matrix variate distribution. The method is developed in [Optimal Inferential Control of Convolutional Neural Networks](https://scholar.google.com/scholar?q=Optimal+Inferential+Control+of+Convolutional+Neural+Networks&hl=en&as_sdt=0&as_vis=1&oi=scholart).

## Highlights
* We propose to address optimal control of CNNs via probabilistic inference.
* This perspective centers around inferring or estimating  the best control actions from the control objectives and constraints, setting up the basis for the optimal inferential control (OIC) framework.
* OIC framework opens up the use of nonlinear state estimation of CNNs. 

* To computationally  implement  OIC for CNNs, we  propose  harnessing  the power of Monte Carlo sampling for inference.
* We further combine it with the computational prowess  of GPUs.
* We develop a sequential ensemble Kalman smoother that uniquely builds on matrix variate  distributions  and   can exploit GPUs' multi-dimensional array operations to accelerate computation considerably. 

## Requirements
Generally, there is no limit on the version of Pytorch or Python. It only depends on the requirements of the convolutional neural network model that is given to you or you trained. We have used
* Python 3.7 or higher(tested on 3.7 and 3.10).
* Pytorch (tested on 1.6 for Python 3.7 and 1.12 for Python 3.10).
* Other packages, such as Matplotlib, Numpy, and Scipy, are also used.

## Data Generation
* For generating data, we have used the following repository:
 `https://github.com/isds-neu/PhyCRNet/tree/main/Datasets`

## 2D Burgers PDE
* Navigate to the `TwoDim_BurgersPDE` folder.
* The pre-trained CNN model is given in `\TwoDim_BurgersPDE\main\128by128\models\checkpoint.pt`
* The code for training a CNN model is in `\TwoDim_BurgersPDE\main\128by128\models\training\Burger2dUnet.py`. The grid size and CNN structure can be changed in this file. Physics-informed techniques are used to train the neural network.
* To reproduce the results in the paper, run the code `main.py`. This will control the PDE's velocity field pixel-wise.

## Comparisons between `Kalman-MPC` and model predictive path integral control
* Navigate to the `Comparisons` folder.
* The pre-trained model is given in `\Comparisons\trained_models\checkpoint.pt`. 
* The code for training a CNN model is in `\Comparisons\Training\Burgers_CNN.py`. 
* To reproduce the results in the paper, run the code 
    * `main_BoundaryControl.py` for applying the force to the top and bottom boundaries of the PDE (finite dimension action space).
    * `main_PixelwiseControl.py` for applying the force to each pixel of the PDE (infinite dimension action space in theory, high-dimension in the simulations).
* `\Comparisons\MPPI\mppi_burgers.py` will generate the comparison results when the model predictive path integral controller is used for this task. You can change the covariance noises, temperature parameter Lambda, the number of samples, and the prediction horizon in the code.

## Citation
* Please consider citing us in your paper if you use our code/paper.
```bibtex
@article{vaziri2024optimal,
  title={Optimal Inferential Control of Convolutional Neural Networks},
  author={Vaziri, Ali and Fang, Huazhen},
  journal={arXiv preprint arXiv:2410.09663},
  year={2024}
}
