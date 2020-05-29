# Deep Learning: Predicting Daily Bike Rental Riderships

[![Twitter Follow](https://img.shields.io/twitter/follow/youldash.svg?style=social?style=plastic)](https://twitter.com/youldash)

[ANN]: assets/neural_network.png "Artificial Neural Network"

## License

By using this site, you agree to the **Terms of Use** that are defined in [LICENSE](https://github.com/youldash/First-Neural-Network/blob/master/LICENSE).

## About

The goal of this project is to build an [Artificial Neural Network (ANN)](https://en.wikipedia.org/wiki/Artificial_neural_network) from scratch to carry out a prediction problem on a real dataset. By building an ANN from the ground up, we'll have a much better understanding of gradient descent, back-propagation, and other concepts that are important to know before we move to higher-level tools such as PyTorch. We'll also get to see how to apply these networks to solve real prediction problems.

<div align="center">
	<img src="assets/neural_network.png" width="100%" />
</div>

The data comes from the [UCI Machine Learning database](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

This project was developed in partial fulfillment of the requirements for Udacity's [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) program.

## Results

### Prediction Losses

![LOSSES](/assets/prediction-losses.png)

### Predictions After Training

![PREDICTIONS](/assets/predictions.png)

## Installation

For best the experience with managing dependencies, we strongly advise you to install [Anconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html).

Create a virtual environment with `conda`:

```
conda create --name deep-learning python=3
```

Then, activate the environment:

```
conda activate deep-learning
```

Then, install the dependencies:

```
pip install -r requirements.txt
```

You can also download/clone this repository and launch the notebook using the following command:

```
jupyter-notebook Your_first_neural_network.ipynb
```

## Usage

Run all code cells in the `Your_first_neural_network.ipynb` notebook file. Also have a look at the file `my_answers.py` for details on how to implement the ANN from scratch.
