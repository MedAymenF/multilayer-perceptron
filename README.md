# multilayer-perceptron
Subject created by the 42AI association. This project is an introduction to artificial neural networks thanks to the implementation of a multilayer perceptron.

## Subject
[Subject](Resources/en.subject.pdf)

## Score
[![mfarhi's 42 multilayer-perceptron Score](https://badge42.vercel.app/api/v2/cl5twx4hw007809mfvxwmzeal/project/2351922)](https://github.com/JaeSeoKim/badge42)

## Split the dataset
```
python3 evaluation.py
```

## Train the model
```
./mlp.py --dataset data_training.csv
```

## Evaluate the model on the test set
```
./mlp.py --dataset data_test.csv --predict saved_model.npy
```