# kilter.net

Using a Convolutional, Conditional Variational Autoencoder for Kilter Board analysis conditional on the difficulty grade and angle.

Models and training scripts provided by [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)


To generate the training data, go into the data_preprocessing folder and run: `python clean_kilter_df.py`

### Required packages
Create a clean environment and install the below packages via `pip`.
```
lightning (>=2.0.1)
scikit-learn (>=1.2.2)
wandb (>=0.14.2)
```
