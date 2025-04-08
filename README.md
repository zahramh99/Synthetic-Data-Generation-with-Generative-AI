# Synthetic Data Generation with Generative AI

This project demonstrates synthetic data generation using Generative Adversarial Networks (GANs). It generates artificial data that mimics real-world app usage patterns while preserving privacy.

## Features
- Preprocessing pipeline for app usage data
- Custom GAN architecture implementation
- Training and evaluation scripts
- Synthetic data generation capabilities

## Requirements
- Python 3.8+
- TensorFlow 2.x
- pandas, numpy, scikit-learn

## Installation
```bash
cd synthetic-data-generation-GAN
pip install -r requirements.txt

## Usage
Place your dataset in data/ directory
Preprocess data:
python src/preprocess.py
Train the GAN model:
python src/train.py
Generate synthetic data:
python src/generate.py

## Results
The trained model can generate synthetic app usage data with similar statistical properties to the original dataset while protecting user privacy.

## Contributing
Pull requests are welcome. For major changes, please open an issue first.
