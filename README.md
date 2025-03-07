# Age Prediction from Health Metrics

This repository contains a complete implementation of an age prediction system based on health metrics, including modular training code, inference pipeline, and an interactive demo. The neural network model is designed to predict a person's age based on common health metrics, and the repository provides both a Jupyter notebook for exploration and production-ready Python modules for deployment.

## Dataset

The model is trained on a synthetic dataset that contains various health-related features for age prediction.

**Data Source:** [Human Age Prediction Synthetic Dataset](https://www.kaggle.com/datasets/abdullah0a/human-age-prediction-synthetic-dataset) on Kaggle.

**Google-colab notebook** You can also find the entire implementation at this [colab link](https://colab.research.google.com/drive/1PMDkkCiwqRIsOdbo4zhyfLPnI4n9O0_C?usp=sharing)

## Project Structure

```
├── model.py        # Neural network architecture
├── train.py        # Training script
├── inference.py    # Inference utilities
├── demo.py         # Gradio demo interface
└── age_predictor.pth  # Pre-trained model weights
```

## Quick Start

### Running the Demo

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Launch the Gradio demo:

```bash
python demo.py
```

The demo will start a local server (typically at http://127.0.0.1:7860) where you can input health metrics and get age predictions.

### Using the Pre-trained Model

The `age_predictor.pth` file contains pre-trained weights that can be used for:

- Quick inference
- Creating API endpoints
- Transfer learning

For inference:

```python
from inference import predict_age
prediction = predict_age(health_metrics)
```
