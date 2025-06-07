# Deep Learning Mini-Project (EE-559) — Group 8  
## DetoxText: Text Detoxification Using Finetuned Encoder-Decoder Models

### Project Overview
This repository contains the implementation of our **Deep Learning Mini-Project** for EE-559 at EPFL. The project focuses on **fostering safer online spaces** by developing deep learning models that detect hate speech in various formats, including text, images, memes, videos, and audio content.

### Objectives
- Develop deep learning models that accurately classify hate speech while minimizing false positives.
- Evaluate model performance with benchmarks and interpretability metrics.
- Address ethical and legal considerations in AI-powered content moderation.


### Project Structure

The repository contains the following directories and files:

#### Directories

- `data_preprocessing/` contains scripts for loading and preprocessing datasets.
- `eval/` includes evaluation routines and performance metrics.
- `plots/` holds Jupyter notebooks and experiment visualizations.
- `test/` includes scripts for testing training and inference behavior.
- `trainer/` implements the training loop and model management.
- `utils/` provides utility functions used across the project.

#### Top-Level Files

- `README.md` – Main project documentation.
- `requirements.txt` – Python dependencies.
- `main_config.yaml` – Configuration file for training and evaluation.
- `main.py` – Entry point to run the training pipeline.
- `basic_running_scripts.sh` – Shell script(s) to launch experiments.
- `tokens.yaml` – Contains token/API configuration if needed.

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
   ```

### Setup & Installation
1. **Clone the repository**
   ```sh
   git clone https://github.com/your_username/deep-learning-project.git
   cd deep-learning-project
   ```

2. **Create a virtual environment** (optional)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

### Dataset
- The dataset used for training includes **text, images, and audio** related to hate speech detection.
- Preprocessing steps include **tokenization, augmentation, and feature extraction**.

### Model Architecture
- **Multi-modal deep learning models** integrating **transformers, CNNs, and RNNs**.
- **Transfer learning** is used with pre-trained models such as **BERT, CLIP, and ResNet**.
- Regularization techniques to mitigate bias and improve generalization.

### Training
To train the model, use:
```sh
python scripts/train.py --config config.yaml
```

### Evaluation
To evaluate model performance:
```sh
python scripts/evaluate.py --model_path models/best_model.pth
```

### Contributors
- **Your Name**
- **Project Group Members**
