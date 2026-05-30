# ATSR

This project provides the implementation of ATSR.

## Environment Setup

Create a conda environment and install all required dependencies:

```bash
conda create -n ATSR python=3.9
conda activate ATSR

pip install -r requirements.txt
```

To enable language processing with SpaCy, install the language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Dataset Preparation

We conduct experiments on three three common datasets: RAMS, WikiEvents and MLEE. 
You can download and preprocess the datasets by running:

```bash
bash ./data/download_dataset.sh
```

## Training and Evaluation

Train ATSR on different datasets using the following scripts:

```bash
bash ./scripts/train_rams.sh

bash ./scripts/train_wikievent.sh

bash ./scripts/train_mlee.sh
```
Each script is tailored to a specific dataset and configuration. You can modify the settings in the corresponding scripts.

After training, evaluate the model using:

```bash
bash ./scripts/infer_rams.sh

bash ./scripts/infer_wikievent.sh

bash ./scripts/infer_mlee.sh
```
---
