# Fine-Tuning a Model for a Large Open-Source Project

## Overview

This repository demonstrates fine-tuning of Llama-3.2-1B on a [APPS](https://arxiv.org/pdf/2105.09938) which consists of python code problems with solutions and expected outputs given program inputs.
We fine tune a model with QLoRA and measure the accuracy using chrf text similarity metric.
Detailed description of the algorithm and training results are in the ```docs``` directory.



## Repository Structure

```
.
├── README.md               # Project documentation
├── requirements.txt        # Required dependencies
├── training_notebook.ipynb # Notebook used during the actual training
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── model.py            # Model loading and configuration
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── results/                # Experiment results
│   ├── chekcpoint-#/       # Saved checkpoints
│   └── finetuned-model/    # Model weights afeter fine tuning
└── docs                    # Documents                       
    ├── analysis.md         # Plot analysis
    └── documentation.md    # Detailed methodology and notes
```

## Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/FilipBaciak/Jet_Brains_Internship
cd Jet_Brains_Internship
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```



## Usage

### **Train the Model**
To fine-tune the model on the dataset, run:
```bash
python src/train.py
```
To function properly, the script requires an environmental variable ```"HF_API_KEY"``` which contains a HuggingFace API key with an acces to LLama-3.2.
It can be also added as a secret variable in the Google Colab.


## Contact

For questions or suggestions, reach out to [Filip Baciak](mailto:f.baciak@student.uw.edu.pl).

