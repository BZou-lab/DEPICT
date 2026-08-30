# DEPICT
This is the GitHub page for DEPICT (Drug rEsponse Prediction in transCriptomics with Transformers).

Paper Title: Learning transferable cellular state transitions from multimodal perturbation data.

## Environment
The "requirements.txt" are used to set up the virtual environment.

Detailed virtual environment setup is listed below:

Python: 3.11.13

Required packages and detailed version:

numpy: 2.2.6

pandas: 2.3.2

scanpy: 1.11.4

scikit-learn: 1.7.1

scipy: 1.16.1

torch: 2.8.0

tqdm: 4.67.1

rdkit: 2025.3.3

anndata: 0.12.2

matplotlib: 3.10.3

seaborn: 0.13.2

After creating the virtual environment, use these commands to download all required packages:

python -m pip install --upgrade pip

pip install -r requirements.txt


## Folders in the main page
### Code
"Code" folder contains all the code used in DEPICT, including preprocessing data; training and inference; downstream analysis.
### Data
"Data" folder contains all the data used in DEPICT. Some of the data are not in GitHub due to size limit, and these data can be obtained from [Zenodo](https://zenodo.org/records/19207077).
### Model
"Model" folder contains the saved best check points for DEPICT after training.
### Results
"Results" folder contains the numerical results for model comparison and tuning results.