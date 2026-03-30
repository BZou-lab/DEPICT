# LLM_derived_Embeddings
The final data preprocessed from this code is called 'gptEmbed_Jul9_final.csv' inside the path './Data/FinalData'. Please run get_Augmented_text_partX.py before get_Embedding_partX.py.

All these codes are under the ChatGPT API framework. When you use these codes, please fill your API key.

## get_Augmented_text_partX.py
This is the code to get LLM-augmented texts for each drug from tabular data. 1, 2, 3, 4 just mean different part of the data as I divided the initial input data into 4 parts to speed up calculation by parallelization.

## get_Embedding_partX.py
This is the code to get the LLM-derived embeddings from the augmented texts. 

## RawData
This folder contains the raw data and some halfway data generated from codes. Stacking the final 4 resulting datasets together will generate the 'gptEmbed_Jul9_final.csv' inside the path './Data/FinalData'.

This folder also contains the prompt used in the 'get_Augmented_text_partX.py'.