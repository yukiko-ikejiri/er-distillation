# Practices of Using Small and Large Language Models for Entity Resolution Task

This repository contains the implementation of the experiments presented in the bachelor thesis, "Practices of Using Small and Large Language Models for Entity Resolution Task".

## Instructions to Reproduce the Experiments

### 1. Install dependencies

Create a conda environment using the provided `environment.yml` file, then activate it:

```sh
conda env create -f environment.yml
conda activate er-distillation
```

### 2.  Download and organize the data
Download the raw Magellan datasets and place them in the `data/raw` directory. Additionally, place the corresponding LLM-generated reasoning files (e.g., `train_results.jsonl`) into the  `data/reasoning` directory. 

### 3. Prepare the datasets
Follow the preparation steps in the `data/preprocess.ipynb` notebook. This notebook preprocesses the raw data, integrates reasoning texts, runs the AutoML filter to identify challenging samples, and generates the final datasets (`train.csv`, `valid.csv`, `test.csv`) in the `data/prepared/` directory.

### 4. Run the Main Experiment in Section 5.1
The following script trains and evaluates both the baseline and the reasoning-enhanced models, comparing their performance for a given dataset:

```sh
python reasoning.py \
    --seed 42 \
    --base_model gpt2 \
    --dataset_name DATASET_NAME \
    --serialization_mode mode1 \
    --row_sample_func automl_filter \
    --patience_start 20 \
    --output_dir results
```
Replace DATASET_NAME with the name of the dataset you want to run (e.g., `amgo`, `foza`, `dbac`).

### 5. Run the Ablation Studies in Section 5.4 & 5.5
The scripts for the ablation studies are organized in the same order as in the paper.
- **Isolating the Effect of Explicit Answers (Section 5.4)**
    To run this experiment, first modify `data/preprocess.ipynb` by setting `remove_answer=True` in the `prepare_magellan_row_pairs` function call. Then, re-run the notebook to generate the modified datasets and execute the script from step 4.
- **Assessing the Impact of "Correct" Reasoning (Section 5.5)**
    To run this experiment, add the `--filter_correct_reasoning` flag to the main script. This will train the model only on samples where the LLM's prediction was correct.
    ```sh
    python reasoning.py \
        --seed 42 \
        --base_model gpt2 \
        --dataset_name DATASET_NAME \
        --serialization_mode mode1 \
        --row_sample_func automl_filter \
        --patience_start 20 \
        --output_dir results \
        --filter_correct_reasoning
    ```

## Locate the main components of the implementation
- **Data Preparation & AutoML filter**: The entire data preparation pipeline, including the integration of reasoning texts and the implementation of the AutoML filter, can be found in `data/preprocess.ipynb`.
- **Data Reading and Sampling**: The logic for reading the prepared data and applying the `automl_filter` sampling strategy is in the `read_single_row_data` function in `utils/data_utils.py`.
- **Model Training & Inference**: The train and inference methods, which handle the model fine-tuning and evaluation loops, are located in `utils/train_eval.py`.
- **Main Experiment Script**: The reasoning.py script orchestrates the experimental flow, from loading data to training models and reporting the final results.

## Appendix: Reasoning Data Generation
The reasoning data used in our experiments was provided by the project supervisor.

The script `generate_reasoning.py` is included in this repository as a **reference implementation** to show how such data could be generated. **This script was not used for the final experiments**. We provide it for transparency and as a potential starting point for generating similar data.

To use it, you would need to run:
```sh
python generate_reasoning.py \
    --dataset_dir data/prepared/DATASET_NAME \
    --output_dir reasoning_generated/DATASET_NAME \
    --model gpt-4 \
    --api_key YOUR_OPENAI_API_KEY
```