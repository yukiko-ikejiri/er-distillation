import argparse
import copy
import os
import torch
import json
import pandas as pd

from utils.data_utils import read_single_row_data
from utils.train_eval import train, inference
from data import GPTDataset
from model import load_model

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

parser = argparse.ArgumentParser(description='AnyMatch with LLM reasoning experiment.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--base_model', type=str, default='gpt2')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--serialization_mode', type=str, default='mode1')
parser.add_argument('--row_sample_func', type=str, default='automl_filter')
parser.add_argument('--patience_start', type=int, default=20)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--filter_correct_reasoning', action='store_true', default=False)
args = parser.parse_args()

seed = args.seed
base_model = args.base_model
dataset_name = args.dataset_name
serialization_mode = args.serialization_mode
row_sample_func = args.row_sample_func
patience_start = args.patience_start
output_dir = args.output_dir
filter_correct_reasoning = args.filter_correct_reasoning

model, tokenizer = load_model(base_model)
dataset_dir = f'data/prepared/{dataset_name}'

if base_model == 'gpt2':
    lr = 2e-5
    DatasetClass = GPTDataset
else:
    raise ValueError('Model not found.')

tbs = 2

# Set random seeds
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.makedirs(output_dir, exist_ok=True)

print('-----' * 10)
print(f'Baseline vs Reasoning experiment on {dataset_name} dataset.', flush=True)
baseline_train_df, baseline_valid_df, baseline_test_df = read_single_row_data(
    dataset_name, dataset_dir, output_dir, serialization_mode, row_sample_func,
    include_reasoning=False, filter_correct_reasoning=filter_correct_reasoning, print_info=True
)
baseline_train_d = DatasetClass(tokenizer, baseline_train_df, max_len=500)
baseline_valid_d = DatasetClass(tokenizer, baseline_valid_df, max_len=500)
baseline_test_d = DatasetClass(tokenizer, baseline_test_df, max_len=10000)

reasoning_train_df, reasoning_valid_df, reasoning_test_df = read_single_row_data(
    dataset_name, dataset_dir, output_dir, serialization_mode, row_sample_func,
    include_reasoning=True, filter_correct_reasoning=filter_correct_reasoning, print_info=True
)
reasoning_train_d = DatasetClass(tokenizer, reasoning_train_df, max_len=500)
reasoning_valid_d = DatasetClass(tokenizer, reasoning_valid_df, max_len=500)
reasoning_test_d = DatasetClass(tokenizer, reasoning_test_df, max_len=10000)

print(f'Dataset sizes:', flush=True)
print(f'  Baseline - Train: {len(baseline_train_d)}, Valid: {len(baseline_valid_d)}, Test: {len(baseline_test_d)}', flush=True)
print(f'  Reasoning - Train: {len(reasoning_train_d)}, Valid: {len(reasoning_valid_d)}, Test: {len(reasoning_test_d)}', flush=True)

print(f'Experiment configuration:')
print(f'\tseed: {seed}\tbase_model: {base_model}\tdataset_name: {dataset_name}\tmode: {serialization_mode}')
print(f'\tmax_len: 500\tlr: {lr}\tbatch_size: {tbs}\tpatience: 6\tp_start: {patience_start}')

print('-----' * 10)
print(f'Training baseline model on {dataset_name} dataset.', flush=True)
baseline_model = copy.deepcopy(model)
best_model_baseline = train(
    tokenizer, baseline_model, baseline_train_d, baseline_valid_d, 
    epochs=50, lr=lr, seed=seed, patient=True, save_model=False, save_freq=50, 
    train_batch_size=tbs, valid_batch_size=tbs*2, save_model_path='',
    save_result_prefix='baseline', patience=6, patience_start=patience_start, 
    base_model=base_model
)
print('Baseline training completed.')

print('-----' * 10)
print(f'Training reasoning-enhanced model on {dataset_name} dataset.', flush=True)
reasoning_model = copy.deepcopy(model)
best_model_reasoning = train(
    tokenizer, reasoning_model, reasoning_train_d, reasoning_valid_d,
    epochs=50, lr=lr, seed=seed, patient=True, save_model=False, save_freq=50, 
    train_batch_size=tbs, valid_batch_size=tbs*2, save_model_path='',
    save_result_prefix='reasoning', patience=6, patience_start=patience_start, 
    base_model=base_model
)
print('Reasoning-enhanced training completed.')

print('-----' * 10)
print('Evaluating models on test data without reasoning.')
baseline_f1, baseline_acc, baseline_preds, baseline_gts = inference(
    tokenizer, best_model_baseline, baseline_test_d, batch_size=tbs*2, base_model=base_model
)

reasoning_f1, reasoning_acc, reasoning_preds, reasoning_gts = inference(
    tokenizer, best_model_reasoning, baseline_test_d, batch_size=tbs*2, base_model=base_model
)

if len(baseline_preds) != len(reasoning_preds):
    raise ValueError(
        f"Prediction length mismatch: "
        f"baseline_preds={len(baseline_preds)}, "
        f"reasoning_preds={len(reasoning_preds)}"
    )
predictions_df = pd.DataFrame({
    'baseline_label': baseline_preds,
    'reasoning_enhanced_label': reasoning_preds
})
output_file = f'{output_dir}/{dataset_name}_{base_model}_test_predictions.csv'
predictions_df.to_csv(output_file, index=False)

print('-----' * 10)
print('Results:')
print(f'Baseline - F1: {baseline_f1*100:.2f}, Acc: {baseline_acc*100:.2f}')
print(f'Reasoning - F1: {reasoning_f1*100:.2f}, Acc: {reasoning_acc*100:.2f}')
print(f'Improvement - F1: {(reasoning_f1-baseline_f1)*100:.2f}, Acc: {(reasoning_acc-baseline_acc)*100:.2f}')

results = {
    'experiment_type': 'single_dataset_baseline_vs_reasoning',
    'dataset': dataset_name,
    'models': {
        'baseline': {
            'f1': float(baseline_f1),
            'accuracy': float(baseline_acc),
            'training_size': len(baseline_train_d)
        },
        'reasoning': {
            'f1': float(reasoning_f1),
            'accuracy': float(reasoning_acc),
            'training_size': len(reasoning_train_d)
        }
    },
    'improvement': {
        'f1_improvement': float(reasoning_f1 - baseline_f1),
        'accuracy_improvement': float(reasoning_acc - baseline_acc),
        'f1_improvement_percent': float((reasoning_f1 - baseline_f1) * 100),
        'accuracy_improvement_percent': float((reasoning_acc - baseline_acc) * 100)
    },
    'experiment_config': {
        'seed': seed,
        'base_model': base_model,
        'serialization_mode': serialization_mode,
        'row_sample_func': row_sample_func,
        'patience_start': patience_start,
        'learning_rate': lr,
        'batch_size': tbs,
        'max_length': 500
    }
}
output_file = f'{output_dir}/{dataset_name}_{base_model}_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f'Results saved to {output_file}')
print('-----' * 10)

