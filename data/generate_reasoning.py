import argparse
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# This code requires the OpenAI Python library v1.0.0 or later
# If you encounter issues, you may need to update the library:
# pip install --upgrade openai

# Example usage:
# python generate_reasoning.py \
#     --dataset_dir data/prepared/itam \
#     --output_dir data/reasoning/itam \
#     --model gpt-4 \
#     --api_key YOUR_API_KEY \

def extract_entity_values(record, suffix):
    # Extract attribute values from a record based on suffix (_l or _r)
    values = {}
    for key, value in record.items():
        if key.endswith(suffix) and key != f'id_{suffix}':
            attr_name = key.replace(f'_{suffix}', '')
            values[attr_name] = value
    return values

def format_entity(entity_values):
    # Format entity values as a string
    return ", ".join([f"{k}: {v}" for k, v in entity_values.items() if pd.notna(v)])

def generate_reasoning(record, client, model="gpt-4"):
    left_values = extract_entity_values(record, 'l')
    right_values = extract_entity_values(record, 'r')

    left_entity = format_entity(left_values)
    right_entity = format_entity(right_values)
    
    prompt = f"""Do the two entity descriptions refer to the same real-world entity? Compare their fields. Decide if differences are minor (e.g. metadata variations) or indicate they refer to different entities. Don’t assume a few field differences mean different entities.
Give reasoning and:
Answer: Yes. / Answer: No.
Limit to 100 tokens.

Entity 1: {left_entity}
Entity 2: {right_entity}"""
    
    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return None

def process_dataset(dataset_df, client, sample_size=None, model="gpt-4", 
                   output_path=None, batch_size=10):
    # Sample the dataset if necessary
    if sample_size and len(dataset_df) > sample_size:
        # Stratified sampling to maintain label distribution
        pos_samples = dataset_df[dataset_df['label'] == 1].sample(
            min(sample_size // 2, sum(dataset_df['label'] == 1)), 
            random_state=42
        )
        neg_samples = dataset_df[dataset_df['label'] == 0].sample(
            min(sample_size - len(pos_samples), sum(dataset_df['label'] == 0)), 
            random_state=42
        )
        sampled_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
    else:
        sampled_df = dataset_df
    
    results = []

    for i in range(0, len(sampled_df), batch_size):
        batch = sampled_df.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sampled_df)-1)//batch_size + 1}")
        
        batch_results = []
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc="Generating reasoning"):
            reasoning = generate_reasoning(row, client, model=model)
            batch_results.append({
                **row.to_dict(),
                "reasoning": reasoning
            })
        
        results.extend(batch_results)
        
        # Save intermediate results
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(f"{output_path}.temp", index=False)
        print(f"Saved intermediate results to {output_path}.temp")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    if os.path.exists(f"{output_path}.temp"):
        os.remove(f"{output_path}.temp")
    
    return results_df

def process_split(split_name, client, dataset_dir, output_dir, model, sample_size, batch_size):
    input_path = os.path.join(dataset_dir, f'{split_name}.csv')
    if not os.path.exists(input_path):
        return None

    df = pd.read_csv(input_path)
    print(f" - {split_name.capitalize()} samples: {len(df)}")
    
    output_path = os.path.join(output_dir, f'{split_name}_reasoning.csv')
        
    print(f"Processing {split_name} set using {model}")
    results_df = process_dataset(
        df,
        client,
        sample_size=sample_size,
        model=model,
        output_path=output_path,
        batch_size=batch_size,
    )
        
    print(f"Generated reasoning for {len(results_df)} samples")
    print(f"Results saved to {output_path}")
        
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Generate reasoning for entity matching pairs')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--sample_size', type=int, default=None, 
                        help='Number of samples to process (use all if not specified)')
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    
    client = OpenAI(api_key=args.api_key)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for split in ['train', 'valid', 'test']:
        process_split(split, client, args.dataset_dir, args.output_dir, args.model, args.sample_size, args.batch_size)

if __name__ == "__main__":
    main()