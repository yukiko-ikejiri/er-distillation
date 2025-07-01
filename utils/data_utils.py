import os
import pandas as pd

def df_serializer(data: pd.DataFrame, mode, include_reasoning=False):
    attrs_l = [col for col in data.columns if col.endswith('_l')]
    attrs_r = [col for col in data.columns if col.endswith('_r')]
    attrs = [col[:-2] for col in attrs_l]

    if mode == 'mode1':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        if include_reasoning and 'reasoning_text' in data.columns:
            data['text'] = data.apply(lambda x: 'Record A is <p>' + x['text_l'] + '</p>. Record B is <p>' + x[
            'text_r'] + '</p>. Consider this reasoning: ' + str(x['reasoning_text']) 
            + '</p>. Given the attributes of the two records, are they the same?', axis=1)
        else:
            data['text'] = data.apply(lambda x: 'Record A is <p>' + x['text_l'] + '</p>. Record B is <p>' + x[
            'text_r'] + '</p>. Given the attributes of the two records, are they the same?', axis=1)

    else:
        raise ValueError('Invalid mode')
    return data[['text', 'label']]

def automl_filter(train_df, dataset_dir):
    dataset_name = dataset_dir.split('/')[-1]
    if len(train_df) < 1200:
        print(f'The training set size of {dataset_name} is less than 1200, which will all be kept.', flush=True)
        return train_df
    else:
        print(f'The training set size of {dataset_name} is larger than 1200, we will do down-sampling '
              f'with automl_filter to maximally 1200 pairs.', flush=True)
        automl_data_dir = '/'.join(dataset_dir.split('/')[:-2] + ['automl'] + [dataset_dir.split('/')[-1]])
        train_preds_df = pd.read_csv(os.path.join(automl_data_dir, 'train_preds.csv'))

        train_pos_wrong_preds = train_df[(train_preds_df['prediction']!=train_df['label']) & (train_df['label']==1)]
        train_pos_num = min(400, train_df['label'].sum())
        if len(train_pos_wrong_preds) < train_pos_num:
            train_pos_supply = train_df[(train_preds_df['prediction']==train_df['label']) & (train_df['label']==1)].sample(n=train_pos_num-len(train_pos_wrong_preds), random_state=42)
            train_pos_df = pd.concat([train_pos_wrong_preds, train_pos_supply])
        else:
            train_pos_df = train_pos_wrong_preds.sample(n=train_pos_num, random_state=42)
        train_neg_df = train_df[train_df['label']==0].sample(n=2*train_pos_num, random_state=42)

        filtered_train_df = pd.concat([train_pos_df, train_neg_df]).reset_index(drop=True)

        return filtered_train_df


def read_single_row_data(dataset_name, dataset_dir, output_dir, mode, sample_func='',
                        include_reasoning=False, filter_correct_reasoning=False, print_info=True):
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(dataset_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    if sample_func:
        sample_func = eval(sample_func)
        train_df = sample_func(train_df, dataset_dir)

    if filter_correct_reasoning:
        correct_mask = train_df['answer_label'] == train_df['label']
        train_df = train_df[correct_mask].copy().reset_index(drop=True)

    if include_reasoning:
        experiment_train_path = os.path.join(output_dir, f'{dataset_name}_experiment_train.csv')
        train_df.to_csv(experiment_train_path, index=False)
        print(f"Experiment train data saved to: {experiment_train_path}", flush=True)

    train_df = df_serializer(train_df, mode, include_reasoning)
    valid_df = df_serializer(valid_df, mode, include_reasoning=False)
    test_df = df_serializer(test_df, mode, include_reasoning=False)

    if print_info:
        dataset_name = dataset_dir.split('/')[-1]
        reasoning_info = " with reasoning" if include_reasoning else ""
        print(f"We will use the {mode} partition of the {dataset_name} dataset{reasoning_info}.", flush=True)
        print(f"A train data example(row level) after the serialization is:\n{train_df.iloc[0]['text']}", flush=True)
        print(f"A test data example(row level) after the serialization is:\n{test_df.iloc[0]['text']}", flush=True)

    return train_df, valid_df, test_df