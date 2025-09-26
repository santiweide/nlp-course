import pandas as pd
from datasets import load_dataset

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

dataset = load_dataset("coastalcph/tydi_xor_rc")
df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()

common_questions = pd.merge(df_train, df_val, on='question', how='inner')['question'].unique()

if len(common_questions) != 0:
    
    train_common_df = df_train[df_train['question'].isin(common_questions)]
    val_common_df = df_val[df_val['question'].isin(common_questions)]

    comparison_df = pd.merge(
        train_common_df,
        val_common_df,
        on='question',
        how='inner',
        suffixes=('_train', '_val')
    )
    
    ko_comparison_df = comparison_df[
        (comparison_df['lang_train'] == 'ko') & 
        (comparison_df['lang_val'] == 'ko') &
        (comparison_df['answerable_train'] != comparison_df['answerable_val'])
    ].copy()

    ar_comparison_df = comparison_df[
        (comparison_df['lang_train'] == 'ar') & 
        (comparison_df['lang_val'] == 'ar') &
        (comparison_df['answerable_train'] != comparison_df['answerable_val'])
    ].copy()

    te_comparison_df = comparison_df[
        (comparison_df['lang_train'] == 'te') & 
        (comparison_df['lang_val'] == 'te') &
        (comparison_df['answerable_train'] != comparison_df['answerable_val'])
    ].copy()

    display_columns = [
        'question',
        'context_train',
        'context_val',
        'answer_train',
        'answer_val',
        'answerable_train',
        'answerable_val'
    ]
    
    final_display_columns = [col for col in display_columns if col in ko_comparison_df.columns]
    
    print(ko_comparison_df[final_display_columns])
    print(ar_comparison_df[final_display_columns])
    print(len(te_comparison_df[final_display_columns]))
    # print(te_comparison_df[final_display_columns])

    print(te_comparison_df[final_display_columns].head(1)['question'])
    print(te_comparison_df[final_display_columns].head(1)['context_train'])
    print(te_comparison_df[final_display_columns].head(1)['context_val'])
    print(te_comparison_df[final_display_columns].head(1)['answer_train'])
    print(te_comparison_df[final_display_columns].head(1)['answer_val'])
