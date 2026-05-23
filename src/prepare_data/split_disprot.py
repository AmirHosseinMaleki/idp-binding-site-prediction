import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_config, get_dataset_path

cfg = load_config()

# df = pd.read_csv('/home/malekia/idp-binding-site-prediction/data/training/protein_binding_training_data.tsv', sep='\t')
df = pd.read_csv(get_dataset_path(cfg, "disprot", "protein_training_tsv"), sep='\t')


train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df.to_csv('protein_binding_train.tsv', sep='\t', index=False)
val_df.to_csv('protein_binding_val.tsv', sep='\t', index=False)
test_df.to_csv('protein_binding_test.tsv', sep='\t', index=False)

print(f"Total: {len(df)}")
print(f"Train: {len(train_df)} (70%)")
print(f"Val: {len(val_df)} (15%)")
print(f"Test: {len(test_df)} (15%)")