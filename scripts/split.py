#pip install split-folders

import splitfolders

INPUT_DIR = "imagens/dataset_andressa"
OUTPUT_DIR = "dataset/dataset_andressa"

print("Iniciando divisão do dataset...")

splitfolders.ratio(
    INPUT_DIR, 
    output=OUTPUT_DIR, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None
)

print(f"Divisão concluída! Os dados estão prontos na pasta: {OUTPUT_DIR}")