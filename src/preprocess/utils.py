from typing import Optional

import pandas as pd


def extract_prompts_csv(
    csv_path: str,
    feature: str | None = None,
    prefix_clean: Optional[str] = None,
    suffix_clean: Optional[str] = None,
    prefix_corrupted: Optional[str] = None,
    suffix_corrupted: Optional[str] = None,
):
    df = pd.read_csv(csv_path)
    df_features = df
    if feature is not None:
        df_features = df_features[df_features["original_feature"].isin([feature])]
    clean_prompts = df_features["clean_prompt"].tolist()
    corrupted_prompts = df_features["corrupted_prompt"].tolist()

    prefix_clean = prefix_clean or ""
    suffix_clean = suffix_clean or ""
    clean_prompts = [f"{prefix_clean}{prompt}{suffix_clean}" for prompt in clean_prompts]
    prefix_corrupted = prefix_corrupted or ""
    suffix_corrupted = suffix_corrupted or ""
    corrupted_prompts = [f"{prefix_corrupted}{prompt}{suffix_corrupted}" for prompt in corrupted_prompts]

    return clean_prompts, corrupted_prompts
