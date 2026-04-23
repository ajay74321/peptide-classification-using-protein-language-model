# 1. Importing the Libraries

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import StandardScaler


# 2. Loading embedding files of ESM amd protT5 model

train_esm = pd.read_csv("train_esm_embeddings.csv")
test_esm = pd.read_csv("test_esm_embeddings.csv")

train_t5 = pd.read_csv("train_prott5_embeddings.csv")
test_t5 = pd.read_csv("test_prott5_embeddings.csv")


# 3. Preparing the labels

train_esm["label"] = train_esm["label"].replace({-1:0, 1:1})
y = train_esm["label"]


# 4. Extracting the embeddings from the features

X_esm = train_esm.drop(columns=["sequence","label"], errors="ignore")
X_test_esm = test_esm.drop(columns=["sequence"], errors="ignore")

X_t5 = train_t5.drop(columns=["sequence","label"], errors="ignore")
X_test_t5 = test_t5.drop(columns=["sequence"], errors="ignore")


# Renaming the ProtT5 columns to avoid the collisions
X_t5.columns = [f"t5_{c}" for c in X_t5.columns]
X_test_t5.columns = [f"t5_{c}" for c in X_test_t5.columns]


# 5. Combining both the embedding files ( ESM + ProtT5 )

X = pd.concat([X_esm, X_t5], axis=1)
X_test = pd.concat([X_test_esm, X_test_t5], axis=1)


# 6. Adding sequence length feature

train_esm["seq_len"] = train_esm["sequence"].apply(len)
test_esm["seq_len"] = test_esm["sequence"].apply(len)

X["seq_len"] = train_esm["seq_len"]
X_test["seq_len"] = test_esm["seq_len"]


# 7. Standardizing the embedding files

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns
)

# 8. Creating training dataframe

train_df = X_scaled.copy()
train_df["label"] = y

# 9. Training AutoGluon ensemble models with multiple seed values

seeds = [42, 2024, 7, 99, 123]

predictions = []

for seed in seeds:

    print(f"\nTraining model with seed {seed}\n")

    predictor = TabularPredictor(
        label="label",
        eval_metric="roc_auc"
    ).fit(
        train_df,
        presets="good_quality",
        num_bag_folds=10,
        num_stack_levels=1,
        # random_seed=seed,  # Removed as it's not a valid argument for .fit()
        time_limit=3600
    )

    preds = predictor.predict_proba(X_test_scaled)[1]

    predictions.append(preds)


# 10. Ensemble predictions of overall models

final_pred = np.mean(predictions, axis=0)


# 11. Creating final submission file

original_test_data = pd.read_csv("test.csv")

submission = pd.DataFrame({
    "# ID": original_test_data["# ID"],
    "label": final_pred
})


# 12. Save submission

output_path = "submission_file.csv" # Changed to save in the current working directory

submission.to_csv(output_path, index=False)

print(f"\nSubmission saved at: {output_path}")
