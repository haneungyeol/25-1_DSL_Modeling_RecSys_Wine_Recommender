
# KGAT with Side Feature Integration for Wine Recommendation

This repository provides an extended implementation of the **Knowledge Graph Attention Network (KGAT)** model, customized for wine recommendation tasks. It integrates structured side information (e.g., flavor, region, winery) into the graph using feature selection and preprocessing techniques.

## Overview

This implementation:
- Transforms structured CSV data into a knowledge graph format.
- Maps wine items and their attributes into item and relation IDs.
- Applies **feature selection** using RandomForest to identify important attributes.
- Trains a KGAT model using user-item interaction data with knowledge graph attention.
- Evaluates performance using standard recommendation metrics.

## Requirements

Install the following libraries:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Data

Ensure your data directory includes:
- `wine_info_processed_quintiles.csv`: Preprocessed wine metadata including features like flavor, price quantile, region, and winery.
- Train/test/validation user-item interaction data if applicable.

## Structure

```
.
├── KGAT/
│   ├── main.py                # Main training script
│   ├── test.py                # Evaluation script
│   ├── bpr_dataset.py         # BPR dataset loader
│   ├── cf_data.py             # Collaborative filtering utilities
│   ├── kgat_config.py         # Model configuration
│   ├── kgat_data.py           # KG data constructor
│   ├── kgat_model.py          # KGAT model definition
│   └── evaluation.py          # Evaluation metrics
│
├── utils/
│   ├── feature_selection.py   # Feature importance via RandomForest
│   └── preprocessor.py        # Convert CSV into KG-compatible formats
│
├── data/
│   └── wine_info_processed_quintiles.csv
```

## Key Enhancements over Standard KGAT

| Aspect          | Standard KGAT                | This Implementation                        |
|-----------------|------------------------------|---------------------------------------------|
| Input Data      | Entity triples                | Structured side info from wine CSV          |
| Preprocessing   | ID mapping                    | Attribute-based relation mapping            |
| Feature Usage   | N/A                           | Feature importance selection via RandomForest |
| Interpretability| Limited                       | Visualized and ranked feature importance    |
| Extensibility   | Manual KG expansion           | Auto-generated KG from CSV attributes       |

## How It Works

1. **Preprocess attributes** into item/relation mappings (`utils/preprocessor.py`)
2. **Evaluate feature importance** using RandomForest (`utils/feature_selection.py`)
3. **Train KGAT** using the enhanced knowledge graph (`KGAT/main.py`)
4. **Evaluate** the trained model (`KGAT/test.py`)

## Evaluation Metrics

- Precision@K
- Recall@K
- NDCG@K
- Hit Ratio

All metrics are implemented in `KGAT/evaluation.py`.

## Usage

```bash
# Step 1: Preprocess
python utils/preprocessor.py

# Step 2: (Optional) Feature Selection
python utils/feature_selection.py

# Step 3: Train KGAT
python KGAT/main.py

# Step 4: Evaluate
python KGAT/test.py
```

## License

MIT License. Feel free to modify and extend for your use cases.
