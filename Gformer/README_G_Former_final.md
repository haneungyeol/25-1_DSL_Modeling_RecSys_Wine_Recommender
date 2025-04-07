
# G-Former with Side Information Implementation

This repository provides a PyTorch implementation of the **G-Former (Graph Transformer for Recommendation)** model integrated with side information, customized specifically for wine recommendation tasks.

## Overview

This implementation:
- Uses BERT embeddings for textual side information (flavor, production details, food pairing).
- Applies dimensionality reduction (PCA) to manage embedding dimensions.
- Incorporates side information into graph embeddings using attention-based fusion.
- Employs edge dropout and reconstruction losses for robustness against data sparsity.

## Requirements

Install the following libraries:

```bash
pip install torch torchvision torchaudio torch-geometric pandas scikit-learn numpy sentence-transformers
```

## Data

Your data should include:
- Wine rating data (`filtered_train_data.csv`, `filtered_val_data.csv`, `filtered_test_data.csv`)
- Wine side information (e.g., flavors, production, food pairing)

Ensure your CSV files are structured correctly.

## Structure

- `G_Former_final.py`: Main Python script containing the model implementation, data preprocessing, training, and evaluation pipelines.

## Usage

Run the script directly:

```bash
python G_Former_final.py
```

Customize parameters (learning rate, epochs, embedding dimensions, dropout rates, etc.) within the script.

## Evaluation Metrics

The implementation evaluates model performance using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Normalized Discounted Cumulative Gain (NDCG@10, NDCG@20)
- Recall@20
- Precision@20

## Citation

If you leverage this implementation, cite the original G-Former paper:

```
@inproceedings{li2023graph,
  title={Graph Transformer for Recommendation},
  author={Li, Chaoliu and Xia, Lianghao and Ren, Xubin and Ye, Yaowen and Xu, Yong and Huang, Chao},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}
```

## License

MIT License. Modify and extend freely to suit your project needs.

---

Customize this README further based on your specific dataset, implementation details, and project requirements.
