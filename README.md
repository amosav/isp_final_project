# Final Project: CLAP Model with Audio Pretraining

Welcome to the final project repository! This project implements **CLAP (Contrastive Language-Audio Pretraining)** with a focus on audio encoding, audio captioning, and classification tasks using the ESC-50 dataset and Music genres data set.

## Project Overview

In this project, we utilize CLAP, a model designed for Contrastive Language-Audio Pretraining, which includes:

- **Audio Encoder:** CNN-14
- **Text Encoder:** Based on the BERT model

The aim of the project is to achieve high performance in various audio classification and captioning tasks using pretraining and fine-tuning techniques, such as **LoRA (Low-Rank Adaptation)**.

## Key Features

1. **Audio Captioning:** The model generates captions for different sound files, including those from the **ESC-50** dataset.
2. **Fine-Tuning with LoRA:** The model is fine-tuned using LoRA, allowing for efficient adaptation with minimal parameter updates.
3. **Embedding Space Evaluation:** t-SNE visualizations and confusion matrix are used to evaluate the embedding space created by the model.
4. **Data Augmentation:** Techniques like **frequency masking**, **time masking**, and **crop-based augmentations** are used to improve model generalization.

## Performance

### Results on ESC-50:
- **CLAP + LoRA (16,32):** 97.25%
- **CLAP + LoRA (8,16):** 96.75%
- **Baseline CLAP:** 73.25%

## Dataset

The project uses the **ESC-50 dataset**, a collection of 50 different sound categories, including categories like vacuum cleaner, rock music, and more. The dataset has been preprocessed and augmented for training purposes in addition we used an music genres data.

## Contributing

If you would like to contribute to this project, feel free to submit a pull request or open an issue in the repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The **CLAP** model is based on the work done in **Contrastive Language-Audio Pretraining**.
- Thanks to the creators of the **ESC-50** dataset for providing a comprehensive set of environmental sound recordings.
"""
