# Final Project: Introduction to Speech and Audio Processing

Our final model architecture is:

Auto-tune CLAP mmodel with LORA 

## Performance Metrics

- **Validation Set**:  
  - Word Error Rate (WER): 0.034  
  - Character Error Rate (CER): 0.016
- **Test Set**:  
  - Word Error Rate (WER): 0.059  
  - Character Error Rate (CER): 0.032

## Data

We used the _____ dataset to address our task. Below is a summary of the data splits used in our experiments:

| Dataset   | # Samples | # Different Words | # Different Speakers | # Words not in Train Set | # Speakers not in Train Set |
|-----------|-----------|-------------------|----------------------|--------------------------|-----------------------------|
| Train set | 853       | 98                | 74                   | -                        | -                           |
| Val set   | 95        | 67                | 56                   | 1                        | 10                          |
| Test set  | 130       | 69                | 10                   | 2                        | 1                           |

### Key Characteristics

- **Word Set**:  
  Each data split contains the full set of English characters.

- **Word Vocabulary**:  
  As shown in Table 1, the word vocabulary is relatively small, containing mostly short words, and each dataset split includes unseen words in the validation and test sets.  

## Visuals

![Table 1: Comparison between different data splits](./path-to-your-image.png)

---

For further details on how these splits were constructed, refer to the image above.

