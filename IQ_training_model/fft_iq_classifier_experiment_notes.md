# FFT-Based IQ Classifier Experiment Notes

## 1. Objective

The goal of this experiment was to build a small supervised classifier that can classify UAV/drone RF signals directly from IQ files using FFT-based frequency-domain features.

The pipeline tested was:

```text
IQ file
→ split into fixed-size chunks
→ apply FFT to each chunk
→ convert FFT output to log-magnitude spectrum
→ train a lightweight 1D CNN
→ evaluate using accuracy, precision, recall, F1-score, and confusion matrix
```

This experiment was designed as a first step to understand whether FFT features alone contain enough information to separate UAV classes.

---

## 2. Dataset Setup

Three UAV classes were used:

```text
YUNZHUO-H12
YUNZHUO-H16
YUNZHUO-H30
```

The IQ files were stored in this structure:

```text
dataset_iq/
├── YUNZHUO-H12/
├── YUNZHUO-H16/
└── YUNZHUO-H30/
```

A balanced dataset was also created using one IQ file per class:

```text
dataset_iq_balanced/
├── YUNZHUO-H12/
├── YUNZHUO-H16/
└── YUNZHUO-H30/
```

Each `.iq` file was large, around 200 MB to 800 MB, so each file contained many usable FFT chunks.

---

## 3. IQ File Format Issue

The original script assumed each `.iq` file was raw `float32` interleaved IQ:

```text
I0, Q0, I1, Q1, I2, Q2, ...
```

Some files had sizes that were not exact multiples of 4 or 8 bytes, causing this error:

```text
ValueError: Size of available data is not a multiple of the data-type size.
```

The fix was to ignore incomplete trailing bytes and only read the usable number of `float32` values. This allowed partial `.iq` files to be used safely.

---

## 4. Model and Feature Pipeline

Each IQ chunk was converted into a frequency-domain feature vector.

### FFT feature extraction

For each chunk:

```text
1. Load complex IQ samples
2. Remove DC offset
3. Normalize signal power
4. Apply Hann window
5. Compute FFT
6. Apply FFT shift
7. Convert to log-magnitude spectrum
8. Standardize the feature vector
```

The model input was:

```text
[1, FFT_length]
```

where the single channel contains the FFT log-magnitude values.

### Model

A lightweight 1D CNN was used:

```text
Conv1D → BatchNorm → ReLU → MaxPool
Conv1D → BatchNorm → ReLU → MaxPool
Conv1D → BatchNorm → ReLU
AdaptiveAvgPool
Linear classifier
```

This model is much smaller than ResNet or ViT and can run on CPU.

---

## 5. Experiment 1: Regular Dataset

### Command

```bash
python3 fft_iq_classifier.py \
  --data-root dataset_iq \
  --classes YUNZHUO-H12 YUNZHUO-H16 YUNZHUO-H30 \
  --chunk-size 4096 \
  --nfft 4096 \
  --max-chunks-per-file 200 \
  --epochs 5 \
  --batch-size 64
```

### Result

```text
Train chunks: 1280
Valid chunks: 320
Best valid accuracy: 70.00%
```

### Classification report

```text
              precision    recall  f1-score   support

 YUNZHUO-H12       0.91      0.53      0.67       160
 YUNZHUO-H16       0.85      0.72      0.78        40
 YUNZHUO-H30       0.57      0.92      0.70       120

    accuracy                           0.70       320
   macro avg       0.78      0.72      0.72       320
weighted avg       0.78      0.70      0.70       320
```

### Interpretation

This result was promising because 70% accuracy is clearly above the 33% random baseline for three classes.

However, the dataset was imbalanced:

```text
YUNZHUO-H12: more chunks
YUNZHUO-H16: fewer chunks
YUNZHUO-H30: more chunks
```

Because of this imbalance, the regular dataset result was not the best comparison.

---

## 6. Experiment 2: Balanced Dataset, 200 Chunks per Class

### Command

```bash
python3 fft_iq_classifier.py \
  --data-root dataset_iq_balanced \
  --classes YUNZHUO-H12 YUNZHUO-H16 YUNZHUO-H30 \
  --chunk-size 4096 \
  --nfft 4096 \
  --max-chunks-per-file 200 \
  --epochs 5 \
  --batch-size 64 \
  --output-dir outputs_fft_balanced
```

### Result

```text
Accuracy: 58%
Macro F1: 56%
Validation support: 40 chunks per class
```

### Classification report

```text
              precision    recall  f1-score   support

 YUNZHUO-H12       0.73      0.28      0.40        40
 YUNZHUO-H16       0.64      0.72      0.68        40
 YUNZHUO-H30       0.50      0.75      0.60        40

    accuracy                           0.58       120
   macro avg       0.63      0.58      0.56       120
weighted avg       0.63      0.58      0.56       120
```

### Interpretation

This result was more honest than the regular dataset result because each class had equal support.

The model still performed above random guessing, but the performance was unstable because only 200 chunks per class were used.

The weakest class was:

```text
YUNZHUO-H12
recall = 0.28
```

This means many real H12 chunks were misclassified as another class.

---

## 7. Experiment 3: Balanced Dataset, 1000 Chunks per Class, FFT Size 4096

### Command

```bash
python3 fft_iq_classifier.py \
  --data-root dataset_iq_balanced \
  --classes YUNZHUO-H12 YUNZHUO-H16 YUNZHUO-H30 \
  --chunk-size 4096 \
  --nfft 4096 \
  --max-chunks-per-file 1000 \
  --epochs 10 \
  --batch-size 64 \
  --output-dir outputs_fft_balanced_1000
```

### Result

```text
Train chunks: 2400
Valid chunks: 600
Best valid accuracy: 81.83%
```

### Classification report

```text
              precision    recall  f1-score   support

 YUNZHUO-H12       0.85      0.75      0.80       200
 YUNZHUO-H16       0.86      0.98      0.92       200
 YUNZHUO-H30       0.74      0.72      0.73       200

    accuracy                           0.82       600
   macro avg       0.82      0.82      0.82       600
weighted avg       0.82      0.82      0.82       600
```

### Confusion matrix

```text
True \ Predicted      H12    H16    H30

YUNZHUO-H12          150      1     49
YUNZHUO-H16            1    196      3
YUNZHUO-H30           25     30    145
```

### Interpretation

This was the strongest 4096 FFT result.

Key observations:

```text
YUNZHUO-H16 was the easiest class.
YUNZHUO-H16 recall = 0.98
YUNZHUO-H16 F1 = 0.92
```

The main confusion was between:

```text
YUNZHUO-H12 ↔ YUNZHUO-H30
```

H12 was often predicted as H30:

```text
True H12 → predicted H30: 49 / 200
```

H30 was also sometimes confused with H12 and H16:

```text
True H30 → predicted H12: 25 / 200
True H30 → predicted H16: 30 / 200
```

This suggests that single-window FFT features can separate the classes reasonably well, but H12 and H30 may have similar frequency-domain patterns.

---

## 8. Experiment 4: Balanced Dataset, 1000 Chunks per Class, FFT Size 8192

### Command

```bash
python3 fft_iq_classifier.py \
  --data-root dataset_iq_balanced \
  --classes YUNZHUO-H12 YUNZHUO-H16 YUNZHUO-H30 \
  --chunk-size 8192 \
  --nfft 8192 \
  --max-chunks-per-file 1000 \
  --epochs 10 \
  --batch-size 32 \
  --output-dir outputs_fft_balanced_8192
```

### Result

```text
Train chunks: 2400
Valid chunks: 600
Best valid accuracy: 82.67%
```

### Classification report

```text
              precision    recall  f1-score   support

 YUNZHUO-H12       0.81      0.83      0.82       200
 YUNZHUO-H16       0.98      0.79      0.87       200
 YUNZHUO-H30       0.73      0.85      0.79       200

    accuracy                           0.83       600
   macro avg       0.84      0.83      0.83       600
weighted avg       0.84      0.83      0.83       600
```

### Interpretation

The 8192 FFT result was slightly better than the 4096 FFT result:

```text
4096 FFT accuracy: 81.83%
8192 FFT accuracy: 82.67%
```

The improvement was small, so it is not enough to claim that 8192 is clearly better. A safer conclusion is:

```text
Increasing FFT size from 4096 to 8192 gave a small improvement, suggesting that higher frequency resolution may help slightly, but the difference is not conclusive.
```

---

## 9. Result Summary

| Experiment | FFT Size | Chunks per Class | Balanced? | Accuracy | Macro F1 |
|---|---:|---:|---|---:|---:|
| Regular dataset | 4096 | mixed | No | 70.00% | 0.72 |
| Balanced small | 4096 | 200 | Yes | 58.00% | 0.56 |
| Balanced larger | 4096 | 1000 | Yes | 81.83% | 0.82 |
| Balanced larger | 8192 | 1000 | Yes | 82.67% | 0.83 |

---

## 10. Main Conclusion

A lightweight FFT-based 1D CNN was able to classify three UAV RF signal classes using only log-magnitude FFT features.

The best balanced chunk-level result was:

```text
82.67% validation accuracy
0.83 macro F1
```

This suggests that FFT-domain features contain useful discriminative information for UAV RF classification.

However, the improvement from FFT size 4096 to 8192 was small, which suggests that simply increasing frequency resolution may not fully solve class confusion. The next improvement may require using time-frequency features such as STFT or spectrograms.

---

## 11. Limitation

The main limitation is that validation was performed at the chunk level.

Because chunks from the same IQ recording can appear in both training and validation sets, the result should be treated as an initial feasibility test rather than a final recording-independent benchmark.

A stronger future evaluation should use:

```text
Train recordings and validation recordings that are completely separate
```

This would test whether the model generalizes to unseen recordings rather than only unseen chunks from the same recording.

---

## 12. Next Steps

Recommended next experiments:

```text
1. Create a normalized confusion matrix for easier visualization.
2. Try STFT or spectrogram features from the IQ files.
3. Compare FFT-only 1D CNN vs spectrogram-based CNN.
4. Download more recordings for each UAV class.
5. Perform recording-level train/validation/test split.
6. Try unsupervised clustering using FFT feature embeddings.
```

A good next pipeline would be:

```text
IQ file
→ STFT
→ spectrogram image or time-frequency tensor
→ CNN classifier
→ compare with FFT-only classifier
```

---

## 13. Short Research Note Version

A small supervised FFT-based 1D CNN was trained on IQ chunks from three YunZhuo UAV classes. Each IQ chunk was converted into a log-magnitude FFT vector before classification. In the balanced setup with 1000 chunks per class, the model achieved 81.83% accuracy using a 4096-point FFT and 82.67% accuracy using an 8192-point FFT. These results suggest that FFT-domain features contain useful class-discriminative information for UAV RF signals. However, because validation was performed at the chunk level rather than the recording level, the results should be interpreted as an initial feasibility test rather than a final benchmark. Future work should evaluate recording-level splits and compare FFT-only features with STFT/spectrogram-based models.
