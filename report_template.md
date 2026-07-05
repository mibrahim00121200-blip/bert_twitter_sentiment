# Short Report - BERT Twitter Sentiment Analysis

**Course:** Natural Language Processing Lab
**Assignment:** 03 - BERT Training on Twitter Sentiment Dataset with PyQt GUI

## 1. Dataset
- Source: <fill in dataset name + link>
- Columns used: text column = `<...>`, label column = `<...>`
- Classes: Positive / Negative / Neutral (or Positive / Negative)
- Total samples: `<...>`, Train/Test split: `<...>% / ...%`

## 2. Model Details
- Base model: `bert-base-uncased`
- Epochs: `<...>`
- Batch size: `<...>`
- Learning rate: `<...>`
- Max sequence length: `<...>`

## 3. Results
Paste values from `report_graphs/metrics.txt`:
- Accuracy: `<...>`
- Precision: `<...>`
- Recall: `<...>`
- F1-score: `<...>`

Insert graphs here:
- `report_graphs/class_distribution.png`
- `report_graphs/loss_curve.png`
- `report_graphs/accuracy_curve.png`
- `report_graphs/confusion_matrix.png`

## 4. Paul's Critical Thinking Standards

**Clarity** - The dataset has two main columns: tweet text and sentiment label
(positive/negative/neutral). The GUI workflow is: load dataset -> load model
-> click tweet -> view prediction, or type a sentence -> click Predict.

**Accuracy** - Report the exact metrics obtained above, including any weak
spots (e.g. low recall on the neutral class), without hiding poor results.

**Precision** - Model: `bert-base-uncased`; train/test split: `<...>`;
epochs: `<...>`; batch size: `<...>`.

**Relevance** - The loss/accuracy curves show whether the model is
overfitting or underfitting; the confusion matrix shows which sentiment
classes get confused with each other; the class distribution graph explains
whether imbalance affected the results.

**Depth** - Explain briefly *why* the model performs well or poorly on
certain tweets, e.g. sarcasm, slang, short/ambiguous tweets, emojis, or class
imbalance can lower accuracy on minority classes.

**Logic** - Confirm the pipeline connects correctly end-to-end: dataset ->
training -> saved model -> GUI load -> GUI prediction, with no broken step.

**Fairness** - Mention any dataset bias (e.g. mostly one sentiment class,
informal Twitter slang, sarcasm not well represented, or dataset skew toward
a specific topic/region/language).

## 5. Conclusion
Summarize overall performance and one or two ideas for improvement (e.g. more
epochs, class balancing, using `bert-base` vs a lighter model like
`distilbert-base-uncased` for faster inference).
