# Twitter Sentiment Analysis - BERT + PyQt GUI

Assignment 03 - NLP Lab, Shifa Tameer-e-Millat University.

## Dataset Source
`dataset/twitter_sentiment.csv` is a cleaned, balanced sample (4,000 positive +
4,000 negative tweets = 8,000 rows) drawn from the **Twitter Sentiment
Analysis Dataset** (BITS Pilani), originally hosted at:
https://github.com/vineetdhanawat/twitter-sentiment-analysis/blob/master/datasets/Sentiment%20Analysis%20Dataset.csv

The original file has ~1.05 million tweets labeled 0 (negative) / 1
(positive). It was cleaned (empty/very short tweets removed), labels mapped
to `negative` / `positive`, and randomly downsampled to a balanced 8,000-row
set so it trains in a reasonable time. Columns:
- `text` — the tweet
- `sentiment` — `positive` or `negative`

If you'd rather use a 3-class (positive/negative/neutral) dataset instead,
another good option is the Twitter Entity Sentiment Analysis dataset on
Kaggle: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
— just drop it in `dataset/` and point `--text_col` / `--label_col` at its
column names when training.

## Project Structure
```
assignment_03_bert_twitter_sentiment/
|-- train_bert.py        # trains BERT, evaluates it, saves model + graphs
|-- app.py                # PyQt5 GUI (loads dataset + saved model, predicts)
|-- requirements.txt
|-- README.md
|-- dataset/
|   |-- twitter_sentiment.csv
|-- saved_bert_model/     # created after training (config.json, model, tokenizer, label_map.json)
|-- report_graphs/        # created after training (loss/accuracy curves, confusion matrix, etc.)
|-- screenshots/
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

Put your dataset CSV at `dataset/twitter_sentiment.csv`, then run:

```bash
python train_bert.py --data dataset/twitter_sentiment.csv --text_col text --label_col sentiment --epochs 3 --batch_size 16
```

Replace `--text_col` / `--label_col` with the actual column names in your CSV
if they're different (e.g. `--text_col tweet --label_col label`).

This will:
- Load and clean the dataset
- Fine-tune `bert-base-uncased` for sentiment classification
- Print accuracy, precision, recall, F1-score, and a full classification report
- Save graphs to `report_graphs/`: `class_distribution.png`, `loss_curve.png`,
  `accuracy_curve.png`, `confusion_matrix.png`, and `metrics.txt`
- Save the trained model + tokenizer + label mapping to `saved_bert_model/`

Training on CPU will be slow. If you have access to Google Colab with a GPU,
run the same script there and then download the `saved_bert_model/` folder.

## 3. Run the GUI

```bash
python app.py
```

Steps inside the GUI:
1. Click **Load Dataset CSV** and select your CSV file.
2. Click **Load BERT Model** and select the `saved_bert_model/` folder.
3. Click any tweet in the table -> the predicted sentiment and confidence
   appear automatically in the right-hand panel.
4. Or type any sentence into the **Manual Input** box and click **Predict**.

## Notes
- If the `saved_bert_model/` folder is too large for GitHub, upload it to
  Google Drive and put the link here instead:
  `<your model download link>`
- Screenshots of the GUI, dataset loading, model loading, and a prediction
  result should be placed in `screenshots/`.
