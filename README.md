
# Reddit Political Leaning Classifier

> A deep learning NLP project using TensorFlow to classify political ideology from Reddit comments — designed to showcase user modeling from behavioral data.

---

## Overview

This project demonstrates the ability to infer **latent user attributes** (political leaning: liberal or conservative) from natural language. 

- Scrape Reddit discussions from political subreddits
- Preprocess them using modern NLP techniques
- Train a bidirectional LSTM model using TensorFlow
- Balance biased data using `class_weight`
- Evaluate accuracy and visualize learning curves

---

## Tech Stack

| Component         | Library/Tool        |
|------------------|---------------------|
| Data scraping     | `PRAW` (Reddit API) |
| Preprocessing     | `spaCy` (lemmatization, stopword removal) |
| Modeling          | `TensorFlow 2` + Keras |
| Tokenization      | `TextVectorization` |
| Balancing         | `class_weight` from `sklearn` |
| Evaluation        | Accuracy, LIME explanations |
| Environment       | Google Colab (GPU-accelerated) |

---

## File Structure

```
├── reddit_scraper.ipynb         # Scrapes and preprocesses data from 4 subreddits
├── reddit_comments_clean.csv    # Final cleaned dataset (comment text + label)
├── lstm_classifier.ipynb        # TensorFlow LSTM model with training and prediction
├── interpretability.ipynb       # (Optional) Explains model decisions using LIME
├── README.md                    # You're here!
```

---

## Model Summary

- Architecture: `Embedding → BiLSTM → Dense + Dropout → Sigmoid`
- Achieves ~**75% validation accuracy** on real Reddit data
- Balanced conservative/liberal data via `class_weight`
- Accepts user-inputted sentences for real-time inference

---

## Example Use

```python
user_input = "We need stronger border security and less government spending."
predict_user_input(user_input)

# Output:
# Prediction: Conservative
# Confidence: 91.32%
```

---


---

## 📌 Future Work

- Use LIME or Integrated Gradients for word-level explainability
- Getting context data for each poster of the user
- Experiment with more models

---

## 🧠 Author

**Kenton Lee** — Data Engineer + M.S. in Data Science (Johns Hopkins)  
Focused on modeling complex user behaviors through clean, production-ready machine learning pipelines.
