
import pandas as pd
from transformers import pipeline
from pathlib import Path

# Dataset folder
DATA = Path("dataset")
DATA.mkdir(exist_ok=True)

# Load article
df = pd.read_csv(DATA / "input_text.csv")
text = str(df["text"].dropna().iloc[0])

print("Article loaded successfully!")

# Sentiment Analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

result = sentiment_pipeline(text[:2000])[0]

print("Sentiment:", result["label"])
print("Confidence:", result["score"])

# Save result
sentiment_df = pd.DataFrame([{
    "text": text,
    "sentiment": result["label"],
    "confidence": result["score"]
}])

sentiment_df.to_csv(
    DATA / "sentiment_results.csv",
    index=False
)

print("Sentiment result saved!")


# Text Summarization
print("\nRunning Text Summarization...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Generate short summary
summary_60 = summarizer(
    text[:3000],
    max_length=60,
    min_length=20,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

# Generate longer summary
summary_100 = summarizer(
    text[:3000],
    max_length=100,
    min_length=30,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

print("\nSummary (max_length=60):")
print(summary_60)

print("\nSummary (max_length=100):")
print(summary_100)

# Save summaries
summary_df = pd.DataFrame([
    {"max_length": 60, "summary": summary_60},
    {"max_length": 100, "summary": summary_100}
])

summary_df.to_csv(
    DATA / "summarized_text.csv",
    index=False
)

print("\nSummaries saved successfully!")


# 3. LANGUAGE TRANSLATION
print("\nRunning Translation...")

# English to Hindi
translator_hi = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-hi"
)

# English to French
translator_fr = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

# Use the same English text
translation_text = text[:400]

# Translate into Hindi
hindi_result = translator_hi(
    translation_text,
    truncation=True
)[0]["translation_text"]

# Translate into French
french_result = translator_fr(
    translation_text,
    truncation=True
)[0]["translation_text"]

print("\nHindi Translation:")
print(hindi_result)

print("\nFrench Translation:")
print(french_result)

# Save translation results
translation_df = pd.DataFrame([
    {
        "language": "Hindi",
        "original_text": translation_text,
        "translation": hindi_result
    },
    {
        "language": "French",
        "original_text": translation_text,
        "translation": french_result
    }
])

translation_df.to_csv(
    DATA / "translated_text.csv",
    index=False
)

print("\nTranslations saved successfully!")


# 4. QUESTION ANSWERING
print("\nRunning Question Answering...")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

# Ask a question about the article
question = "What can students use AI tools for?"

# Get the answer from the article
qa_result = qa_pipeline(
    question=question,
    context=text[:3000]
)

print("\nQuestion:", question)
print("Answer:", qa_result["answer"])
print("Confidence:", qa_result["score"])

# Save QA results
qa_df = pd.DataFrame([
    {
        "question": question,
        "answer": qa_result["answer"],
        "confidence": qa_result["score"]
    }
])

qa_df.to_csv(
    DATA / "qa_results.csv",
    index=False
)

print("\nQA results saved successfully!")


# Pipeline Comparison
comparison_df = pd.DataFrame([
    {
        "task": "Sentiment Analysis",
        "method": "DistilBERT",
        "output": result["label"]
    },
    {
        "task": "Summarization",
        "method": "max_length=60",
        "output": summary_60
    },
    {
        "task": "Summarization",
        "method": "max_length=100",
        "output": summary_100
    },
    {
        "task": "Translation",
        "method": "English to Hindi",
        "output": hindi_result
    },
    {
        "task": "Translation",
        "method": "English to French",
        "output": french_result
    },
    {
        "task": "Question Answering",
        "method": "DistilBERT QA",
        "output": qa_result["answer"]
    }
])

comparison_df.to_csv(
    DATA / "pipeline_comparison.csv",
    index=False
)

print("\nAll NLP tasks completed!")
print("All results saved in dataset folder.")
