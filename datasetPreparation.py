import pandas as pd
from textblob import TextBlob

# Load the dataset
dataset_path = 'data/drugsComTrain_raw.csv'
dataset = pd.read_csv(dataset_path)

# Perform sentiment analysis and add the 'sentiment' column
def get_sentiment(review):
    analysis = TextBlob(str(review))
    # Assign sentiment polarity as 'positive', 'negative', or 'neutral'
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

dataset['sentiment'] = dataset['review'].apply(get_sentiment)

# Save the updated dataset to a new CSV file
updated_dataset_path = 'drugsComTrain_with_sentiment.csv'
dataset.to_csv(updated_dataset_path, index=False)

print("Sentiment analysis completed and saved to", updated_dataset_path)