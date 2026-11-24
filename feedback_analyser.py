import pandas as pd
import matplotlib.pyplot as plt
import re


def read_feedback():

    try:
        df = pd.read_csv("feedback.csv")

        print("CSV file loaded successfully!")

        return df

    except FileNotFoundError:

        print("Error: CSV file not found!")

        return None

def validate_data(df):

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    valid_rows = (
        df["Rating"].notna() &
        (df["Rating"] >= 1) &
        (df["Rating"] <= 5) &
        (df["Rating"] % 1 == 0)
    )

    invalid_count = (~valid_rows).sum()

    if invalid_count > 0:
        print("Warning:", invalid_count, "invalid entries skipped.")

    df = df[valid_rows].copy()

    df["Rating"] = df["Rating"].astype(int)

    return df

# FUNCTION 3: Display feedback summary
def show_summary(df):

    total = len(df)
    average = df["Rating"].mean()

    print("\n----- FEEDBACK SUMMARY -----")

    print("Total valid feedback:", total)
    print("Average rating:", round(average, 2))

    print("\nRating Frequency:")

    rating_counts = df["Rating"].value_counts()

    for rating in range(1, 6):
        count = rating_counts.get(rating, 0)
        print("Rating", rating, ":", count)

# FUNCTION 4: Sentiment analysis
def sentiment_analysis(df):

    total = len(df)

    if total == 0:
        print("No valid feedback available for sentiment analysis.")
        return

    positive = df["Rating"].isin([4, 5]).sum()
    neutral = (df["Rating"] == 3).sum()
    negative = df["Rating"].isin([1, 2]).sum()

    print("\n----- SENTIMENT ANALYSIS -----")

    print("Positive:", positive, "(",
          round((positive / total) * 100, 2), "%)")

    print("Neutral:", neutral, "(",
          round((neutral / total) * 100, 2), "%)")

    print("Negative:", negative, "(",
          round((negative / total) * 100, 2), "%)")

    # Display sentiment distribution as a chart
    categories = ["Positive", "Neutral", "Negative"]
    counts = [positive, neutral, negative]

    plt.bar(categories, counts)

    plt.title("Student Feedback Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Feedback Entries")

    plt.show()



# FUNCTION 5: Find the top three words
def top_words(df):

    stop_words = ["and", "but", "the", "is", "a", "of", "too"]

    comments = df["Comment"].fillna("").astype(str)

    all_words = re.findall(r"[a-z]+", " ".join(comments).lower())

    word_count = {}

    for word in all_words:
        if word not in stop_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    sorted_words = sorted(
        word_count.items(),
        key=lambda x: (-x[1], x[0])
    )

    print("\n----- TOP THREE WORDS -----")

    for word, count in sorted_words[:3]:
        print(word, ":", count)





df = read_feedback()


if df is not None:
    df = validate_data(df)
    show_summary(df)
    top_words(df)
    sentiment_analysis(df)
