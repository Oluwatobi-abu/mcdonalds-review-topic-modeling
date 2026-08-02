# Topic Modelling of McDonald's Store Reviews

LDA topic modelling of ~33,000 McDonald's Google Maps store reviews to identify the main
themes customers discuss, and translate them into business recommendations.

## Contents

```
├── data/
│   └── McDonald_s_Reviews.csv          # raw dataset (Kaggle)
├── notebook/
│   ├── mcdonalds_topic_modeling.ipynb  # full analysis notebook (executed, outputs saved)
│   └── mcdonalds_topic_modeling.py     # jupytext source (same content, plain-text/diffable)
├── outputs/
│   ├── fig1_rating_distribution.png
│   ├── fig2_review_length.png
│   ├── fig3_top_stores.png
│   ├── fig4_coherence_scores.png
│   ├── fig5_topic_volume.png
│   ├── fig6_avg_rating_by_topic.png
│   ├── mcdonalds_lda_visualization.html   # interactive pyLDAvis topic map
│   ├── reviews_with_topics.csv            # every review + its assigned dominant topic
│   └── topic_summary.csv                  # topic names + top words
├── report/
│   └── McDonalds_Topic_Modeling_Report.docx   # 2-page written summary
└── requirements.txt
```

## Dataset

[McDonald's Store Reviews](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)
(Kaggle) — 33,396 Google Maps reviews across 40 US McDonald's locations, including review
text, star rating, store address/coordinates, and relative review date.

## Methodology summary

1. **EDA** — reviewed dataset size, missing values, rating distribution, and review length.
2. **Preprocessing** — lowercase, strip punctuation/numbers, tokenise, remove stop words,
   lemmatise (see notebook Section 2 for the rationale behind each step).
3. **LDA modelling** — built a Gensim dictionary/corpus, scanned topic counts k=4–10 using
   C_v coherence, and selected **k=4** (highest and most interpretable coherence score).
4. **Interpretation** — assigned each review its dominant topic, then compared topics by
   review volume and average star rating to separate complaint-driven themes from
   positive-experience themes.
5. **Recommendations** — five concrete, topic-grounded recommendations for management.

## Key result

| Topic | Avg. rating | Reviews |
|---|---|---|
| Drive-Thru & Wait Time Experience | 2.04★ | 10,080 |
| Staff, Customer Service & Store Hours | 2.92★ | 9,359 |
| Menu Item Quality & Freshness | 4.09★ | 3,876 |
| Positive Overall Service Experience | 4.09★ | 9,547 |

**Operational speed and staff interaction — not the food itself — drive the large majority
of negative reviews.**

## Reproducing

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt punkt_tab
jupyter execute notebook/mcdonalds_topic_modeling.ipynb --output=notebook/mcdonalds_topic_modeling.ipynb
```

The pipeline pins single-threaded BLAS (`OMP_NUM_THREADS=1`, etc.) inside the notebook so
results are fully reproducible run-to-run.
