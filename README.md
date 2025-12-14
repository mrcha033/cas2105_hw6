# AG News Headline Classification

Lightweight AI pipeline comparing a hand-written keyword baseline to a MiniLM embedding + logistic regression classifier on AG News headlines. All experiments run locally on CPU in under a minute using small dataset subsets.

## Task and Motivation
- **Task:** Assign one of four AG News topics to a headline: World, Sports, Business, Sci/Tech.
- **Motivation:** Topic tagging supports downstream feed routing and analytics; this small setup highlights the gap between rules and modern embeddings.
- **Success criteria:** Higher accuracy and macro-F1 than a keyword rule baseline on a held-out test split.

## Dataset
- Source: Hugging Face `ag_news`.
- Split: 2,000 train / 500 test sampled with seed 42.
- Preprocessing: lowercase + strip whitespace (no other cleaning).

## Methods
- **Naive baseline:** Keyword counts per class; tie/zero-hit falls back to the majority training label.
- **AI pipeline:** SentenceTransformer `all-MiniLM-L6-v2` embeddings → multinomial logistic regression (`C=4.0`, `max_iter=1000`, `n_jobs=-1`). Uses GPU if available.

## Results
| Method                     | Accuracy | Macro-F1 |
|----------------------------|----------|----------|
| Naive keyword baseline     | 0.460    | 0.447    |
| MiniLM + logistic regression | 0.850  | 0.852    |

### Qualitative differences
- “paris tourists search for key to `da vinci code` mystery” — true: World; baseline: Sports; pipeline: World  
- “profit plunges at international game tech” — true: Business; baseline: Sports; pipeline: Business  
- “general mills goes whole grains” — true: Business; baseline: Sports; pipeline: Business  

## Reflection
- Baseline is fast but brittle; collapses to majority when cue words are missing.
- MiniLM embeddings deliver a large jump in accuracy/F1 and handle implicit cues better.
- Common errors: Business vs. World ambiguity; Sci/Tech headlines that read like market news.
- Future tweaks: small hyperparameter sweep, class rebalancing, or a prompt-based zero-shot classifier to see if marginal gains justify extra compute.

## Reproducibility
- Environment: see `requirements.txt`.
- Run notebook end-to-end:  
  ```bash
  jupyter nbconvert --to notebook --execute --inplace notebooks/news_topic_pipeline.ipynb
  ```
