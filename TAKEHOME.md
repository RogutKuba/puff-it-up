# Solutions Engineer Take Home: Late Interaction

Many customers have expressed interest in improving search quality with late interaction techniques like ColBERT, but they are unsure about how to implement them effectively on turbopuffer.

In this take home exercise, you’ll write a guide about how to implement late interaction using turbopuffer.

# Expectations

- Spend 4-5 hours preparing the deliverables below.
- Be ready to walk your interviewers through what you’ve produced.

# Deliverables

Author the deliverables in any tool you like (Jupyter notebook, Google Doc, raw markdown, etc.).

### **D1. User-facing guide**

This guide should be similar to the other guides on the turbopuffer website, like [hybrid search](https://turbopuffer.com/docs/hybrid) and the [FTS guide](https://turbopuffer.com/docs/fts). 

Your aim should be to produce something that, were you to join turbopuffer, you could ship in your first few days on the job. Use only the features that exist in turbopuffer today.

You should cover:

1. **Introduction**
What is late interaction? When should you consider using late interaction?
    
    Assume your audience is an engineer who is a skilled programmer but has limited search expertise.
    
2. **Implementation**
Provide sample code showing how to implement late interaction on turbopuffer.
    
    Feel free to use any sample dataset you wish. Some datasets worth considering are:
    
    - [Quora Duplicates](https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
    - [Amazon Reviews ‘23](https://amazon-reviews-2023.github.io)
    - [Reddit comments](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits)
3. **Analysis**
Compare late interaction against traditional dense retrieval on:
    - Result quality
    - Latency
    - Cost (in $)

### **D2. Internal roadmap suggestions**

Suggest how turbopuffer could evolve to better support late interaction natively. What API changes would you propose?