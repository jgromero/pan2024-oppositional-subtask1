# Repository for the Oppositional thinking analysis task
See task description at [Oppositional thinking analysis: Conspiracy theories vs critical thinking narratives](https://pan.webis.de/clef24/pan24-web/oppositional-thinking-analysis.html).

## Summary
The proposed approach aims to improve model generalization and reduce bias by anonymizing named entities during preprocessing. Two binary text classification models for Spanish and English were developed using sentence embeddings and feed-forward neural
networks trained on an 8,000-message dataset (4,000 messages per language). Then, two modified models were trained with the same neural network architecture but with named entities replaced by type placeholders. Performance metrics showed that the modified models were competitive with other submissions, achieving MCC scores of 0.797 for English and 0.672 for Spanish.

## Code and results
Submitted results were created with _final_model_xx_replace.h5_. 

The script [check-inferences](https://github.com/jgromero/pan2024-oppositional-subtask1/blob/6cae26248519b00e5209a4532ab52beffa6f7371/check-inferences.py) requires [embeddings_utils.py](https://github.com/openai/openai-python/blob/release-v0.28.0/openai/embeddings_utils.py) and proper configuration of the Open AI API key.
