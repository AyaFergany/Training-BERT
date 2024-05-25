# Training-BERT 


**Model Description**

- BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that is pretrained on a large corpus of English data in a self-supervised manner.


**The Project contains the following:**
1. Masked Language Modeling (MLM) using BERT
2. Next Sentence Prediction (NSP) using BERT
3. Fine tunning BERT


1. **The Process of Masked Language Modeling (MLM)**

Masked Language Modeling (MLM) is a key component of the BERT model's training process. Here are the logical steps that we need to follow in code to implement MLM:

1. **Tokenize the Text**: Just like we usually would with transformers, we begin with text tokenization. This process converts the raw text into a format that can be used as input to the model. From tokenization, we will receive three different tensors: `input_ids`, `token_type_ids`, and `attention_mask`. For MLM, the `input_ids` tensor is the most important as it contains a tokenized representation of the text that we will be modifying.
2. **Create a Labels Tensor**: We're training our model here, so we need a labels tensor to calculate loss against and optimize towards. The labels tensor is simply a copy of the `input_ids` tensor.
3. **Mask Tokens in `input_ids`**: Now that we've created a copy of `input_ids` for labels, we can go ahead and mask a random selection of tokens. The BERT paper uses a 15% probability of masking each token during model pre-training, with a few additional rules. We'll use a simplified version of this and assign a 15% probability of each word being masked.
4. **Calculate Loss**: We process the `input_ids` and `labels` tensors through our BERT model and calculate the loss between them both. This loss is used to update the model's weights during training.

By following these steps, we can implement MLM and train the BERT model to better understand the specific style of language in our use-cases.



2. **Next Sentence Prediction**

NSP consists of giving BERT two sentences, sentence A and sentence B. We then say, ‘hey BERT, does sentence B come after sentence A?’ — and BERT says either IsNextSentence or NotNextSentence.

1. Tokenization:

* We tokenize our text, just like we usually would with transformers.
* From tokenization, we will receive three different tensors: `input_ids`, `token_type_ids`, and `attention_mask`.
* For our purposes, the `input_ids` tensor is the most important, as it contains a tokenized representation of our text that we will modify.

2. Create classification label:

* We need to create a label tensor for our classification task.
* The label tensor should contain a 1 for positive examples (i.e., `text` and `text2` are consecutive sentences) and a 0 for negative examples (i.e., `text` and `text2` are not consecutive sentences).

3. Calculate loss:

* Calculate loss — Finally, we get around to calculating our loss. We start by processing our inputs and labels through our model.

4. Prediction

* We may also not need to train our model, and would just like to use the model for inference. In this case, we would have no labels tensor, and we would modify the last part of our code to extract the logits tensor like so:



3. **Fine-Tuning BERT**

BERT is a powerful language model that can be fine-tuned for a variety of downstream tasks. There are two main approaches to fine-tuning BERT:

1. **Fine-Tuning the Core BERT Model**: This approach involves using the same training approach used by Google when training the original BERT model. It allows us to fine-tune BERT to better understand the specific style of language in our use-cases.

We can also add different heads to the BERT model, which gives it new abilities. These are extra layers at the end of the model that modify the outputs for different use-cases. For example, we would use different heads for question-answering or classification.

In this article, we will be focusing on fine-tuning the core BERT model.

2. **Adding Different Heads to the BERT Model**: This approach involves adding task-specific layers on top of the BERT model. For example, we can add a classification layer for text classification tasks or a question-answering layer for extractive question-answering tasks.

**Fine-Tuning the Core BERT Model**

The core of the BERT model is trained using two main methods:

1. **Next Sentence Prediction (NSP)**: This method involves concatenating two sentences as inputs to the model and predicting whether the second sentence is the next sentence in the original text or a random sentence from the corpus.
2. **Masked Language Modeling (MLM)**: This method involves masking 15% of the words in the input sentence and predicting the masked words. This allows the model to learn a bidirectional representation of the sentence.

3. **And our training data** — Meditations dataset

