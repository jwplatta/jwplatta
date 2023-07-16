---
layout: post
title:  "Exploring Textual Entailment: Naive Bayes, Neural Network, and Text-Davinci"
date:   2023-07-15 10:00:00 -0500
categories: machine-learning nlp
---

During my time at Georgia Tech, one of the datasets I enjoyed working on for projects was the Stanford Natural Language Inference (SNLI) dataset. This particular dataset consists of 570k sentence pairs, each labeled with an inferential relationship: neutral, contradiction, or entailment. Although we may often take for granted our own ability to recognize these inferential connections (even when our judgment fails us), the task of enabling computers to discern such relationships is challenging. For instance, any proficient English speaker would intuitively recognizes that "A man inspects the uniform of a figure in some East Asian country." contradicts "The man is sleeping". Yet training a computer to do the same requires significant effort. Since this task is so fundamental to using language, the capacity for NLP systems to accurately identify inferential relationships is highly valuable and has widespread application in many domains and other NLP tasks. The below experiments compare OpenAI's `text-davinci-002` performance on the textual entailment task with Naive Bayes classifier and a simple neural network with one hidden layer.

## Model Design

The models compared in the experiments are a Naive Bayes classifier, a neural network, and OpenAI's `text-davinci-002`. The comparison is not a perfect apples to apples since each model was operating on a slightly different set of features.

I used the common TF-IDF technique to vectorize the sentence pairs to build the feature set for the Naive Bayes model. TF-IDF, which is often used for text classification, scores the importance of terms based on their frequency within a specific document and their inverse frequency across the entire corpus. The underlying assumption of the Naive Bayes model is that each pair of words in the TF-IDF vectorization is conditionally independent given the inferential relationship.  Although this assumption may oversimplify the textual entailment task, the model is a good benchmark against which I expected both OpenAI and the neural network to outperform. Only the top the top 2000 features identified by the TF-IDF vectorizer are used in order to reduce noise and emphasize the most important features.

The neural network model is more complex. It consists of an input layer, one hidden layer, and an output layer, each separated by a Rectified Linear Unit (ReLU) activation function.
```python
self.module_list = [
	nn.Linear(768, 50),
	nn.ReLU(),
	nn.Linear(50, 50),
	nn.ReLU(),
	nn.Linear(50, 3)
]
```
To vectorize the text inputs for the neural network, I used the BERT model to generate word embeddings. Preprocessing the text had several steps. First, the text is tokenized, i.e. broken into an array of subwords for the BERT model. Next, token type IDs are created to differentiate between the two sentences. Additionally, token IDs, or unique identifiers from the BERT model's vocabulary, are generated to numerically represent the text. Finally, an attention mask is created to indicate which tokens the model should prioritize during input processing. Sequences longer than the specified maximum length of 128 were truncated, while shorter sequences were padded with zeros.
```
# sentence pair sequence after tokenization
['[CLS]', 'this', 'church', 'choir', 'sings', 'to', 'the', 'masses', 'as', 'they', 'sing', 'joy', '##ous', 'songs', 'from', 'the', 'book', 'at', 'a', 'church', '.', '[SEP]', 'the', 'church', 'has', 'cracks', 'in', 'the', 'ceiling', '.', '[SEP]']

# token ids
[101, 101, 2023, 2277, 6596, 10955, 2000, 1996, 11678, 2004, 2027, 6170, 6569, 3560, 2774, 2013, 1996, 2338, 2012, 1037, 2277, 1012, 102, 1996, 2277, 2038, 15288, 1999, 1996, 5894, 1012, 102, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# attention  mask
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# token type ids
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The word embeddings represent a sequence of words as real-number vectors and encode the contextual information of each word. To get these word embeddings, the preprocessed sentence pairs are fed into the BERT model. We then take the last hidden state of the output returned by the BERT model which is a tensor with the shape `[batch_size, sequence_length, hidden_size]`. In order to get a single embedding for each pair of sentences, we use average pooling to find the mean score for each pair of sentences over the sequence length. This produces a final embedding of size 768 for each pair of sentences.
```python
with torch.no_grad():
	outputs = self.model(
	  input_ids=input_tensor,
	  attention_mask=attn_mask,
	  token_type_ids=tkn_type
	)

embeddings = outputs.last_hidden_state
embeddings = torch.mean(embeddings, dim=1)
```

Finally, the setup for OpenAI's text-davinci-002 model is straightforward. I wrote a simple prompt for the textual entailment task. The temperature, frequency penalty, and presence penalty were all set to zero.
```
Does the premise entail, contradict, or have no relationship to the hypothesis? Label the sentence pair as contradiction, entailment, or neutral.
###
premise: {premise}
hypothesis: {hypothesis}
###
Label:
```

## Results

<!-- {:refdef: style="text-align: center;"} -->
![confusion_matrices](/assets/images/post-2023-07-15/confusion_matrices.png){:style="text-align: center;"}
<!-- {: refdef} -->

The neural network model demonstrated the highest accuracy and outperformed the Naive Bayes classifier and the `text-davinci-002` model on most metrics with a few noteworthy exceptions. This outcome was anticipated due to inherent disadvantages faced by the other models. The Naive Bayes classifier relies on the assumption of conditional independence among features, which oversimplifies the contextual meaning within sentence pairs by disregarding word relationships. On the other hand, the `text-davinci-002` model does not make this assumption, thanks to the self-attention layers in its transformer architecture, which allow it to consider word context. However, unlike the neural network, the `text-davinci-002` model was not trained on the SNLI dataset and did not have the opportunity to directly model the patterns within the data through weight updates. Instead, it was merely prompted for the inference label. In contrast, the simple neural network was trained on the SNLI dataset using BERT-generated embeddings as features. This enabled the neural network to directly learn the intricate relationships among words in the sentence pairs, as these relationships are represented by the learned weights of the network.

|                  | text-davinci-002 | |          | Neural Network	|        |          | Naive Bayes  |        |          |         |
|                  | precision | recall | f1-score | precision      | recall | f1-score | precision	   | recall | f1-score | support |
| ---------------- | --------- | ------ | -------- | -------------- | ------ | -------- | ------------ | ------ | -------- | ------- |
| contradiction    |   0.9     | 0.42   | 0.57     | 0.75           | 0.72   | 0.74     | 0.51         | 0.46   | 0.48     | 3237    |
| entailment       | 0.6       | 0.84   | 0.7      | 0.69           | 0.83   | 0.76     | 0.48         | 0.58   | 0.52     | 3368    |
| neutral          | 0.35      | 0.39   | 0.37     | 0.73           | 0.6    | 0.66     | 0.52         | 0.46   | 0.49     | 3219    |
| accuracy         |           |        | 0.55     |                | 			 | 0.72     | 						 |        | 0.5      | 9824    |
| macro avg        | 0.62      | 0.55   | 0.55     | 0.73           | 0.72   | 0.72     | 0.5          | 0.5    | 0.5      | 9824    |
| weighted avg     | 0.62      | 0.55   | 0.55     | 0.73           | 0.72   | 0.72     | 0.5          | 0.5    | 0.5      | 9824    |


## Final Thoughts

The performance of these toy models has some ground to cover in order to reach the state-of-the-art results on the SNLI dataset. Several models have achieved accuracy levels surpassing 90% on the test SNLI dataset for the textual entailment task. Many of these high-performing models leverage a combination of foundational models like BERT along with additional features. Foundational language models such as BERT exhibit impressive performance on the SNLI dataset when fine-tuned. However, despite their ability to capture contextual word information, they may overlook other important linguistic features. For instance, SemBERT combines the BERT model with contextual semantic information and achieves 91.1% accuracy on SNLI's test dataset. Likewise, models like GPT-2 or LLAMA could be paired with complementary upstream or downstream models to enhance performance on the textual entailment task.

The code the experiments can be found [here](https://github.com/jwplatta/textual_entailment).

## References