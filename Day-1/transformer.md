# Transformers: A Beginner's Learning Guide

## The big idea

GPT stands for **Generative Pre-trained Transformer**:

- **Generative**: It creates new text, one piece at a time.
- **Pre-trained**: It learned patterns from a large collection of text before you started chatting with it.
- **Transformer**: It uses a neural-network design that helps it work out how words and other tokens relate to each other.

A language model does not usually predict a whole sentence at once. It repeatedly estimates what token is likely to come next, then uses that token to help choose the next one.

> **Word or token?** A token is a small piece of text. It might be a whole word, part of a word, punctuation, or another text fragment.

## Follow the text through the model

Here is the journey from your prompt to an answer:

```text
Your text
   |
   v
Split text into tokens
   |
   v
Turn each token into a vector (an embedding)
   |
   v
Repeatedly update token vectors using:
   attention -> feed-forward network
   |
   v
Estimate probabilities for the next token
   |
   v
Choose a token and add it to the text
   |
   +---- repeat until the answer is complete
```

### 1. Text becomes tokens

The model first splits the prompt into tokens. For example, a long or uncommon word might be split into smaller pieces. The model works with these tokens rather than directly reading words the way a person does.

### 2. Tokens become vectors

Each token is mapped to a list of numbers called an **embedding**. These vectors give the model a numerical starting representation for each token. An embedding does not, by itself, contain a complete, fixed definition of a word; the surrounding text matters too.

### 3. Attention adds context

In a **self-attention** operation, each token can use information from other tokens in the text. A useful way to understand the calculation is through three learned projections:

- **Query**: what information is this token looking for?
- **Key**: what information does each token offer?
- **Value**: what information can each token contribute?

The model compares a token's query with the keys to calculate relevance scores, then combines the values according to those scores. This is a simplified picture of attention, but it helps explain how context is shared.

For example, in "She deposited cash at the bank," the words around "bank" point toward a financial institution. In "They sat on the river bank," the surrounding words point toward the side of a river.

### 4. Feed-forward layers transform each token

A **feed-forward network**, also called a multilayer perceptron (MLP), transforms each token's current representation. Unlike attention, this part processes each position separately.

### 5. Transformer blocks repeat

The model applies many transformer layers. Each layer uses attention and feed-forward computation to refine the representations. By the end, the model has built a context-aware representation of the text.

## How does a model learn?

During training, the model practices predicting the next token from text. It compares its prediction with the actual next token and adjusts its internal values to improve future predictions.

- **Parameters (weights)** are the learned numerical values inside the model. They shape how it responds.
- **Backpropagation** calculates how the parameters contributed to a prediction error so that a training algorithm can adjust them.
- Much of the computation uses large matrix operations on these values and the token representations.

The prompt is not the same thing as the model's learned parameters. Your prompt is temporary input for the current interaction; the parameters are the learned values the model uses across interactions.

## What happens when you send a prompt and receive an answer?

The work is commonly described in two phases:

### Prefill: read the prompt

The model processes the prompt's tokens together. This phase builds intermediate attention information called the **key-value cache**, or **KV cache**. The cache can be reused while generating the answer, so the model does not have to recompute all earlier token information from scratch at every step.

Prompt tokens can be processed in parallel by the hardware. This does not mean the work is free: the amount of computation depends on the prompt length, model size, and implementation.

### Decode: write the answer

The model generates the answer **one token at a time**:

1. Use the prompt and tokens already generated to estimate the next-token probabilities.
2. Select the next token.
3. Add it to the KV cache and to the growing answer.
4. Repeat until the answer is complete.

Because each new token depends on the previous text, these generation steps are sequential. The KV cache stores attention information for earlier tokens, so the model can reuse it instead of recomputing those states from scratch. Each step still needs to process the new token in the context of what came before.

Decode is often limited by how quickly the hardware can move model weights and cached attention data from memory, rather than by its maximum arithmetic throughput. The exact bottleneck depends on the model, hardware, context length, and serving setup.

## Answers to the questions

### Q1: Why does generating 1,000 tokens cost about 10 times more than generating 100?

Generating 1,000 tokens requires 1,000 sequential **token-generation steps**, compared with 100 steps for a 100-token answer. Each step runs the model to choose another token, so the decode portion usually takes more computation and time for the longer answer.

This is why the decode work is often summarized as "about 10 times as many steps." It does not guarantee exactly 10 times the total cost or elapsed time: prefill adds fixed prompt work, and later steps have more context to process. The result also depends on the model, hardware, batching, and serving setup.

### Q2: If "bank" has the same embedding every time, how does the model know which meaning to use?

Assuming "bank" is represented by the same token in both sentences, it starts with the same token embedding. The embedding is a learned starting vector, not a complete definition that already selects one meaning.

As the token passes through transformer layers, self-attention combines information from relevant surrounding tokens. In "cash deposited at the bank," its query can give more weight to keys associated with words such as "cash" and "deposited"; in "steep river bank," it can give more weight to "river" and "steep." The representation (often called a **hidden state**) is updated at each layer, so it becomes context-sensitive.

By the final layer, the two uses of "bank" can have substantially different hidden states. The model uses these processed representations to help predict the next token; it does not make that prediction from the initial embedding alone.

### Q3: What is the difference between a model's parameters and its context window?

- **Parameters** are the learned numerical values (weights) inside the model. They encode patterns learned during training. In ordinary inference, these weights are used to answer but are not updated by the prompt. For example, 70 billion parameters stored at 16 bits each would take about 140 GB just for the parameter values; real deployments also need additional memory and may use compression or other formats.
- **Context window** is the maximum amount of text, measured in tokens, that a model can process in one interaction. It may include system instructions, conversation history, retrieved documents, and generated output. The exact limit depends on the model and product.

In short: parameters are the model's learned machinery; the context window is the temporary working space for the text it can consider at once. A product may retain conversation history or saved memories outside an individual model call, but that does not mean the model's parameters learned from the conversation.

This distinction is why retrieval-augmented generation (RAG), vector databases, and memory features are useful: they can store information outside the model and supply relevant pieces in a later prompt. They do not remove the context-window limit; the information sent to the model still has to fit within it.

One rough analogy:

```text
Parameters  -> learned machinery       -> reused across requests
Context     -> working notes           -> supplied for a particular interaction
Vector DB   -> searchable storage      -> retrieves relevant notes when needed
```

These are conceptual comparisons, not literal mappings to disk, RAM, or a specific product's session behavior.

## One-line recap

```text
Text -> tokens -> embeddings -> context-aware transformer layers
     -> next-token prediction -> repeat to create an answer
```
