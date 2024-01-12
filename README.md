# LLama2 in Pytorch
## WIP

## Overview
This projects implements [LLama2](https://arxiv.org/abs/2307.09288) transformer architecture for self-supervised prediction, which is at the core of LLMs. This project aims to provide a simple and efficient implementation of popular Llama model which is based on the original [transformer architecture](https://arxiv.org/abs/1706.03762) which is highly flexible and powerful, but implements few upgrades such as: [rotary embeddings](https://arxiv.org/pdf/2104.09864.pdf), [grouped query attention for a tradeoff between MHA and MQA](https://arxiv.org/abs/2305.13245v3), [SwiGLU](https://arxiv.org/abs/2002.05202v1), [RMS Norm](https://arxiv.org/abs/1910.07467) and [KV Caching](https://arxiv.org/pdf/2211.05102.pdf).

## Llama2 Architecture

![LLaMa2 ](https://images.datacamp.com/image/upload/v1700044736/image9_02d9fcb498.png)

 The Llama architecture consists of the Transformer Decoder architecture, coupled with few upgrades such as :
 * Rotary Embeddings
 * SwiGLU
 * Grouped Query Attention
 * KV Caching


**Decoder**: The decoder takes in the output of the encoder and generates the final output sequence. It also consists of a stack of decoder layers. Each decoder layer has a grouped query multi-head self-attention mechanism, feed-forward neural network.
It benefits from RoPe encodings, KV caching and everything mentioned above.

 **Grouped Query Attention**: The grouped query attention mechanism is a modification to the traditional attention mechanism in the transformer architecture. It allows the model to attend to different groups of queries within the input sequence, enabling a tradeoff between multi-head attention and multi-query attention. This helps improve the model's ability to capture complex dependencies and relationships within the data.


For more details on the transformer architecture, refer to the original paper: [Llama](https://arxiv.org/abs/2307.09288).



## Features

✨ Easy-to-use: The project provides a straightforward setup and training loop for self-supervised prediction.


🔀 Self-Supervised Prediction: The training loop is designed to support self-supervised prediction, enabling the model to learn from unlabeled data.

## Setup

To get started with Transformer Plain, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/paulilioaica/Llama2-Pytorch
    cd Llama2-Pytorch/src/

    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

1. Dataset: Make sure you have a dataset suitable for self-supervised prediction from Huggingface (or use the AG-NEWS one). Simply pass the `dataset_name` for training on your dataset of choice.

2. Configure the training parameters: Adjust the hyperparameters by passing your own arguments.

3. Train the model: Run the training script to start the self-supervised prediction training loop.

4. Evaluate the model: Use the trained model to make predictions on your test dataset and evaluate its performance.

## Example run
```
python main.py  --num_layers 2 --n_heads 8 --seq_len 128 --num_hidden 128 --num_epochs 10 --batch_size 32 --lr 0.001 --device cpu --embedding_dim 128 --dataset_name ag_news
```

## License

This project is licensed under the MIT License. 
