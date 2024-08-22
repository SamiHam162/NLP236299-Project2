# CS236299 Project 2: Sequence Labeling - Slot Filling Task

This repository contains the implementation of Project Segment 2 for the CS236299 course. The focus of this project is on sequence labeling for the slot-filling task using different approaches including Hidden Markov Models (HMM), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM) networks.

## Table of Contents

- [Project Overview](#project-overview)
- [Goals](#goals)
- [Implementation Details](#implementation-details)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Discussion](#discussion)
- [License](#license)

## Project Overview

The primary objective of this project is to label tokens in a text sequence with appropriate slot labels as part of a sequence labeling task. The slot-filling task is crucial for understanding the meaning of queries in natural language processing, particularly for tasks such as question answering. This project involves implementing and comparing three different approaches:

1. **Hidden Markov Model (HMM)** for sequence labeling.
2. **Recurrent Neural Network (RNN)** for sequence labeling.
3. **Long Short-Term Memory (LSTM)** for sequence labeling.

## Goals

1. Implement an HMM-based approach for sequence labeling.
2. Implement an RNN-based approach for sequence labeling.
3. Implement an LSTM-based approach for sequence labeling.
4. Compare the performance of HMM and RNN/LSTM models with different amounts of training data.

## Implementation Details

### 1. HMM for Sequence Labeling

- **Training the HMM**: The HMM is trained using transition and emission probabilities derived from the training data. Add-δ smoothing is applied to handle unseen events.
- **Viterbi Algorithm**: The Viterbi algorithm is used to predict the most likely sequence of tags for a given sequence of words.

### 2. RNN for Sequence Labeling

- **RNN Architecture**: An RNN is implemented to predict the sequence of tags. The RNN is trained using a cross-entropy loss function, and the performance is evaluated on the validation set.
- **Forward Pass and Loss Computation**: The forward pass of the RNN computes the logits for each token in the sequence, which are then used to compute the loss.

### 3. LSTM for Sequence Labeling

- **LSTM Architecture**: An LSTM is implemented as an extension of the RNN to address the vanishing gradient problem. The LSTM is trained similarly to the RNN and is expected to perform better on longer sequences.
- **Training and Evaluation**: The LSTM is trained and evaluated on the same dataset, and its performance is compared with that of the RNN and HMM.

### 4. Comparison of Models

- **Varying Training Data**: The models' performances are compared by varying the amount of training data. This comparison helps in understanding the pros and cons of statistical vs. neural approaches for sequence labeling.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SamiHam162/NLP236299-Project2.git
   cd NLP236299-Project2
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the necessary datasets and scripts as outlined in the `project2_sequence.ipynb` file.

## Usage

1. **HMM Implementation**:
   - Train the HMM using the provided training data.
   - Use the Viterbi algorithm to predict tag sequences for the test data.
2. **RNN Implementation**:
   - Train the RNN model and evaluate its performance on the validation and test sets.
3. **LSTM Implementation**:
   - Train the LSTM model and compare its performance with the RNN and HMM models.
4. **Comparison**:
   - Vary the amount of training data and analyze the performance of the models.

## Evaluation

The evaluation of the models is based on their accuracy in predicting the correct sequence of tags for the test data. The results are presented in terms of overall accuracy and are compared across different models and training data sizes.

## Discussion

The final section of the project involves a discussion on the strengths and weaknesses of the HMM, RNN, and LSTM approaches for sequence labeling, particularly in the context of slot filling. The discussion also includes insights into the impact of the amount of training data on model performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
