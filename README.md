# Machine-Unlearning-Implementation

# Overview
This project implements a Machine Learning pipeline for loan data prediction using a deep neural network, and extends it with SISA (Sharded, Isolated, Sliced, Aggregated) training to enable efficient machine unlearning.

Machine unlearning is critical for:

Data privacy (GDPR compliance)
Removing biased or incorrect data
Efficient model updates without retraining
This project demonstrates a practical and scalable approach using SISA.

# The goal is to:

Train a predictive model on financial/loan data
Partition training into shards and slices
Efficiently remove specific data points without retraining the full model
Key Features
End-to-end ML pipeline (data → preprocessing → training → evaluation)

Deep Neural Network built using PyTorch

Implementation of SISA training framework

Efficient data deletion (machine unlearning) mechanism

Performance tracking using:

RMSE
Loss curves
Accuracy plots

# Tech Stack
Python
PyTorch
NumPy, Pandas
Scikit-learn
Matplotlib, Seaborn

# Future Improvements
Optimize shard/slice selection strategy
Add GPU acceleration
Implement differential privacy
Extend to classification tasks
Build API for real-time unlearning


