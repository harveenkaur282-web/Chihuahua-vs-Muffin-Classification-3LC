# 3LC-MNNIT-AI-Hackathon-task

# Chihuahua vs Muffin Classification (3LC)

## What is this?

A binary image classifier that solves the internet's most important question: **is it a chihuahua or a muffin?**

Built using **ResNet** as the backbone and **3LC** for dataset management, interactive labeling, and iterative retraining — this project showcases how human-in-the-loop feedback can dramatically improve model accuracy.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ResNet](https://img.shields.io/badge/ResNet-Transfer%20Learning-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![3LC](https://img.shields.io/badge/3LC-Dataset%20Management-6C63FF?style=for-the-badge&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## Approach

* Used ResNet-18 (trained from scratch as per competition rules)
* Leveraged 3LC for data-centric workflow
* Iteratively improved labels using 3LC dashboard

## Key Steps

* Initial training on provided dataset
* Identified low-confidence and misclassified samples
* Manually corrected labels in 3LC
* Retrained model using updated dataset

## Results

* Achieved validation accuracy: ~94%
* Final Kaggle score: **0.93412**

## Tools Used

* PyTorch
* 3LC (data labeling + analysis)

## Key Insight

Improving data quality (labels) significantly boosted performance more than model changes.

## Screenshots

See `/screenshots` folder for training runs and dataset views.

## Future Work
1. Further improve performance by refining edge-case samples,
2. Experiment with advanced architectures under relaxed constraints,
3. Explore automated label correction using model feedback,
4. Extend this workflow to larger, real-world datasets
