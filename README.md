# 🎙️ Emotional Speech Analysis using Surprisal Theory

This project investigates **emotional speech in Serbian** through the lens of **surprisal theory**, combining linguistic features, acoustic analysis, and machine learning models to study and predict speech behavior.

The repository contains code used for:
- surprisal estimation from language models
- feature extraction from text and audio
- dataset construction
- modeling and statistical analysis
- visualization of experimental results

---

## 📚 Publications

This repository accompanies the following publications:

### 📄 Conference Paper
**Analysis of Emotional Speech in Serbian from Surprisal Theory Perspective**  
*IcETRAN 2025*  
DOI: https://doi.org/10.1109/IcETRAN66854.2025.11114098  

### 📄 Journal Paper
**Influence of the surprisal power adjustment on spoken word duration in emotional speech in Serbian**  
*Computer Speech & Language, 2025*  
DOI: https://doi.org/10.1016/j.csl.2025.101803  

---

## 🧠 Project Overview

The project explores how **surprisal (information-theoretic measure)** influences:

- word duration in speech  
- emotional expression  
- prosodic features  

It combines:

- **N-gram models**
- **Transformer models** (BERT, BERTic, XLM-R, GPT-2, GPT-Neo, Yugo-GPT)
- **Acoustic analysis** (duration, pitch, energy, MFCC)
- **Machine learning models** for prediction tasks

---

## 📁 Project Structure

The repository is organized into several functional modules:

### 🔹 Surprisal Estimation
`Surprisal estimation/`
- estimation using N-gram and transformer models
- contextual probability and entropy computation

### 🔹 Feature Extraction
`Fetures extraction/`, `Mel coefficients and surprisals/`, `Prominence/`
- extraction of linguistic and acoustic features
- MFCC, pitch, energy, prosodic features

### 🔹 Dataset Construction
Multiple `build_dataset.py` scripts across folders
- merging linguistic, acoustic, and surprisal features
- preparing data for modeling

### 🔹 Modeling
- `Emotion recognition/` → CNN-based emotion classification
- `Linear regression/`, `Duration Prediction based on Surprisals/` → duration prediction

### 🔹 Analysis & Results
`Additional files after recension/`
- final analyses and plots used in publications
- statistical evaluation (AIC, log-likelihood, etc.)

### 🔹 Visualization
`Generate graphs/`
- plotting time-series and feature behavior

### 🔹 Experimental Variants
- `Different information measurement parameters/`
- `Split-over effect/`
- `Pervious Surprisals/`

These contain alternative experimental setups and parameter variations.

---

## ⚙️ Setup

### 1. Install dependencies

Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt