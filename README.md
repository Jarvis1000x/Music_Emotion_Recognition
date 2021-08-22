# Explainable Music Emotion Recognition using Mid-Level features

This Repository is an implementation of [Towards Explainable  Music Emotion Recognition The Route via Mid-level features](https://arxiv.org/pdf/1907.03572) by S. Chowdhury et al. The model tries to give a musically meaningful and intuitive explanation for its Music Emotion predictions, a VGG-style deep neural network has been used that learns to predict emotional characteristics of a musical piece together with (and based on) human-interpretable, mid-level perceptual features.

### Dataset

For datasets, the Aljanaki & Soleymani’s [Mid-level Perceptual Features dataset](https://osf.io/5aupt/) provides mid-level feature annotations. For the actual emotion prediction experiments, [the Soundtracks dataset](https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/projects2/past-projects/coe/materials/emotion/soundtracks/Index) has been used, which contains the Aljanaki collection as a subset, and comes with numeric emotion ratings along 8 dimensions.
