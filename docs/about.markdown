---
layout: page
title: >-
    Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
permalink: /about/
---

#  Sound of DAFx Special Issue - 2025

Accompanying page with supplemental materials


## Abstract
This paper investigates the feasibility of running neural audio generative models on embedded systems, by comparing the performance of various models and evaluating their trade-offs in audio quality, inference speed, and memory usage. We focus on Differentiable Digital Signal Processing (DDSP) models, due to their hybrid architecture, which combines the efficiency and interoperability of traditional DSP with the flexibility of neural networks. In addition, we explore the application of Knowledge Distillation (KD) to improve the performance of smaller models. We implemented and evaluated two types of distillation strategies: audio distillation and control distillation. We applied these methods to three foundation DDSP generative models which integrate Harmonic-plus-Noise, FM, and Wavetable synthesis. Our results demonstrate the overall effectiveness of KD: we were able to train student models that are up to a hundred times smaller than their teacher counterparts, while maintaining comparable performance and significantly improving inference speed and memory efficiency. However, we have also observed cases where KD failed to improve or even degrade student performance. We provide a critical reflection on the advantages and limitations of KD, exploring its application in diverse use cases, and emphasizing the need for carefully tailored strategies to maximize its potential.
