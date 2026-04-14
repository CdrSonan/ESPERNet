# ESPERNet
ESPERNet is a set of AI models for speech processing, along with the training and ONNX export code used to build and deploy them.

ESPERNet is built from three models: an encoder, a decoder, and a classifier.
Together, they form a VLGAN architecture.
***(Lee, Je-Yeol & Choi, Sang-Il. (2020). Improvement of Learning Stability of Generative Adversarial Network Using Variational Learning. Applied Sciences. 10. 4528. 10.3390/app10134528. )***

The VAE latent space is split into two parts: a low-dimensional, time-dependent phoneme space and a larger, time-constant style-and-speaker space.
This difference in dimensionality enables the encoder to separate speech content from speaker identity.
***(Xie, Yuying, et al. "Speaker and style disentanglement of speech based on contrastive predictive coding supported factorized variational autoencoder." 2024 32nd European Signal Processing Conference (EUSIPCO). IEEE, 2024.)***

Paired with the decoder, which benefits from adversarial training to produce high-quality outputs, the model can be used for speech transfer from one speaker to another.

Unlike most audio-processing networks, ESPERNet uses the ESPER format instead of MEL spectrograms.
That format was designed specifically for speech processing and enables high-quality speech reconstruction without an auxiliary network, as well as a range of effects, augmentations and transforms through [libESPER-V2](https://github.com/CdrSonan/libESPER-V2).
