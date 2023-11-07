# DiffTransfer

## TIMBRE TRANSFER USING IMAGE-TO-IMAGE DENOISING DIFFUSION IMPLICIT MODELS


Accompanying code to the paper  _Timbre transfer using image-to-image denoising diffusion implicit models_
[[1]](#references). 

For any question, please write at [luca.comanducci@polimi.it](mailto:luca.comanducci@polimi.it).



- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Dependencies
Tensorflow (>2.11), librosa, pretty_midi, os, numpy, [essentia](https://pypi.org/project/essentia/), [frechet_audio_distance](https://pypi.org/project/frechet-audio-distance/)

### Data generation
The model is trained using the StarNet dataset, freely available on Zenodo [link](https://zenodo.org/records/6917099)

### Network training
- audio_utils.py --> contains shared audio utilities and functions
- params.py --> Contains parameters shared along scripts
- network_lib_attention.py --> Contains Denoising Diffusion Implicit Model Implementation
- DiffTransfer.py --> Actually runs the training, takes the following arguments:
  - dataset_train_path: String, path to training data
  - desired_instrument: String, name of desired output instrument
  - conditioning_instrument: String, name of input instrument
  - GPU: number of GPU, in case you have multiple ones

### Results computation

- compute_eval_tracks_mixture.py
- compute_eval_tracks_separate.py
- compute_frechet.py
- compute_jaccard.py
- compute_listening_test_results.py
- preprocess_tracks_listening_test.py



# References
>[1] Comanducci, Luca, Fabio Antonacci, and Augusto Sarti. "Timbre transfer using image-to-image denoising diffusion models. _ISMIR International Society for Music Information Retrieval Conference_ [arXiv](https://arxiv.org/pdf/2307.04586.pdf)

