# SSTA
The officially code of our paper "SSTA: Salient Spatially Transformed Attack".


## Introduction

This repository contains the PyTorch implementation for the paper [SSTA: Salient Spatially Transformed Attack](https://ieeexplore.ieee.org/abstract/document/10447882).


## Abstract
Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial examples (AEs), posing significant security risks to real-world AI applications. To address these vulnerabilities and enhance model robustness against malicious inputs, various attack methods have been proposed to craft AEs. However, despite recent progress, existing methods often rely heavily on noise perturbations, making the adversarial patterns perceptible to the human eye. This visibility greatly increases the risk of exposure and reduces the effectiveness of the attacks. To overcome this limitation, we propose the **Salient Spatially Transformed Attack (SSTA)**—a novel framework that generates imperceptible AEs by estimating smooth spatial transformations focused on critical regions, rather than applying noise across the entire image. Compared to state-of-the-art baselines, extensive experiments demonstrate that SSTA significantly improves imperceptibility while maintaining a 100% attack success rate.


### Run SSTA
```CUDA_VISIBLE_DEVICES=0 python -W ignore -u attack.py``` 


### Acknowledgements

We extend our gratitude to the following repositories for their contributions and resources:

- [stAdv](https://github.com/rakutentech/stAdv)
- [TRACER](https://github.com/Karel911/TRACER)

Their works have significantly contributed to the development of our work.

## Citation

If you think this work or our codes are useful for your research, please cite our paper via:

```bibtex
@inproceedings{liu2024ssta,
  title={SSTA: Salient Spatially Transformed Attack},
  author={Liu, Renyang and Zhou, Wei and Wu, Sixing and Zhao, Jun and Lam, Kwok-Yan},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5910--5914},
  year={2024},
  organization={IEEE}
}
```