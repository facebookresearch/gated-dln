# Gated Deep Linear Networks (DLN)
Code for the ICML 2022 paper `The Neural Race Reduction: Dynamics of Abstraction in Gated Networks`

## Abstract

Our theoretical understanding of deep learning has not kept pace with its empirical success. While network architecture is known to be criti- cal, we do not yet understand its effect on learned representations and network behavior, or how this architecture should reflect task structure.In this work, we begin to address this gap by introduc- ing the Gated Deep Linear Network framework that schematizes how pathways of information flow impact learning dynamics within an architec- ture. Crucially, because of the gating, these net- works can compute nonlinear functions of their in- put. We derive an exact reduction and, for certain cases, exact solutions to the dynamics of learn- ing. Our analysis demonstrates that the learning dynamics in structured networks can be concep- tualized as a neural race with an implicit bias to- wards shared representations, which then govern the model’s ability to systematically generalize, multi-task, and transfer. We validate our key in- sights on naturalistic datasets and with relaxed assumptions. Taken together, our work gives rise to general hypotheses relating neural architecture to learning and provides a mathematical approach towards understanding the design of more com- plex architectures and the role of modularity and compositionality in solving real-world problems.
## Citation

```
@InProceedings{pmlr-v162-saxe22a,
  title = 	 {The Neural Race Reduction: Dynamics of Abstraction in Gated Networks},
  author =       {Saxe, Andrew and Sodhani, Shagun and Lewallen, Sam Jay},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {19287--19309},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/saxe22a/saxe22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/saxe22a.html},
  abstract = 	 {Our theoretical understanding of deep learning has not kept pace with its empirical success. While network architecture is known to be critical, we do not yet understand its effect on learned representations and network behavior, or how this architecture should reflect task structure.In this work, we begin to address this gap by introducing the Gated Deep Linear Network framework that schematizes how pathways of information flow impact learning dynamics within an architecture. Crucially, because of the gating, these networks can compute nonlinear functions of their input. We derive an exact reduction and, for certain cases, exact solutions to the dynamics of learning. Our analysis demonstrates that the learning dynamics in structured networks can be conceptualized as a neural race with an implicit bias towards shared representations, which then govern the model’s ability to systematically generalize, multi-task, and transfer. We validate our key insights on naturalistic datasets and with relaxed assumptions. Taken together, our work gives rise to general hypotheses relating neural architecture to learning and provides a mathematical approach towards understanding the design of more complex architectures and the role of modularity and compositionality in solving real-world problems. The code and results are available at https://www.saxelab.org/gated-dln.}
}
```

## Links

1. [Paper](https://proceedings.mlr.press/v162/saxe22a.html)
2. [Code](https://github.com/facebookresearch/gated-dln)
3. [Slides](slides.pdf)
4. [Recording]()

## Install

* `conda create --name abstraction_by_gating python=3.9`
* `conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch`
* `pip install -r requirements.txt`

## Experiments

* `./scripts/mnist.sh` for running MNIST experiments.
* `./scripts/cifar10.sh` for running CIFAR10 experiments.

## License

[Creative Commons Attribution-NonCommercial 4.0 International](LICENSE.md)
