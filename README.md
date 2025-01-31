# Infinite-Video
# $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation
Official implementation of the paper **$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation**.

*Saul Santos*, *Vlad Niculae*, *Daniel McNamee* and *André Martins*

<p align="center">
  <img src="./inf_video_llama.png" alt="Alt text" width="1000"/>
</p>
**Abstract**: *Current video-language models struggle with long-video understanding due to limited context lengths and reliance on sparse frame subsampling, often leading to information loss.
This paper introduces \textsc{$\infty$-Video}, which can process arbitrarily long videos through a continuous-time long-term memory (LTM) consolidation mechanism. Our framework augments video Q-formers by allowing them to process unbounded video contexts efficiently and without requiring additional training. 
Through continuous attention, our approach dynamically allocates higher granularity to the most relevant video segments, forming ``sticky'' memories that evolve over time. 
Experiments with Video-LLaMA and VideoChat2 demonstrate improved performance in video question-answering tasks, showcasing the potential of continuous-time LTM mechanisms to enable scalable and training-free comprehension of long videos.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](https://arxiv.org/abs/2411.08590) (arXiv)

All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


## Synthetic, MNIST and Multiple Instance Learning Experiments
### Python requirements and installation

This code was tested on `Python 3.10.10`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen.git) repository to the main folder: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/nunonmg/lp-sparsemap) fork repository to main folder, and follow the installation instructions found there
4. Install the requirements: `pip install -r requirements.txt`
5. Run the corresponding scripts

### Reproducibility
#### Memory Retrieval Modeling

Run the script `memory_retrieval_modeling.py`

#### Image Retrieval
Run the script `ME_plotting.py`

#### MNIST MIL
Run the script `MNIST_bags.py` with the desired parameters (nomenclature can be found in the beginning of the script)

#### Benchmarks MIL

Download and upzip the dataset

```bash
$ wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz 
```

Run the script `MIL_Data_2002.py` with the desired parameters (nomenclature can be found in the beginning of the script)

#### Countours and Basins of Attraction
Run the scripts `countours.py` and `basins.py` 

#### Metastable State Counting
Run the script `MNIST_metastable.py`

## Spectra Experiments
### Python requirements and installation
Follow the instructions of the branch in [hopfield-spectra](https://github.com/deep-spin/spectra-rationalization/tree/hopfield-spectra)

## Acknowledgment

The experiments in this work benefit from the following open-source codes:
* Saul Santos, Vlad Niculae, Daniel C McNamee, and Andre F.T. Martins. Sparse and structured hopfield networks. In International Conference on Machine Learning, 2024. https://github.com/deep-spin/SSHN
* Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber et al. "Hopfield networks is all you need." arXiv preprint arXiv:2008.02217 (2020). https://github.com/ml-jku/hopfield-layers
* Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention and multi-label classification." In International conference on machine learning, pp. 1614-1623. PMLR, 2016. https://github.com/deep-spin/entmax
* Correia, Gonçalo M., Vlad Niculae, and André FT Martins. "Adaptively sparse transformers." arXiv preprint arXiv:1909.00015 (2019). https://github.com/deep-spin/entmax
* Peters, Ben; Niculae, Vlad; Martins, André FT. "Sparse Sequence-to-Sequence Models." In Proceedings of ACL, 2019. [Online] Available: https://www.aclweb.org/anthology/P19-1146.  https://github.com/deep-spin/entmax
* Guerreiro, N. M. and Martins, A. F. T. Spectra: Sparse structured text rationalization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6534–6550, 2021. https://github.com/deep-spin/spectra-rationalization/tree/hopfield-spectra
* Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." In International conference on machine learning, pp. 2127-2136. PMLR, 2018. https://github.com/AMLab-Amsterdam/AttentionDeepMIL
