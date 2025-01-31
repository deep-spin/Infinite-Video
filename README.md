# Infinite-Video
# $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation
Official implementation of the paper **$\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation**.

*Saul Santos*, *Vlad Niculae*, *Daniel McNamee* and *André Martins*

<p align="center">
  <img src="./inf_video_llama.png" alt="Alt text" width="1000"/>
</p>
**Abstract**: *Current video-language models struggle with long-video understanding due to limited context lengths and reliance on sparse frame subsampling, often leading to information loss.
This paper introduces ∞-Video, which can process arbitrarily long videos through a continuous-time long-term memory (LTM) consolidation mechanism. Our framework augments video Q-formers by allowing them to process unbounded video contexts efficiently and without requiring additional training. 
Through continuous attention, our approach dynamically allocates higher granularity to the most relevant video segments, forming ``sticky'' memories that evolve over time. 
Experiments with Video-LLaMA and VideoChat2 demonstrate improved performance in video question-answering tasks, showcasing the potential of continuous-time LTM mechanisms to enable scalable and training-free comprehension of long videos.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](to add) (arXiv)

All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


## Video LLaMA
### Python requirements and installation

This code was tested on `Python 3.10.10`. To install, follow the steps of [moviechat](https://github.com/rese1f/MovieChat)

### Reproducibility
1 - Run ```eval_code/extract_features``` on the intended dataset with the desired number of frames.

2 - Run each script in ```eval_code/eval``` with the hyperparameters mentioned in the paper:
Example: 
```
 python3 eval_code/eval/run_inference_inf_video_llama_nextqa.py     --cfg-path eval_configs/infvideollama.yaml     --num-beams 1   --temperature 1 --video-folder next_qa/features --q-folder /mnt/scratch-artemis/saul/next_qa/val.csv     --output-dir /MovieChat/nextqa_val     --max_int 256    --num_basis 256  --tau 0.75  --alpha 1.0 --task inf_video_llama --sticky
```

3 - For open-ended questions run ```eval_code/validate/run_eval_qa_chatgpt.py```with the output of the moviechat script run.

4 - For multiple-choice questions we predict the answers as open-ended and use langchain to select the most similar option, run ```eval_code/validate/run_eval_langchain.py```with the output from the dataset run script as:

```
python eval_code/validate/run_eval_langchain.py --pred_path egoschema/nframes_8_nchunks_256_moviechatplus/preds.json --num_tasks 100
```

Then compute accuracy with ```run_eval.py```
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
