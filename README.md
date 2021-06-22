
![](https://img.shields.io/badge/language-python-orange.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v1.0.0-519dd9.svg)
![99%](https://progress-bar.dev/100)
# Variational Autoencoders for Anti-cancer Drug Response Prediction
Welcome to my first cancer machine learning project. It's my second years' undergraduate research. Here's my teammates: Yuan and Varus and Dexin, who are all undergradaute at their third, second and first year. We are supervised by Prof. Manolis Kellis from MIT CSAIL LAB. Our paper link:(https://arxiv.org/pdf/2008.09763.pdf), accepted by 2021 ICLR AI4PH workshop (paper 33)



## Check your environment
You should have a pytorch version of 1.5.0 or above. Also you should have a cuda version of 9 or above.

The following order in the command line can download the newest version of pytorch and torchvision with cuda version of 10.2. 
```Bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
In /Code/CancerML/mlps_drug_exp env, run train.py to test our model on breast cancer dataset(dont need to change anything)
```python
python train.py
```
run train_pan.py to test our model on pancancer datasets
```python
python train_pan.py
```
## Citation
If you use to cite our results in your research paper, please consider citing:
```
@misc{dong2021variational,
      title={Variational Autoencoder for Anti-Cancer Drug Response Prediction}, 
      author={Hongyuan Dong and Jiaqing Xie and Zhi Jing and Dexin Ren},
      year={2021},
      journals={AI4PH workshop, ICLR 2021},
}
```
