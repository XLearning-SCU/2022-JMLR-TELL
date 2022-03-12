# 2022-JMLR-TELL
This is the code for the paper "XAI Beyond Classiﬁcation: Interpretable Neural Clustering (TELL)" (JMLR 2022)

# Usage
To train the model on the MNIST dataset, run
> python main.py

The performance of the model is evaluated during the training process, together with the the visualization of the reconstructed cluster centers.

To perform tSNE visualization, run
>python tSNE.py

# Citation
If you find TELL useful in your research, please consider citing:
```
@article{peng2022xai,
  title={XAI Beyond Classification: Interpretable Neural Clustering},
  author={Peng, Xi and Li, Yunfan and Tsang, Ivor W and Zhu, Hongyuan and Lv, Jiancheng and Zhou, Joey Tianyi},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={6},
  pages={1--28},
  year={2022}
}
```
