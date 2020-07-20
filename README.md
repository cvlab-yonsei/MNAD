# PyTorch implementation of "Learning Memory-guided Normality for Anomaly Detection"

<p align="center"><img src="./MNAD_files/overview.png" alt="no_image" width="40%" height="40%" /><img src="./MNAD_files/teaser.png" alt="no_image" width="60%" height="60%" /></p>
This is the implementation of the paper "Learning Memory-guided Normality for Anomaly Detection (CVPR 2020)".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/MNAD/)] and the paper [[PDF](http://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)].

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``

## Training
* The training and testing codes are based on prediction method
```bash
git clone https://github.com/cvlab-yonsei/projects
cd projects/MNAD/code
python Train.py # for training
```
* You can freely define parameters with your own settings like
```bash
python Train.py --gpus 1 --dataset_path 'your_dataset_directory' --dataset_type avenue --exp_dir 'your_log_directory'
```

## Pre-trained model and memory items
* Download our pre-trained model and memory items <br>Link: [[model and items](https://drive.google.com/file/d/11f65puuljkUa0Z4W0VtkF_2McphS02fq/view?usp=sharing)]
* Note that, these are from training with the Ped2 dataset

## Evaluation
* Test the model with our pre-trained model and memory items
```bash
python Evaluate.py --model_dir pretrained_model.pth --m_items_dir m_items.pt
```
* Test your own model
```bash
python Evaluate.py --model_dir your_model.pth --m_items_dir your_m_items.pt
```

## Bibtex
```
@inproceedings{park2020learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}
```
