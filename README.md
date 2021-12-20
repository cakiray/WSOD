# WSOD

Weakly Supervised Object Detection in Point Cloud

Connect to the server via ssh 
```python 
> ssh \<username>@131.159.10.176
> cd /data/Ezgi/wsod
```
_**TRAIN AND TEST WSCD**_

**TRAIN** 
```python 
> conda activate sparse-env
> torchpack dist-run -np 1 python spvnas/train.py spvnas/configs/kitti/default.yaml
```
Note: Weights of the best model during training is saved in /data/Ezgi/best_model. The full path of saved weights is printed out during training.

**GENERATE EVALUATION RESULTS**

Modify _outputs_ in config file to save results to that directory.

To save results in KITTI format to evaluate later:
```python 
> conda activate sparse-env
> torchpack dist-run -np 1 python spvnas/evaluate.py spvnas/configs/kitti/default.yaml --weights <path/to/weights>
```
Example:
- torchpack dist-run -np 1 python spvnas/evaluate.py spvnas/configs/kitti/default.yaml --weights /data/Ezgi/best_model/07-09-00:49.pt

To evaluate results with KITTI evaluation:
```python 
> conda activate sparse-env
> python kitti_eval.py --det_path=<path/to/predictions> --gt_path=/data/dataset/kitti/object/training/label_2 --val_txt=/data/dataset/kitti/object/training/val.txt
```

Example:
- python kitti_eval.py --det_path=/data/Ezgi/preds --gt_path=/data/dataset/kitti/object/training/label_2 --val_txt=/data/dataset/kitti/object/training/val.txt

To generate Peak Response Maps
Modify _outputs_ in config file to save peak repsonse maps to that directory. 
```python 
> torchpack dist-run -np 1 python spvnas/prm_generator.py spvnas/configs/kitti/default.yaml --weights <path/to/weights>
```

Note:
- Parameters are set in spvnas/configs/kitti/default.yaml . You can change parameters for model or paths for saving results in that file.


_**TRAIN AND TEST CENTERPOINT**_
```python 
> ssh \<username>@131.159.10.176
> cd /data/Ezgi/wsod/CenterPoint
> python setup.py develop
```

See CenterPoint/docs/GETTING_STARTED.md Data Preparation for KITTI

```
> python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

**TRAIN**
```python 
> cd tools
> python train.py --cfg_file ${CONFIG_FILE} --extra-tag ${tag}
```

**TEST**
```python 
python test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CKPT}
```

For any question, refer to docs in CenterPoint repo.