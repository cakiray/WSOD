# WSOD

Weakly Supervised Object Detection in Point Cloud

HOW TO RUN IN SERVER:

Connect to the server via ssh 
- ssh \<username>@131.159.10.176

Go to repository
- cd /data/Ezgi/wsod

Activate the environment using conda
- conda activate sparse-env

For training : 
- torchpack dist-run -np 1 python spvnas/train.py spvnas/configs/kitti/default.yaml

  Weights of the best model during training is saved in /data/Ezgi/best_model. The full path of saved weights is printed out during training.

To save results in KITTI format to evaluate later:
- torchpack dist-run -np 1 python spvnas/evaluate.py spvnas/configs/kitti/default.yaml --weights <path/to/weight/model>

eg:  torchpack dist-run -np 1 python spvnas/evaluate.py spvnas/configs/kitti/default.yaml --weights /data/Ezgi/best_model/07-09-00:49.pt

To evaluate results with KITTI evaluation:
- python kitti_eval.py --det_path=<path/to/predictions> --gt_path=/data/dataset/kitti/object/training/label_2 --val_txt=/data/dataset/kitti/object/training/val.txt

eg: python kitti_eval.py --det_path=/data/Ezgi/preds --gt_path=/data/dataset/kitti/object/training/label_2 --val_txt=/data/dataset/kitti/object/training/val.txt

Note:
- For testing, second half of validation set in KITTI is used.
- Parameters are set in spvnas/configs/kitti/default.yaml . You can change parameters for model or paths for saving results in that file.
