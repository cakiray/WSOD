import kittieval.kitti_common as kitti
from kittieval.eval import get_official_eval_result
import argparse

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

 #How to run it:
 # kitti_eval.py --det_path=/data/Ezgi/preds --gt_path=/data/dataset/kitti/object/training/label_2 --val_txt=/data/dataset/kitti/object/training/val.txt

parser = argparse.ArgumentParser()
parser.add_argument('--det_path', metavar='DIR', help='detections saved path')
parser.add_argument('--gt_path', metavar='DIR', help='ground truth path')
parser.add_argument('--val_txt', metavar='DIR', help='val.txt file path')
args, opts = parser.parse_known_args()


det_path = args.det_path
dt_annos = kitti.get_label_annos(det_path)
gt_path = args.gt_path
gt_split_file = args.val_txt
val_image_ids = _read_imageset_file(gt_split_file)

#val_image_ids = val_image_ids[len(val_image_ids)//2:]

gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
class_ = 0 # CAR
print(get_official_eval_result(gt_annos, dt_annos, class_))
