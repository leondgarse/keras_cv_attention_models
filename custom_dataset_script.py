#!/usr/bin/env python3
import os
import json
import random
from glob2 import glob
from tqdm import tqdm

IMAGE_SUFFIX = ["*.jpg", "*.jpeg", "*.png"]


def walk_through_image_folder(data_path, depth=2, image_classes_rule=None):
    if not os.path.exists(data_path):
        print(">>>> [Error] data_path not exists, data_path:", data_path)
        return [], []

    image_base = os.path.join(data_path, "*") if depth == 2 else data_path
    image_names = []
    for suffix in IMAGE_SUFFIX:
        image_names += glob(os.path.join(image_base, suffix))
    random.shuffle(image_names)

    if image_classes_rule is None:
        return image_names
    else:
        image_classes = [image_classes_rule(ii) for ii in image_names]
        return image_names, image_classes


""" Recognition dataset """


class ImageClassesRule_map:
    def __init__(self, dir, dir_rule="*", excludes=[]):
        raw_labels = [os.path.basename(ii) for ii in glob(os.path.join(dir, dir_rule))]
        raw_labels = [ii for ii in raw_labels if ii not in excludes]
        is_all_numeric = sum([str.isnumeric(ii) for ii in raw_labels]) == len(raw_labels)
        if is_all_numeric:
            self.raw_labels = sorted(raw_labels, key=lambda xx: int(xx))
            self.labels_2_indices = {ii: int(ii) for ii in self.raw_labels}
        else:
            self.raw_labels = sorted(raw_labels)
            self.labels_2_indices = {ii: id for id, ii in enumerate(self.raw_labels)}
        self.indices_2_labels = {vv: kk for kk, vv in self.labels_2_indices.items()}

    def __call__(self, image_name):
        raw_image_label = os.path.basename(os.path.dirname(image_name))
        return self.labels_2_indices[raw_image_label]


def build_recognition_dataset_json(train_path, test_path=None, test_split=0.0, save_name=None):
    if save_name is None:
        save_name = os.path.basename(os.path.dirname(train_path)) + ".json"
    elif not save_name.endswith(".json"):
        save_name += ".json"
    # print(f">>>> {train_path = }, {test_path = }, {test_split = }, {save_name = }")
    image_classes_rule = ImageClassesRule_map(train_path)
    x_train, y_train = walk_through_image_folder(train_path, image_classes_rule=image_classes_rule)
    if test_path is not None:
        x_test, y_test = walk_through_image_folder(test_path, image_classes_rule=image_classes_rule)
    else:
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_split, random_state=42)

    x_train = [os.path.abspath(ii) for ii in x_train]
    x_test = [os.path.abspath(ii) for ii in x_test]
    train = [{"image": ii, "label": jj} for ii, jj in zip(x_train, y_train)]
    test = [{"image": ii, "label": jj} for ii, jj in zip(x_test, y_test)]
    num_classes = len(image_classes_rule.indices_2_labels)
    info = {"num_classes": num_classes}

    print(">>>> total_train_samples: {}, total_test_samples: {}, num_classes: {}".format(len(train), len(test), num_classes))

    with open(save_name, "w") as ff:
        json.dump({"train": train, "test": test, "indices_2_labels": image_classes_rule.indices_2_labels, "info": info}, ff, indent=2)
    return save_name


""" Detection dataset """


def match_detection_labels_dir(image_names, label_path):
    xxs, yys = [], []
    for image_name in tqdm(image_names, "Matching image name with label"):
        # print(image_name)
        yy = glob(os.path.join(label_path, os.path.splitext(os.path.basename(image_name))[0] + ".*"))
        if len(yy) == 0:
            print(">>>> Found none label for:", image_name)
        elif len(yy) > 1:
            print(">>>> Found multi labels for:", image_name)
        else:
            xxs.append(image_name)
            yys.append(yy[0])
    return xxs, yys, None


def match_detection_labels_coco_annotation(image_names, label_path, target_ids=None):
    with open(label_path, "r") as ff:
        aa = json.load(ff)

    if image_names is not None:
        image_names = {os.path.basename(ii): ii for ii in image_names}  # map filename to file path

    image_info_dict = {ii["id"]: ii for ii in aa["images"]}
    rrs = {}
    for ii in aa["annotations"]:
        if target_ids is not None and ii["category_id"] not in target_ids:
            continue

        image_info = image_info_dict[ii["image_id"]]
        file_name = image_info["file_name"]
        file_name = file_name if image_names is None else image_names.get(file_name, None)
        if file_name is None:
            continue

        bbox = ii["bbox"]
        left = bbox[0] / image_info["width"]
        top = bbox[1] / image_info["height"]
        right = (bbox[0] + bbox[2]) / image_info["width"]
        bottom = (bbox[1] + bbox[3]) / image_info["height"]

        rr = rrs.get(file_name, {"label": [], "bbox": []})
        rr["label"].append(ii["category_id"])
        rr["bbox"].append([top, left, bottom, right])
        rrs[file_name] = rr

    indices_2_labels = {int(ii["id"]): ii["name"] for ii in aa["categories"]}
    return list(rrs.keys()), list(rrs.values()), indices_2_labels


def match_detection_labels(image_names, label_path):
    if label_path.endswith(".json"):
        return match_detection_labels_coco_annotation(image_names, label_path)
    else:
        return match_detection_labels_dir(image_names, label_path)


def convert_to_corner_by_format(bbox, bbox_source_format="yxyx"):
    if bbox_source_format == "yxyx":
        return bbox
    if bbox_source_format == "xyxy":
        return [[ii[1], ii[0], ii[3], ii[2]] for ii in bbox]
    if bbox_source_format == "yxhw":
        return [[ii[0], ii[1], ii[0] + ii[2], ii[1] + ii[3]] for ii in bbox]
    if bbox_source_format == "xywh":
        return [[ii[1], ii[0], ii[1] + ii[3], ii[0] + ii[2]] for ii in bbox]
    if bbox_source_format == "cycxhw":
        top_left = [[ii[0] - ii[2] / 2, ii[1] - ii[3] / 2] for ii in bbox]
        return [[ii[0], ii[1], ii[0] + jj[2], ii[1] + jj[3]] for ii, jj in zip(top_left, bbox)]
    if bbox_source_format == "cxcywh":
        top_left = [[ii[1] - ii[3] / 2, ii[0] - ii[2] / 2] for ii in bbox]
        return [[ii[0], ii[1], ii[0] + jj[3], ii[1] + jj[2]] for ii, jj in zip(top_left, bbox)]


def read_coco_objects(label_path):
    with open(label_path, "r") as ff:
        cc = [ii.strip().split(" ") for ii in ff.readlines()]
    label = [ii[0] for ii in cc]
    bbox = [[float(jj) for jj in ii[1:]] for ii in cc]
    return {"label": label, "bbox": bbox}


def convert_bbox_labels(objects, label_convert_func=int, bbox_source_format="yxyx"):
    label = [label_convert_func(ii) for ii in objects["label"]]
    bbox = convert_to_corner_by_format(objects["bbox"], bbox_source_format)
    return {"label": label, "bbox": bbox}


def build_detection_dataset_json(
    train_image_path, train_label_path, test_image_path=None, test_label_path=None, test_split=0.0, bbox_source_format="yxyx", save_name=None
):
    if save_name is None:
        save_name = os.path.basename(os.path.dirname(train_image_path)) + ".json"
    elif not save_name.endswith(".json"):
        save_name += ".json"

    """ Read training data """
    x_train = walk_through_image_folder(train_image_path, depth=1)
    x_train, y_train, indices_2_labels = match_detection_labels(x_train, train_label_path)

    """ Read or split test data """
    if test_image_path is not None:
        x_test = walk_through_image_folder(test_image_path, depth=1)
        test_label_path = train_label_path if test_label_path is None else test_label_path
        x_test, y_test, _ = match_detection_labels(x_test, test_label_path)
    else:
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_split, random_state=42)

    """ Read bbox + label data """
    x_train = [os.path.abspath(ii) for ii in x_train]
    x_test = [os.path.abspath(ii) for ii in x_test]
    train = [{"image": ii, "objects": jj if isinstance(jj, dict) else read_coco_objects(jj)} for ii, jj in zip(x_train, y_train)]
    test = [{"image": ii, "objects": jj if isinstance(jj, dict) else read_coco_objects(jj)} for ii, jj in zip(x_test, y_test)]
    # num_classes = max([max(ii["objects"]["label"]) for ii in train]) + 1

    """ Convert label string to int, convert bbox to corner [top, left, bottom, right] format """
    all_labels = set()
    for ii in train:
        all_labels.update(set(ii["objects"]["label"]))
    is_digit_labels = min([isinstance(ii, int) or ii.isdigit() for ii in all_labels])  # Check if all True
    if indices_2_labels is not None:  # Already inited from annotation json file
        label_convert_func = lambda xx: xx
        num_classes = max(indices_2_labels.keys()) + 1
    elif is_digit_labels:
        label_convert_func = int
        indices_2_labels = {int(ii): ii for ii in all_labels}
        num_classes = max(indices_2_labels.keys()) + 1
    else:
        all_labels = sorted(list(all_labels))
        labels_2_indices = {ii: id for id, ii in enumerate(all_labels)}
        indices_2_labels = {vv: kk for kk, vv in labels_2_indices.items()}
        label_convert_func = lambda xx: labels_2_indices.get(xx, -1)
        num_classes = len(indices_2_labels)
    train = [{"image": ii["image"], "objects": convert_bbox_labels(ii["objects"], label_convert_func, bbox_source_format)} for ii in train]
    test = [{"image": ii["image"], "objects": convert_bbox_labels(ii["objects"], label_convert_func, bbox_source_format)} for ii in test]

    """ Write target json file """
    info = {"num_classes": num_classes}
    print(">>>> total_train_samples: {}, total_test_samples: {}, num_classes: {}".format(len(train), len(test), num_classes))

    with open(save_name, "w") as ff:
        json.dump({"train": train, "test": test, "indices_2_labels": indices_2_labels, "info": info}, ff, indent=2)
    return save_name


def parse_arguments(argv):
    import argparse

    BBOX_FORMAT = {
        "yxyx": "[top, left, bottom, right]",
        "xyxy": "[left, top, right, bottom]",
        "yxhw": "[top, left, height, width]",
        "xywh": "[left, top, width, height]",
        "cycxhw": "[center_height, center_width, height, width]",
        "cxcywh": "[center_width, center_height, width, height]",
    }
    BBOX_FORMAT_STR = ", ".join([kk + " -> " + vv for kk, vv in BBOX_FORMAT.items()])

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_images", required=True, type=str, help="Train images path")
    parser.add_argument("--test_images", type=str, default=None, help="Test images path")
    parser.add_argument("--test_split", type=float, default=0, help="Test split if `test_images` is None")
    # parser.add_argument("--is_int_label", action="store_true", help="[Recognition] convert label by int, or will be sorted and enumerated label")
    parser.add_argument(
        "--train_labels", type=str, default=None, help="[Detection] train bbox + label path, can be directory or COCO format annotation json file"
    )
    parser.add_argument(
        "--test_labels",
        type=str,
        default=None,
        help="[Detection] test bbox + label path. None for same with `train_labels`, can be directory or COCO format annotation json file",
    )
    parser.add_argument("--bbox_source_format", type=str, default="yxyx", help="[Detection] Bbox source format: " + BBOX_FORMAT_STR)
    parser.add_argument("-s", "--save_name", type=str, default=None, help="Target json file save name")

    args = parser.parse_known_args(argv)[0]
    assert args.test_images or args.test_split
    assert args.bbox_source_format in BBOX_FORMAT

    while args.train_images.endswith(os.sep):
        args.train_images = args.train_images[:-1]
    while args.test_images is not None and args.test_images.endswith(os.sep):
        args.test_images = args.test_images[:-1]
    while args.train_labels is not None and args.train_labels.endswith(os.sep):
        args.train_labels = args.train_labels[:-1]
    while args.test_labels is not None and args.test_labels.endswith(os.sep):
        args.test_labels = args.test_labels[:-1]

    return args


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    if args.train_labels is None:
        save_name = build_recognition_dataset_json(args.train_images, args.test_images, args.test_split, args.save_name)
    else:
        save_name = build_detection_dataset_json(
            args.train_images, args.train_labels, args.test_images, args.test_labels, args.test_split, args.bbox_source_format, args.save_name
        )
    print(">>>> Saved to:", save_name)
