import os

import torch
from tqdm import tqdm

from ssa.utils.logging import get_log

log = get_log(__file__)


def filter_objects(obj, obj_bounding_box, instance_score, filter_list):
    """
    Filter objects, bounding boxes, and scores to only include objects in filter_list
    """
    filtered_obj = []
    filtered_obj_bounding_box = []
    filtered_instance_score = []

    for i, current_obj_name in enumerate(obj):
        if current_obj_name in filter_list:
            filtered_obj.append(current_obj_name)
            filtered_obj_bounding_box.append(obj_bounding_box[i])
            filtered_instance_score.append(instance_score[i])

    log.debug(f"Objects to filter for: {filter_list}")
    found_objects = []
    for item_to_filter in filter_list:
        if item_to_filter in filtered_obj:
            item_idx_in_filtered = filtered_obj.index(item_to_filter)
            log.debug(
                f"Box ({item_to_filter}): {filtered_obj_bounding_box[item_idx_in_filtered]}"
            )
            found_objects.append(item_to_filter)
        else:
            log.debug(f"Object '{item_to_filter}' not found in detected objects.")

    if not found_objects:
        log.debug(
            f"None of the specified objects {filter_list} found in detected objects."
        )

    # Remove duplicates using IoU calculation for objects with same name
    final_obj = []
    final_bbox = []
    final_scores = []

    for i in range(len(filtered_obj)):
        flag = 0
        for j in range(len(final_obj)):
            if (
                calculate_iou(filtered_obj_bounding_box[i], final_bbox[j])
                and filtered_obj[i] == final_obj[j]  # Compare object names
            ):
                flag = 1
                break
        if flag == 0:
            final_obj.append(filtered_obj[i])
            final_bbox.append(filtered_obj_bounding_box[i])
            final_scores.append(filtered_instance_score[i])

    log.debug(f"Filtered detected objects: {final_obj}")
    log.debug(f"Filtered bounding boxes: {final_bbox}")
    log.debug(f"Filtered instance scores: {final_scores}")

    return final_obj, final_bbox, final_scores


def plot_bboxes(
    img_path,
    obj,
    obj_bounding_box,
    instance_score,
    title=None,
    save_path=None,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    # Load the image
    img = plt.imread(img_path)
    img_basename = os.path.basename(img_path).split(".")[0]
    plt.imshow(img)

    # Set prompt as title if provided
    if title is not None:
        plt.title(title, fontsize=10, wrap=True)

    # Plot bounding boxes with labels and scores
    for i in range(len(obj)):
        if i < len(obj_bounding_box) and i < len(instance_score):
            bbox = obj_bounding_box[i]
            score = (
                instance_score[i].item()
                if hasattr(instance_score[i], "item")
                else instance_score[i]
            )
            label = obj[i]

            # Extract bounding box coordinates
            x1, y1, x2, y2 = bbox

            # Create rectangle
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
            )
            plt.gca().add_patch(rect)

            # Add label with score
            plt.text(
                x1,
                y1 - 5,
                f"{label}: {score:.2f}",
                bbox=dict(facecolor="red", alpha=0.5),
                fontsize=10,
                color="white",
            )

    plt.axis("off")  # Hide axes

    save_name = f"debug_image_{img_basename}.png"
    # Create directory for debug image if it doesn't exist
    if save_path is None:
        debug_dir = os.path.dirname("ssa/thirdparty/saneval/data/examples/debug/")
        os.makedirs(debug_dir, exist_ok=True)
        debug_save_path = f"ssa/thirdparty/saneval/data/examples/debug/{save_name}"
    else:
        debug_save_path = os.path.join(save_path, save_name)
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
    plt.savefig(debug_save_path, bbox_inches="tight")
    plt.close()

    # Print the debug save path
    log.info(f"Debug image saved to: {debug_save_path}")
    return debug_save_path


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(
        0, y2_inter - y1_inter + 1
    )

    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    if (
        iou > 0.9
        or (intersection_area / float(area_bbox1) > 0.9)
        or (intersection_area / float(area_bbox2) > 0.9)
    ):
        return 1
    return 0


def get_data(image_path, transform):
    # Import here to avoid circular import issues
    from ssa.thirdparty.saneval.experts.obj_detection.generate_dataset import (
        Dataset,
        collate_fn,
    )

    batch_size = 1
    dataset = Dataset(image_path, transform, single_image=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    for i, data in enumerate(tqdm(data_loader)):
        return data
