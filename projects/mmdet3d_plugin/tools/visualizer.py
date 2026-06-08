import numpy as np
import torch
import imageio
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from ..fiptr.utils.instance import predict_instance_segmentation_and_trajectories as predict_instance_segmentation_and_trajectories_beverse
from ..fiptr.visualize.motion_visualisation import plot_instance_map

def visualize_flow(pred_flows, gt_flows, img_metas):
    def draw_flow_map(flow_map, is_gt=False, scale=5):
        C, H, W = flow_map.shape
        flow_img = np.full((H * scale, W * scale, 3), 255, dtype=np.uint8)

        cv2.line(
            flow_img,
            (flow_img.shape[1] // 2, 0),
            (flow_img.shape[1] // 2, flow_img.shape[0]),
            (128, 128, 128),
            thickness=1,
        )
        cv2.line(
            flow_img,
            (0, flow_img.shape[0] // 2),
            (flow_img.shape[1], flow_img.shape[0] // 2),
            (128, 128, 128),
            thickness=1,
        )

        for h in range(0, H, 2):
            for w in range(0, W, 2):
                flow = flow_map[:, h, w].copy()
                if np.all(flow == 255):
                    continue

                flow *= scale
                start = np.array([w, h]) * scale
                end = start + flow
                cv2.arrowedLine(
                    flow_img,
                    start.astype(np.int32),
                    end.astype(np.int32),
                    (255, 0, 0),
                    thickness=1,
                    tipLength=0.2,
                    line_type=cv2.LINE_AA,
                )

        flow_img = flip_rotate_image(flow_img)
        flow_img = cv2.copyMakeBorder(flow_img, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return flow_img

    B, T, C, H, W = pred_flows.shape

    pred_flows = pred_flows.detach().cpu().numpy()
    gt_flows = gt_flows.detach().cpu().numpy()

    for b in range(B):
        pred_flow_imgs = []
        gt_flow_imgs = []

        for t in range(T):
            pred_flow_img = draw_flow_map(pred_flows[b, t], is_gt=False)
            gt_flow_img = draw_flow_map(gt_flows[b, t], is_gt=True)

            pred_flow_imgs.append(pred_flow_img)
            gt_flow_imgs.append(gt_flow_img)

        pred_flow_imgs = np.hstack(pred_flow_imgs)
        cv2.putText(
            pred_flow_imgs, "pred_flow_imgs", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 128, 0), thickness=3
        )

        gt_flow_imgs = np.hstack(gt_flow_imgs)
        cv2.putText(gt_flow_imgs, "gt_flow_imgs", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 128, 0), thickness=3)

        flow_imgs = np.vstack((gt_flow_imgs, pred_flow_imgs))

        return flow_imgs

def visualize_motion(motion_targets, motion_preds, model = "fistr", sample_idx=None):
    if model == "beverse":
        segmentation_binary = motion_targets['segmentation']
        segmentation = segmentation_binary.new_zeros(
                segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)

        motion_labels = dict()
        motion_labels['segmentation'] = segmentation.float() * 10
        motion_labels["instance"] = motion_targets["instance"]
        motion_labels['instance_center'] = motion_targets['centerness']
        motion_labels['instance_offset'] = motion_targets['offset']
        motion_labels['instance_flow'] = motion_targets['flow']
        gt_image = plot_motion(motion_labels, "beverse")
        pred_image = plot_motion(motion_preds, "beverse")
    elif model == "fistr":
        gt_image = plot_motion(motion_targets, "fistr")
        pred_image = plot_motion(motion_preds, "fistr")

    pred_image = flip_rotate_image(pred_image)
    gt_image = flip_rotate_image(gt_image)

    cv2.putText(
        pred_image,
        "pred_motion",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 128, 0),
        thickness=1,
    )
    cv2.putText(
        gt_image,
        "gt_motion",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 128, 0),
        thickness=1,
    )

    final_image = np.hstack((gt_image, pred_image))
    cv2.line(
        final_image,
        (final_image.shape[1] // 2, 0),
        (final_image.shape[1] // 2, final_image.shape[0]),
        (128, 128, 128),
        thickness=1,
    )

    # cv2.imwrite(f"run/debug/{sample_idx}_loss_motion_maps.png", cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    return final_image

def prepare_canvas(canvas_height):
    xmin, xmax, ymin, ymax = -50, 100, -50, 50
    canvas_width = (canvas_height * (ymax - ymin)) // (xmax - xmin)
    bev_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    pixels_per_meter = bev_canvas.shape[0] / (xmax - xmin)

    ego_pos = (int(xmax * pixels_per_meter), int(bev_canvas.shape[1] / 2))
    ego_length = 5 * pixels_per_meter
    ego_width = 2 * pixels_per_meter
    cv2.rectangle(
        bev_canvas,
        (int(ego_pos[1] - ego_width / 2), int(ego_pos[0] - ego_length / 2)),
        (int(ego_pos[1] + ego_width / 2), int(ego_pos[0] + ego_length / 2)),
        (255, 128, 0),
        -1,
    )

    interval = 10 * pixels_per_meter
    dist_marker_color = (128, 128, 128)
    for i in range(-4, 5):
        cv2.line(
            bev_canvas,
            (int(ego_pos[1] + i * interval), 0),
            (int(ego_pos[1] + i * interval), bev_canvas.shape[0]),
            dist_marker_color,
            1,
            cv2.LINE_AA,
        )
    for i in range(-9, 5):
        cv2.line(
            bev_canvas,
            (0, int(ego_pos[0] + i * interval)),
            (bev_canvas.shape[1], int(ego_pos[0] + i * interval)),
            dist_marker_color,
            1,
            cv2.LINE_AA,
        )
    return bev_canvas, pixels_per_meter, ego_pos

def visualize_bev(
    canvas_height, img_metas, bbox_results, gt_bboxes, gt_labels, vis_thresh
):
    bev_canvas, pixels_per_meter, ego_pos = prepare_canvas(canvas_height)

    bbox_results = bbox_results["pts_bbox"]
    pred_lidar_boxes = bbox_results["boxes_3d"]
    pred_labels = bbox_results["labels_3d"]
    pred_scores_3d = bbox_results["scores_3d"]
    pred_score_mask = pred_scores_3d > vis_thresh
    pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
    pred_labels = pred_labels[pred_score_mask]

    if isinstance(gt_bboxes, LiDARInstance3DBoxes):
        gt_lidar_boxes = gt_bboxes
        gt_labels = gt_labels
    else:
        gt_lidar_boxes = gt_bboxes.data[0][0]
        gt_labels = gt_labels.data[0][0]

    # gt
    if len(gt_labels) != 0:
        for label, corners in zip(gt_labels, gt_lidar_boxes.corners):
            bottom_corners = corners[[0, 3, 4, 7]][:, :2]
            bottom_corners *= pixels_per_meter

            corners_px = np.array(
                [(int(ego_pos[1] - c[1]), int(ego_pos[0] - c[0])) for c in bottom_corners]
            )
            # sort corners
            corners_px = corners_px[
                np.argsort(
                    np.arctan2(
                        corners_px[:, 1] - np.mean(corners_px[:, 1]),
                        corners_px[:, 0] - np.mean(corners_px[:, 0]),
                    )
                )
            ]
            cv2.fillConvexPoly(
                bev_canvas,
                corners_px,
                color=(0, 255, 0),
                lineType=cv2.LINE_AA,
            )

    # pred
    lidar2ego_rt = np.eye(4)
    lidar2ego_rt[:3, :3] = img_metas["lidar2ego_rots"]
    lidar2ego_rt[:3, -1] = img_metas["lidar2ego_trans"]

    if len(pred_labels) != 0:
        for label, corners in zip(pred_labels, pred_lidar_boxes.corners):
            ones = np.ones((corners.shape[0], 1))
            corners = np.concatenate([corners.cpu().numpy(), ones], axis=1)
            corners = corners @ lidar2ego_rt.T

            bottom_corners = corners[[0, 3, 4, 7]][:, :2]
            bottom_corners *= pixels_per_meter

            corners_px = np.array(
                [(int(ego_pos[1] - c[1]), int(ego_pos[0] - c[0])) for c in bottom_corners]
            )
            # sort corners
            corners_px = corners_px[
                np.argsort(
                    np.arctan2(
                        corners_px[:, 1] - np.mean(corners_px[:, 1]),
                        corners_px[:, 0] - np.mean(corners_px[:, 0]),
                    )
                )
            ]
            cv2.fillConvexPoly(
                bev_canvas,
                corners_px,
                color=(255, 0, 0),
                lineType=cv2.LINE_AA,
            )

    return bev_canvas

def visualize_det(img_metas, bbox_results, gt_bboxes, gt_labels, vis_thresh):

    img_infos = img_metas['img_info'][-1]

    # prediction
    bbox_results = bbox_results["pts_bbox"]
    pred_lidar_boxes = bbox_results["boxes_3d"]
    pred_labels = bbox_results['labels_3d']
    pred_scores_3d = bbox_results["scores_3d"]
    pred_score_mask = pred_scores_3d > vis_thresh
    pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
    pred_labels = pred_labels[pred_score_mask]

    if isinstance(gt_bboxes, LiDARInstance3DBoxes):
        gt_lidar_boxes = gt_bboxes
        gt_labels = gt_labels
    else:
        gt_lidar_boxes = gt_bboxes.data[0][0]
        gt_labels = gt_labels.data[0][0]

    gt_bbox_color = (0, 255, 0)
    pred_bbox_color = (255, 0, 0)

    pred_imgs = {}
    for cam_type, img_info in img_infos.items():
        img_filename = img_info['data_path']
        img = imageio.imread(img_filename)

        cam2lidar_rt = np.eye(4)
        cam2lidar_rt[:3, :3] = img_info['sensor2lidar_rotation']
        cam2lidar_rt[:3, -1] = img_info['sensor2lidar_translation']
        lidar2cam_rt = np.linalg.inv(cam2lidar_rt)

        lidar2ego_rt = np.eye(4)
        lidar2ego_rt[:3, :3] = img_metas['lidar2ego_rots']
        lidar2ego_rt[:3, -1] = img_metas['lidar2ego_trans']
        ego2lidar_rt = np.linalg.inv(lidar2ego_rt)

        ego2cam_rt = lidar2cam_rt @ ego2lidar_rt
        intrinsic = img_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0],
                :intrinsic.shape[1]] = intrinsic
        lidar2img = (viewpad @ lidar2cam_rt)
        ego2img = (viewpad @ ego2cam_rt)

        if len(pred_lidar_boxes.tensor) == 0:
            img_with_pred = img
        else:
            img_with_pred = draw_lidar_bbox3d_on_img(
                        pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color,  thickness=2)

        img_with_pred = draw_lidar_bbox3d_on_img(
            gt_lidar_boxes,
            img_with_pred,
            ego2img,
            None,
            color=gt_bbox_color,
            thickness=2,
        )
        img_with_pred = cv2.resize(
            img_with_pred,
            (int(img_with_pred.shape[1] / 4), int(img_with_pred.shape[0] / 4)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        cv2.putText(
            img_with_pred,
            cam_type,
            (5, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            thickness=1,
        )
        pred_imgs[cam_type] = img_with_pred

    return pred_imgs

def plot_motion(motion_preds, model):
    if model == "beverse" :
        consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories_beverse(motion_preds, compute_matched_centers=True)
    elif model == "fistr":
        consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(motion_preds, compute_matched_centers=True)
    
    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(consistent_instance_seg[0, 0].cpu().numpy(), instance_map)

    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, tuple(path[t].astype(np.int)), tuple(path[t + 1].astype(np.int)), color, 4)

    # # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]
    return vis_image

def predict_instance_segmentation_and_trajectories(output, compute_matched_centers=False, ):
    """
    返回instance map和当前帧instance在未来的轨迹（如果compute_matched_centers=True）。instance map的shape为(1, seq_len, H, W)，轨迹是一个字典，key为instance id，value为一个长度为seq_len的列表，每个元素是一个二元组(x, y)，表示该instance在对应帧的中心位置。
    """
    preds = output['segmentation'].detach()

    batch_size, seq_len = preds.shape[:2]
    pred_inst = output["instance"]
    consistent_instance_seg = pred_inst
    if compute_matched_centers:
        assert batch_size == 1
        # Generate trajectories
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device),
        ))
        for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

        return consistent_instance_seg, matched_centers
    return consistent_instance_seg

def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for instance_id, global_instance_id in instance_map.items()}

def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)

INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])