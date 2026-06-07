import os
import time
import importlib
from pathlib import PurePath

import torch

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    AddChannel,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim,
)

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
    SurfaceDistanceMetric,
    MeanIoU,
    SurfaceDistanceMetric,
    SurfaceDiceMetric,
    get_confusion_matrix,
    compute_confusion_matrix_metric,
    compute_average_surface_distance,
)

import surface_distance

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data["image"].to(device))
        output = torch.argmax(output, dim=1)
    return output


def check_channel(inp):
    # check shape is 5
    add_ch = AddChannel()
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 4:
        inp = add_ch(inp)
    if len_inp_shape == 3:
        inp = add_ch(inp)
        inp = add_ch(inp)
    return inp


def compute_surface_dice(pred, label, spacing, threshold_mm=1.0):
    pred_np = pred.cpu().numpy().astype(np.bool_)
    label_np = label.cpu().numpy().astype(np.bool_)
    spacing = np.array(spacing)

    surface_distances = surface_distance.compute_surface_distances(
        label_np, pred_np, spacing
    )

    return surface_distance.compute_surface_dice_at_tolerance(
        surface_distances, threshold_mm
    )


def eval_label_pred(data, cls_num, device):
    # post transform
    post_label = AsDiscrete(to_onehot=cls_num)

    # 定義 metrics
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    hd95_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean", get_not_nans=False
    )
    iou_metric = MeanIoU(include_background=False)
    assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        compute_sample=False,
        reduction="mean",
        get_not_nans=False,
    )

    # 準備資料
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)

    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    val_output_convert = [post_label(val_pred_tensor) for val_pred_tensor in val_pred]

    # =========================================================================
    # [修改重點] 將每個指標獨立包在 try-except 中
    # =========================================================================

    # 1. Dice (通常最快最省，不太會爆，但為了保險也可以包)
    try:
        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    except Exception as e:
        print(f"[Warning] Dice OOM: {e}")
        dc_vals = np.full((len(val_output_convert), cls_num - 1), -1.0).squeeze()

    # 2. HD95 (這次的兇手)
    try:
        hd95_metric(y_pred=val_output_convert, y=val_labels_convert)
        hd95_vals = hd95_metric.get_buffer().detach().cpu().numpy().squeeze()
    except Exception as e:
        print(f"[Warning] HD95 OOM: {e}")
        hd95_vals = np.full((len(val_output_convert), cls_num - 1), -1.0).squeeze()

    # 3. IoU
    try:
        iou_metric(y_pred=val_output_convert, y=val_labels_convert)
        iou_vals = iou_metric.get_buffer().detach().cpu().numpy().squeeze()
    except Exception as e:
        print(f"[Warning] IoU OOM: {e}")
        iou_vals = np.full((len(val_output_convert), cls_num - 1), -1.0).squeeze()

    # 4. ASSD
    try:
        assd_metric(y_pred=val_output_convert, y=val_labels_convert)
        assd_vals = assd_metric.get_buffer().detach().cpu().numpy().squeeze()
    except Exception as e:
        print(f"[Warning] ASSD OOM: {e}")
        assd_vals = np.full((len(val_output_convert), cls_num - 1), -1.0).squeeze()

    # 5. Surface Dice (之前的兇手)
    try:
        surfdice_metric = SurfaceDiceMetric(
            include_background=False,
            class_thresholds=[1.0] * (cls_num - 1),
        )
        surface_dice_vals = (
            surfdice_metric(y_pred=val_output_convert, y=val_labels_convert)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
    except Exception as e:
        print(f"[Warning] Surface Dice OOM: {e}")
        surface_dice_vals = np.full(
            (len(val_output_convert), cls_num - 1), -1.0
        ).squeeze()

    # 6. Confusion Matrix (Sensitivity/Specificity)
    try:
        confusion_metric(y_pred=val_output_convert, y=val_labels_convert)
        confusion_vals = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()

        # 避免維度問題 (如果是單一 batch)
        if confusion_vals.ndim == 1:
            confusion_vals = confusion_vals[np.newaxis, :]

        tp = confusion_vals[:, 0]
        fp = confusion_vals[:, 1]
        tn = confusion_vals[:, 2]
        fn = confusion_vals[:, 3]

        # 避免除以零
        sensitivity_vals = np.divide(
            tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0
        )
        specificity_vals = np.divide(
            tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0
        )

    except Exception as e:
        print(f"[Warning] Confusion Matrix OOM: {e}")
        sensitivity_vals = np.full(
            (len(val_output_convert), cls_num - 1), -1.0
        ).squeeze()
        specificity_vals = np.full(
            (len(val_output_convert), cls_num - 1), -1.0
        ).squeeze()

    # =========================================================================

    return (
        dc_vals,
        hd95_vals,
        iou_vals,
        assd_vals,
        surface_dice_vals,
        sensitivity_vals,
        specificity_vals,
    )


def get_filename(data):
    return PurePath(data["image_meta_dict"]["filename_or_obj"]).parts[-1]


def get_label_transform(data_name, keys=["label"]):
    transform = importlib.import_module(f"transforms.{data_name}_transform")
    get_lbl_transform = getattr(transform, "get_label_transform", None)
    return get_lbl_transform(keys)


def run_infering(model, data, model_inferer, post_transform, args):
    ret_dict = {}

    # test
    start_time = time.time()
    data["pred"] = infer(model, data, model_inferer, args.device)
    end_time = time.time()
    ret_dict["inf_time"] = end_time - start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')

    # post process transform
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])

    # eval infer tta
    if "label" in data.keys():
        (
            tta_dc_vals,
            tta_hd95_vals,
            tta_iou_vals,
            tta_assd_vals,
            tta_surfdice_vals,
            _,
            _,
        ) = eval_label_pred(data, args.out_channels, args.device)
        print("infer test time aug:")
        print("dice:", tta_dc_vals)
        print("hd95:", tta_hd95_vals)
        print("iou:", tta_iou_vals)
        print("assd:", tta_assd_vals)
        print("surface dice:", tta_surfdice_vals)
        ret_dict["tta_dc"] = tta_dc_vals
        ret_dict["tta_hd"] = tta_hd95_vals
        ret_dict["tta_iou"] = tta_iou_vals
        ret_dict["tta_assd"] = tta_assd_vals
        ret_dict["tta_surfdice"] = tta_surfdice_vals

        # post label transform 
        sqz_transform = SqueezeDimd(keys=["label"])
        data = sqz_transform(data)

    # post transform
    data = post_transform(data)

    # eval infer origin
    if "label" in data.keys():
        # get orginal label
        lbl_dict = {"label": data["label_meta_dict"]["filename_or_obj"]}
        label_loader = get_label_transform(args.data_name, keys=["label"])
        lbl_data = label_loader(lbl_dict)

        data["label"] = lbl_data["label"]
        data["label_meta_dict"] = lbl_data["label_meta_dict"]

        (
            ori_dc_vals,
            ori_hd95_vals,
            ori_iou_vals,
            ori_assd_vals,
            ori_surfdice_vals,
            ori_sensitivity_vals,
            ori_specificity_vals,
        ) = eval_label_pred(data, args.out_channels, args.device)
        print("infer test original:")
        print("dice:", ori_dc_vals)
        print("hd95:", ori_hd95_vals)
        print("iou:", ori_iou_vals)
        print("assd:", ori_assd_vals)
        print("surface dice:", ori_surfdice_vals)
        print("sensitivity:", ori_sensitivity_vals)
        print("specificity:", ori_specificity_vals)
        ret_dict["ori_dc"] = ori_dc_vals
        ret_dict["ori_hd"] = ori_hd95_vals
        ret_dict["ori_iou"] = ori_iou_vals
        ret_dict["ori_assd"] = ori_assd_vals
        ret_dict["ori_surfdice"] = ori_surfdice_vals
        ret_dict["ori_sensitivity"] = ori_sensitivity_vals
        ret_dict["ori_specificity"] = ori_specificity_vals

    # =========================================================================
    # [修改] MMWHS 標籤轉換區塊：加上 try-except 防止 OOM
    # =========================================================================
    if args.data_name == "mmwhs":
        try:
            mmwhs_transform = Compose(
                [
                    LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
                    MapLabelValue(
                        orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                        target_labels=[0, 500, 600, 420, 550, 205, 820, 850],
                    ),
                ]
            )
            # 嘗試轉換，如果記憶體不夠會報錯
            data["pred"] = mmwhs_transform(data["pred"])
            print("[Info] MMWHS label mapped successfully.")

        except Exception as e:
            print(f"\n[Warning] MMWHS Label Mapping OOM: {e}")
            print("跳過標籤數值還原，將直接儲存原始類別 (0, 1, 2...)")
            # 這裡不動作，data["pred"] 保持原本的 0,1,2... 讓程式繼續跑下去儲存

    # =========================================================================

    if not args.test_mode:
        # save pred result
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)

        save_img(data["pred"], data["pred_meta_dict"], infer_img_pth)

    return ret_dict
