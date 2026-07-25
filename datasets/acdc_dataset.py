import os
from glob import glob
from sklearn.model_selection import train_test_split

def get_data_dicts(args):
    """
    獲取 ACDC 資料集的訓練與驗證集字典列表。
    預設路徑結構: args.data_root / acdc / train / patientXXX /
    """
    data_dir = os.path.join(args.data_root, 'acdc', 'train') 
    patient_folders = sorted(glob(os.path.join(data_dir, 'patient*')))
    
    data_dicts = []
    
    for folder in patient_folders:
        # 只尋找結尾是 _gt.nii.gz 的檔案 (確保只拿到 ED 和 ES 的 3D 標註)
        gt_files = glob(os.path.join(folder, '*_gt.nii.gz'))
        
        for gt_path in gt_files:
            # 影像檔名就是標註檔名拿掉 '_gt'
            img_path = gt_path.replace('_gt.nii.gz', '.nii.gz')
            
            if os.path.exists(img_path):
                data_dicts.append({
                    'image': img_path,
                    'label': gt_path
                })
                
    # 劃分訓練集與驗證集 (例如 80% 訓練, 20% 驗證)
    # 若您有預設的 split 方式 (如 k-fold)，可在此處修改
    train_files, val_files = train_test_split(
        data_dicts, 
        test_size=0.2, 
        random_state=args.seed if hasattr(args, 'seed') else 42
    )
    
    return train_files, val_files