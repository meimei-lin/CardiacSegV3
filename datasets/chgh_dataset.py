import os


def get_data_dicts(data_dir):
    #patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    # patient_dirs = ['pid_30', 'pid_02', 'pid_8', 'pid_9', 'pid_13', 'pid_15', 'pid_20', 'pid_27', 'pid_29','pid_33','pid_34','pid_44','pid_45','pid_100','pid_119','pid_08', 'pid_08_1', 'pid_27', 'pid_31','pid_40','pid_46', 'pid_52', 'pid_56', 'pid_57', 'pid_106','pid_107','pid_108','pid_110','pid_115','pid_140', 'pid_1000', 'pid_1002', 'pid_1003']
    patient_dirs = ["patient0051","patient0052","patient0053","patient0054","patient0055","patient0056","patient0057","patient0058","patient0059","patient0060",
                    "patient0061","patient0062","patient0063","patient0064","patient0065","patient0066","patient0067","patient0068","patient0069","patient0070",
                    "patient0071","patient0072","patient0073","patient0074","patient0075","patient0076","patient0077","patient0078","patient0079","patient0080",
                    "patient0081","patient0082","patient0083","patient0084","patient0085","patient0086","patient0087","patient0088","patient0089","patient0090",
                    "patient0091","patient0092","patient0093","patient0094","patient0095","patient0096","patient0097","patient0098","patient0099","patient0100"]
    # patient_dirs = ["patient0001","patient0002","patient0003","patient0004","patient0005","patient0006","patient0007","patient0008","patient0009","patient0010",
    #                 "patient0011","patient0012","patient0013","patient0014","patient0015","patient0016","patient0017","patient0018","patient0019","patient0020",
    #                 "patient0021","patient0022","patient0023","patient0024","patient0025","patient0026","patient0027","patient0028","patient0029","patient0030",
    #                 "patient0031","patient0032","patient0033","patient0034","patient0035","patient0036","patient0037","patient0038","patient0039","patient0040",
    #                 "patient0041","patient0042","patient0043","patient0044","patient0045","patient0046","patient0047","patient0048","patient0049","patient0050"]

    # patient_dirs = ["patient0051"]
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}_gt.nii.gz'))
        })

    return data_dicts
