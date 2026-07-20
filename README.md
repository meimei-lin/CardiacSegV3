# Frequency Domain and Spatial Domain Fusion Image Segmentation

<br>
<div align="center">
    <img src="image/wicnet.png" width="100%">
</div>
<br><br>


## Abstract
Cardiac anatomical structures are highly complex, and the low contrast of computed tomography images makes accurate segmentation of the myocardium and aortic valve a challenging task. Existing U-Net-based architectures often lose high-frequency boundary information during successive downsampling operations. To address this issue, this study proposes WIC-Net, a novel dual-stream architecture that effectively preserves fine anatomical details.
The encoder adopts the InceptionWT block, which consists of two branches. The spatial branch employs InceptionNeXt with large-kernel convolutions to capture global features, while the frequency branch utilizes wavelet convolution to decompose features through discrete wavelet transform and generate frequency-aware attention maps to further enhance spatial features. The decoder integrates a residual coordinate attention block to progressively refine features and accurately guide spatial reconstruction. This strategy not only effectively restores fine anatomical structures but also ensures precise boundary delineation.
Five-fold cross-validation was conducted on the M-WHS-100 and MM-WHS datasets. Experimental results demonstrate that WIC-Net outperforms existing state-of-the-art models across multiple evaluation metrics, particularly in the segmentation of fine anatomical structures, thereby demonstrating its high segmentation accuracy for clinical applications.

## Approach
We propose **WIC-Net**, a dual-domain fusion network for 3D medical image segmentation. To prevent boundary detail loss during downsampling, we integrate the **InceptionWT** block into the encoder to extract high-frequency wavelet features. Additionally, we incorporate the **ResCA** block into the decoder to enhance 3D spatial coordinate perception.

## Setup
### 1. Create environment
```bash
git clone https://github.com/meimei-lin/CardiacSegV3.git
cd CardiacSegV3
conda create -n CardiacSegV3 python=3.9
conda activate CardiacSegV3
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install tabulate==0.9.0
pip install -r requirements.txt
python setup_dir.py
pip install git+https://github.com/deepmind/surface-distance.git
pip install PyWavelets
```


### 2. Dataset deployment
Cheng Hsin General Hospital Dataset: Place the dataset in .\CardiacSeg\dataset\chgh

Fudan University Dataset: Place the dataset in .\CardiacSeg\dataset\mmwhs

### 3. Train
Open /exps/exp_chgh.ipynb or /exps/exp_mmwhs_myo.ipynb
<br>
<div align="center">
    <img src="image/1.png" width="100%">
</div>
<br><br>


### 4. Infer
Open /exps/infer_chgh.ipynb or /exps/infer_mmwhs_myo.ipynb
<div align="center">
    <img src="image/2.png" width="100%">
</div>
<br><br>
