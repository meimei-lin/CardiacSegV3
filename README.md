# Frequency Domain and Spatial Domain Fusion Image Segmentation

<br>
<div align="center">
    <img src="image/wicnet.png" width="100%">
</div>
<br><br>


## Abstract
Due to complex cardiac anatomy and low image contrast, accurate segmentation of the myocardium and aortic valve in computed tomography (CT) images remains a major challenge. Current mainstream U-Net-based structures often lose high-frequency boundary information during continuous downsampling. To address this issue, this study proposes a new model WIC-Net, a novel dual-stream architecture that effectively preserves fine anatomical details. Regarding the model design, the encoder employs the InceptionWT block, which combines two branches: the spatial branch utilizes InceptionNeXt with large-kernel convolutions to capture global features, while the frequency branch adopts Wavelet Convolutions (WTConv). WTConv decomposes features via Discrete Wavelet Transform (DWT) and generates a frequency-aware attention map to further enhance the spatial features. The decoder integrates the Residual Coordinate Attention (ResCA) block to progressively refine feature representations and accurately guide spatial reconstruction. This strategy not only effectively recovers micro-structures but also ensures precise boundary delineation. This study evaluated the proposed model using five-fold cross-validation on both the M-WHS 2025 and MM-WHS datasets. Experimental results show that WIC-Net achieves superior performance compared to existing state-of-the-art models across several key metrics, demonstrating remarkable advantages particularly in the segmentation of small structures. This fully proves its highly accurate segmentation capabilities for clinical applications.

## Approach
We propose **WIT-Net**, a dual-domain fusion network for 3D medical image segmentation. To prevent boundary detail loss during downsampling, we integrate the **InceptionWT** block into the encoder to extract high-frequency wavelet features. Additionally, we incorporate the **ResCA** block into the decoder to enhance 3D spatial coordinate perception.

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

# install diffvg
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

Please be careful that the version of each library is suitable for diffvg. - see [the issue](https://github.com/BachiLi/diffvg/issues/37#issuecomment-1336335574) for details
### Download and setup a font dataset
```bash
python setup_data.py
```

## Finetune CLIP to produce FontCLIP
You can run the processor finetuning using the following command.

__:warning: gwfonts from [the download website](https://www.dgp.toronto.edu/~donovan/font/) lacks more than 50 font files necessary for running the finetuning program. Download them manually. (see https://github.com/yukistavailable/FontCLIP/issues/3)__
```
python train.py --random_prompt_num_per_font 10000 --sample_num 50 --color_jitter_sample_num 200 --use_lora_text
```

## ExCLIP
We have tried several finetuning methods (direct finetuning, [CoOp](https://github.com/KaiyangZhou/CoOp/), [VPT](https://github.com/KMnP/vpt), [LoRA](https://github.com/microsoft/LoRA), and [OFT](https://github.com/Zeju1997/oft)) and integrate them into one Python class named `ExCLIP`. - see [ex_clip.py](models/ex_clip.py) for details

## Licenses
### Main License
This project, based on [CLIP](https://github.com/openai/CLIP/), is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

### Additional Licenses
The source codes for [CoOp](https://github.com/KaiyangZhou/CoOp/) in ExCLIP, is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

The source codes for [VPT](https://github.com/KMnP/vpt) in ExCLIP, is licensed under CC-BY-NC 4.0 License - see the [LICENSE.CC_BY_NC_SA_4.0](LICENSE.CC_BY_NC_SA_4.0.md) for details

The source codes for [OFT](https://github.com/Zeju1997/oft) in ExCLIP, is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

The source codes for vector optimization are based on [Word-As-Image](https://github.com/Shiriluz/Word-As-Image), particularly the files under `optimizer` folder. - see the [LICENSE.CC_BY_NC_SA_4.0](LICENSE.CC_BY_NC_SA_4.0.md) for details


## Attribution
The source codes for vector optimization are based on [Word-As-Image](https://github.com/Shiriluz/Word-As-Image) created by Shiriluz.
The original work can be found at https://github.com/Shiriluz/Word-As-Image and is licensed under CC BY-NC-SA 4.0.
