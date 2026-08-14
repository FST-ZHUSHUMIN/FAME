# [TMM 2026] FAME: Mask-Guided Multimodal Fashion Attribute Editing

<img width="1255" height="840" alt="image" src="https://github.com/user-attachments/assets/7401a118-7804-4406-8ac4-b170396a1227" />


## Environment configuration
```bash
# Create and activate conda environment
conda env create -f environment_full.yml
conda activate qwen-train
```

## Training with FAME
Before training please download [FAME data](https://drive.google.com/file/d/1IQcS0IdQZLoLBGvOinlM1UawJImJZqTy/view?usp=drive_link), which includes MM-FAME and test datasets (AFED_H_Product and In-the-wild).

Then change the path config from /FAME/configs/1_fame_config.yaml

```bash
python -m qflux.main --config /FAME/configs/1_fame_config.yaml
```

## Inference with pretrained checkpoint in 
/FAME/chechpoint/ provides the pretrained chechpoint for inference.
```bash
python -m tests.1_fame_mask_retrain
```

## Acknowledgments
This implementation is built based on [qwen-image-finetune](https://github.com/tsiendragon/qwen-image-finetune)

This study was partially supported by the Laboratory for Artificial Intelligence in Design, the InnoHK Initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government.
This work was also partially supported by a grant from the Research Grants Council of the Hong Kong, SAR.(Project No. PolyU/RGC
Project PolyU 25211424).
