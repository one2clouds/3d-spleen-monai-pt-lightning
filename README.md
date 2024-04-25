# 3d-spleen-monai-pt-lightning

https://docs.wandb.ai/tutorials/monai_3d_segmentation
https://github.com/wandb/examples/blob/89171e83a82fed0b0155cf4a211d65aeb288bb3b/colabs/monai/3d_brain_tumor_segmentation.ipynb#L7



If you get error like this : RuntimeError: applying transform <monai.transforms.io.dictionary.LoadImaged object at 0x7fc1f0a5e9b0>

Do this in terminal:
pip install 'monai[all]==1.1.0'
