# Inference with trained LoRA model
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml
from PIL import Image

# Load configuration
config = load_config_from_yaml("configs/face_seg_config.yaml")
config.model.lora.pretrained_weight = "/path/to/your/lora/weights.safetensors"
# Initialize trainer (LoRA will be loaded automatically in setup_predict)
trainer = QwenImageEditTrainer(config)

# Setup for inference
trainer.setup_predict()

# Load input image
input_image = Image.open("data/face_seg/control_images/060002_4_028450_FEMALE_30.jpg")

# Generate face segmentation
result = trainer.predict(
    prompt_image=input_image,
    prompt="change the image from the face to the face segmentation mask",
    num_inference_steps=20,
    true_cfg_scale=4.0
)
# show the image
result[0]
# Save result
result[0].save("output_segmentation.png")
print("Generated face segmentation saved as output_segmentation.png")