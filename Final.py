from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained model
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define your text prompt
prompt = "a photo of an astronaut riding a horse on mars"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("output.png")


