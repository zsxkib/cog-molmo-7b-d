# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess

MODEL_CACHE = "models"
BASE_URL = f"https://weights.replicate.delivery/default/molmo-7b-d/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch

def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global load_t5, load_clap, RF, build_model

        model_files = [
            "models--allenai--Molmo-7B-D-0924.tar",
            "modules.tar",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        self.processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        text: str = Input(description="Text prompt or question about the image"),
        top_k: int = Input(
            description="Number of highest probability vocabulary tokens to keep for top-k-filtering",
            default=50,
            ge=1,
            le=100,
        ),
        top_p: float = Input(
            description="Cumulative probability for top-p-filtering",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=200,
            ge=1,
            le=1000,
        ),
        temperature: float = Input(
            description="Randomness in token selection (higher values increase randomness)",
            default=1.0,
            ge=0.1,
            le=2.0,
        ),
        length_penalty: float = Input(
            description="Exponential penalty to the length (values < 1.0 encourage shorter outputs, > 1.0 encourage longer outputs)",
            default=1.0,
            ge=0.1,
            le=2.0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        # Process the image and text
        inputs = self.processor.process(images=[Image.open(image)], text=text)

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # Create GenerationConfig with the input parameters
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=length_penalty,
            stop_strings="<|endoftext|>",
        )

        # Generate output
        output = self.model.generate_from_batch(
            inputs, gen_config, tokenizer=self.processor.tokenizer
        )

        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text
