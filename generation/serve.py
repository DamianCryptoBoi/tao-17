from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time

from omegaconf import OmegaConf

from DreamGaussianLib import GaussianProcessor, ModelsPreLoader, HDF5Loader
from utils.video_utils import VideoUtils

import requests
import numpy as np
from PIL import Image
from typing import Optional
from functools import lru_cache
import base64
import threading
from diffusers import DiffusionPipeline, DDIMScheduler
import torch
from pydantic import BaseModel
from io import BytesIO
from huggingface_hub import hf_hub_download

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/image_sai.yaml")
    return parser.parse_args()

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"


class SampleInput(BaseModel):
    prompt: str

class DiffUsers:
    def __init__(self):

        print("setting up model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        ## n step lora
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 8
        self.guidance_scale = 5

        self._lock = threading.Lock()
        print("model setup done")

    def generate_image(self, prompt: str):
        generator = torch.Generator(self.device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)
        image = self.pipeline(
            prompt="3d model of " + prompt + ", white background",
            negative_prompt="worst quality, low quality",
            num_inference_steps=self.steps,
            generator=generator,
            guidance_scale=self.guidance_scale,
        ).images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        image = base64.b64encode(buf.read()).decode()
        return {"image": image}
    
    def sample(self, input: SampleInput):
        try:
            with self._lock:
                return self.generate_image(input.prompt)
        except Exception as e:
            print(e)
            with self._lock:
                return self.generate_image(input.prompt)


args = get_args()
app = FastAPI()
diffusers = DiffUsers()


def get_config() -> OmegaConf:
    config = OmegaConf.load("configs/image_sai.yaml")
    return config


def get_models(config: OmegaConf = Depends(get_config)):
    return ModelsPreLoader.preload_model(config, "cuda")


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    config: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    best_score = 0
    best_buffer = ""

    for i in range(10):
        buffer = await _generate(models, config, prompt)
        buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")

        response = requests.post("http://localhost:8094/validate/", json={"prompt": prompt, "data": buffer})

        # Check if the request was successful
        if response.status_code == 200:
            print("Data sent successfully!")
            score = response.json().get("score", 0)
            print(f"Score: {score} in attempt {i + 1}")
            if score >= 0.8:
                print("Score is high enough, stopping")
                return buffer
            if score > best_score and score > 0.6:
                best_score = score
                best_buffer = buffer
            elif score < 0.6:
                print("Score is too low, trying again")
        else:
            print(f"Failed to send data: {response.text}")

    # If the loop completes without returning, return the buffer with the best score
    if best_score > 0.6:
        print(f"Did not receive a high enough score after 10 attempts, returning buffer with best score: {best_score}")
        return best_buffer
    else:
        print("Did not receive a score greater than 0.6 after 10 attempts, returning empty buffer")
        return ""


def get_img_from_prompt(prompt:str=""):
    data = diffusers.sample(SampleInput(prompt=prompt))
    return data["image"]

async def _generate(models: list, opt: OmegaConf, prompt: str) -> BytesIO:
    try:
        start_time = time()
        print("Trying to get image from diffusers")
        img = get_img_from_prompt(prompt)
        print("Got image from diffusers")
        gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt="", base64_img = img)
        processed_data = gaussian_processor.train(models, opt.iters)
        hdf5_loader = HDF5Loader.HDF5Loader()
        buffer = hdf5_loader.pack_point_cloud_to_io_buffer(*processed_data)
        print(f"[INFO] It took: {(time() - start_time) / 60.0} min")
        return buffer
    except Exception as e:
        print(e)
        return ""


@app.post("/generate_raw/")
async def generate_raw(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    buffer = await _generate(models, opt, prompt)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/generate_model/")
async def generate_model(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
) -> Response:
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    buffer = BytesIO()
    gaussian_processor.get_gs_model().save_ply(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    processed_data = gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    buffer = video_utils.render_video(*processed_data)

    return StreamingResponse(content=buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
