import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# scale_factor = 4cd 
def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)
    
    # Perform bilinear interpolation
    # image = F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)


    ref = np.array(Image.open(mask).convert("RGB"))
    ref = ref.astype(np.float32)/255.0
    ref = ref[None].transpose(0,3,1,2)
    ref = torch.from_numpy(ref)
    # mask = mask[None,None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)

    # mask = np.array(Image.open(mask).convert("L"))
    # mask = mask.astype(np.float32)/255.0
    # mask = mask[None,None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)

    masked_image = (1-ref)*image

    # batch = {"image": image, "mask": mask, "masked_image": masked_image}
    batch = {"image": image, "ref": ref}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_SR.png")))
    refs = sorted(glob.glob(os.path.join(opt.indir, "*_ref.png")))
    images = [x.replace("_SR.png", ".png") for x in masks]
    # images = masks
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/ref-SR/config.yaml")
    # config = OmegaConf.load("configs/latent-diffusion/celebahq-ldm-vq-4.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("/home/masi/VS_Projects/latent-diffusion/logs/2024-07-17T20-27-56_celebahq-ldm-vq-4/checkpoints/last.ckpt")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            
            for image, mask in tqdm(zip(images, refs)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["ref"])
                # cc = model.cond_stage_model.encode(batch["image"])

                # c = model.get_learned_conditioning(batch["ref"])
                cc = torch.nn.functional.interpolate(batch["image"],
                                                     size=c.shape[-2:]) 
                # c = torch.cat((c, cc), dim=1)
                # c = c * cc
                # shape = (c.shape[1]-3,)+c.shape[2:]
                # reshaped_c = c[:, :3, :, :]   
                shape = (cc.shape[1],)+cc.shape[2:]

                # samples_ddim, _ = model.sample_log(cond=c, batch_size=1, ddim=True,
                #                             ddim_steps=50, quantize_denoised=True, eta=1.)

                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                #  quantize_denoised=True,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                
                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["ref"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                # sample_image = predicted_image.squeeze().cpu().numpy()  # Assuming the output shape is (batch_size, channels, height, width)
                # sample_image = (sample_image.transpose(1, 2, 0) * 255).astype('uint8')  # Assuming pixel values are in range [0, 1]
                # sample_image = Image.fromarray(sample_image)

                # # Save the image
                # sample_image.save(outpath)
                # # inpainted = (1-mask)*image+mask*predicted_image
                inpainted = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)



              


                # seg = batch["image"].to('cuda').float()
                # seg = model.get_learned_conditioning(seg)

                # samples, _ = model.sample_log(cond=seg, batch_size=1, ddim=True,
                #                             ddim_steps=50, eta=1.)

                # samples = model.decode_first_stage(samples)

                # # Assuming samples are in tensor format
                # # Convert tensor to numpy array and then to PIL image
                # sample_image = samples.squeeze().cpu().numpy()  # Assuming the output shape is (batch_size, channels, height, width)
                # sample_image = (sample_image.transpose(1, 2, 0) * 255).astype('uint8')  # Assuming pixel values are in range [0, 1]
                # sample_image = Image.fromarray(sample_image)

                # # Save the image
                # sample_image.save(outpath)
                    


########################################################## OLD

                # encode masked image and concat downsampled mask
                # c = model.cond_stage_model.encode(batch["masked_image"])
                # cc = torch.nn.functional.interpolate(batch["mask"],
                #                                      size=c.shape[-2:])        
                # c = torch.cat((c, cc), dim=1)
                # c = c[:, :3, :, :]

                # shape = (c.shape[1],)+c.shape[2:]
                # samples_ddim, _ = sampler.sample(S=opt.steps,
                #                                  conditioning=c,
                #                                  batch_size=c.shape[0],
                #                                  shape=shape,
                #                                  verbose=False)
                # x_samples_ddim = model.decode_first_stage(samples_ddim)

                # image = torch.clamp((batch["image"]+1.0)/2.0,
                #                     min=0.0, max=1.0)
                # mask = torch.clamp((batch["mask"]+1.0)/2.0,
                #                    min=0.0, max=1.0)
                # predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                #                               min=0.0, max=1.0)

                # inpainted = (1-mask)*image+mask*predicted_image
                # inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                # Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
