"""Reusable loader for the pretrained LeWorldModel (LeWM) checkpoint. Works on Colab/Linux or local.
Requires: transformers==4.x (5.x renames the ViT keys and the encoder silently loads as random),
stable_pretraining, and the cloned le-wm repo (for jepa.py / module.py) on sys.path."""
import sys
import json
import torch
from huggingface_hub import hf_hub_download
import stable_pretraining as spt


def load_lewm(lewm_dir="le-wm", repo="quentinll/lewm-pusht", device="cpu"):
    """Build LeWM from config + load pretrained weights. Returns (model.eval(), cfg)."""
    if lewm_dir not in sys.path:
        sys.path.insert(0, lewm_dir)
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    wp = hf_hub_download(repo, "weights.pt")
    enc = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"], patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"], pretrained=False, use_mask_token=False)
    mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                        hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=enc, predictor=ARPredictor(**clean(cfg["predictor"])),
                 action_encoder=Embedder(**clean(cfg["action_encoder"])),
                 projector=mlp("projector"), pred_proj=mlp("pred_proj"))
    miss, unexp = model.load_state_dict(torch.load(wp, map_location=device, weights_only=False), strict=False)
    assert not miss and not unexp, (
        f"state_dict mismatch (missing {len(miss)}, unexpected {len(unexp)}) -- "
        f"you almost certainly need transformers==4.x; 5.x refactors the ViT key names.")
    return model.to(device).eval(), cfg
