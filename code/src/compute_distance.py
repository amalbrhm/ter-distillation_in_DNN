import torch
import cka
from data_prep import test_loader, device

# IMPORTANT : reconstruire EXACTEMENT le modèle avant load_state_dict
# Ici, adapte l'import selon ton fichier modèle
from resnet2 import resnet20  # ou: from resnet_cifar import resnet20

MODEL_PATH = "models/resnet20_w16.pth"   # adapte au vrai nom
N_MAX = 512                               # nombre d'images pour former Gram (N x N)

# 1) Charger le modèle
model = resnet20(width=16, num_classes=10).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# 2) Définir les couches à hooker
layers = {
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
}

acts = {k: [] for k in layers}

def make_hook(name):
    def hook(module, inp, out):
        # out: (B, C, H, W) -> on pool pour obtenir (B, C)
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        acts[name].append(out.detach().cpu())
    return hook

handles = []
for name, layer in layers.items():
    handles.append(layer.register_forward_hook(make_hook(name)))

# 3) Faire un forward sur N_MAX images
seen = 0
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        _ = model(images)
        seen += images.size(0)
        if seen >= N_MAX:
            break

for h in handles:
    h.remove()

# 4) Concat activations: X \in R^{N x d}
X = {k: torch.cat(v, dim=0)[:N_MAX] for k, v in acts.items()}

# 5) Construire Gram: G = X X^T (N x N)
def gram(feats):
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats @ feats.T

G = {k: gram(X[k]) for k in X}

# 6) CKA + Angular distance
def angular_distance(Ga, Gb):
    c = cka.compute_cka(Ga, Gb, remove_diag=True)      # score CKA
    theta = cka.compute_angular_cka(c)                 # angle (radians)
    return float(theta), float(c)

pairs = [("layer1", "layer2"), ("layer2", "layer3"), ("layer1", "layer3")]

for a, b in pairs:
    theta, cscore = angular_distance(G[a], G[b])
    print(f"{a} vs {b} | CKA={cscore:.4f} | Angular={theta:.4f} rad")
