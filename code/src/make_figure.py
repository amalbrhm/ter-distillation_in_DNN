import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import re

import cka  # cka.py
from data_prep import valid_loader, test_loader, device
from resnet import resnet20 , resnet32 , resnet56
from vgg import VGG_CIFAR

import gc
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# for paths 
def get_model_path(depth: int, width: int, models_dir: str = "models" , net : str ="resnet") -> str:
    p1 = os.path.join(models_dir, f"{net}_d{depth}_w{width}.pth")
    if os.path.isfile(p1):
        return p1
    
    raise FileNotFoundError(f"Model not found: tried {p1}")


def discover_models(models_dir: str = "models"):
    """Retourne une liste de (depth,width,filepath) en scannant models/"""
    out = []
    r1 = re.compile(r"^resnet_d(\d+)_w(\d+)\.pth$")
    for fn in os.listdir(models_dir):
        if not fn.endswith(".pth"):
            continue
        m = r1.match(fn)
        if m:
            d, w = int(m.group(1)), int(m.group(2))
            out.append((d, w, os.path.join(models_dir, fn)))
    out.sort(key=lambda t: (t[0], t[1]))
    if not out:
        raise RuntimeError(f"No resnet models found in {models_dir}")
    return out


def orient_U(P2: np.ndarray, names: list) -> np.ndarray:
    """
    Fixe les flips PCA pour obtenir une forme en U (convention Lange-like):
    - target à droite de input (PC1)
    - un point "milieu" (layer2.0 si existe sinon médian) en dessous des endpoints (PC2)
    """
    P = P2.copy()
    i_in = names.index("input")
    i_tg = names.index("target")

    # PC1: target à droite
    if P[i_tg, 0] < P[i_in, 0]:
        P[:, 0] *= -1

    # choisir un anchor au "milieu"
    if "layer2.0" in names:
        i_mid = names.index("layer2.0")
    else:
        core = [i for i, n in enumerate(names) if n not in ("input", "target")]
        i_mid = core[len(core)//2] if core else len(names)//2

    # PC2: mid en dessous des endpoints
    end_y = 0.5 * (P[i_in, 1] + P[i_tg, 1])
    if P[i_mid, 1] > end_y:
        P[:, 1] *= -1

    return P

# -------------------------
# MDS + PCA
# -------------------------
def classical_mds(D: np.ndarray, k: int = 20) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    Y = eigvecs[:, :k] * np.sqrt(eigvals[:k] + 1e-12)
    return Y


def pca_2d_with_basis(X: np.ndarray):
    """
    PCA via SVD:
    - retourne la projection 2D
    - retourne aussi la base (2 vecteurs) pour projeter de nouveaux points
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :2]  # base 2D
    P = Xc @ V
    mean = X.mean(axis=0, keepdims=True)
    return P, V, mean



# -------------------------
# Embedding "papier" via cka.py
# -------------------------
def gram_from_features(F: torch.Tensor) -> torch.Tensor:
    F = F - F.mean(dim=0, keepdim=True)
    return F @ F.T


def paper_embedding_from_gram(G: torch.Tensor) -> torch.Tensor:
    Gc = cka.double_center(G, vector_in=False, vector_out=False)          # cka.py
    v = cka.to_vector_torch(Gc, kdiag=False, upper=False)       #cka.py
    v = v.reshape(-1)
    v = v / (torch.norm(v) + 1e-12)
    return v


def angular_distance(v1: torch.Tensor, v2: torch.Tensor) -> float:
    v1 = v1.reshape(-1).float()
    v2 = v2.reshape(-1).float()

    cos = torch.dot(v1, v2).clamp(-1.0, 1.0)
    return float(torch.acos(cos))


# -------------------------
# Extraire un subset FIXE de m exemples (images + labels)
# -------------------------
def take_m_examples(loader, m: int):
    X = torch.empty((m, 3, 32, 32), dtype=torch.float32)
    Y = torch.empty((m,), dtype=torch.long)

    filled = 0
    for x, y in loader:
        b = x.size(0)
        end = min(m, filled + b)
        k = end - filled
        X[filled:end].copy_(x[:k])
        Y[filled:end].copy_(y[:k])
        filled = end
        if filled >= m:
            break

    return X, Y



# -------------------------
# Activations: blocs + options conv1/avgpool
# -------------------------
def collect_activations(model, X_imgs: torch.Tensor, include_conv1= True):
    model.eval()
    model.to(device)

    steps = []   # list of (name, module)
    if include_conv1:
        steps.append(("conv1", model.conv1))

    # BasicBlocks
    for i, b in enumerate(model.layer1):
        steps.append((f"layer1.{i}", b))
    for i, b in enumerate(model.layer2):
        steps.append((f"layer2.{i}", b))
    for i, b in enumerate(model.layer3):
        steps.append((f"layer3.{i}", b))

        
    #print("layers ", steps)
    
    acts = {name: [] for name, _ in steps}

    def make_hook(name):
        def hook(module, inp, out):
            if out.dim() == 4:
                out2 = out.mean(dim=(2, 3))  # GAP -> (B,C)
            else:
                out2 = out
            acts[name].append(out2.detach().to(torch.float32).contiguous().cpu())

        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in steps]

    # Forward en batchs (on réutilise X_imgs)
    with torch.no_grad():
        # on passe tout d'un coup si ça passe en mémoire, sinon mini-batch
        B = 128 # = 64 si m = 512 mais ca marche pas non plus
        for i in range(0, X_imgs.size(0), B):
            xb = X_imgs[i:i+B].to(device)
            _ = model(xb)

    for h in handles:
        h.remove()

    feats = {name: torch.cat(v, dim=0) for name, v in acts.items()}  # (m,d)
    names = [name for name, _ in steps]
    return names, feats


# -------------------------
# Embeddings Input/Target
# -------------------------
def embedding_input(X_imgs: torch.Tensor) -> torch.Tensor:
    # X_imgs: (m,3,32,32) -> (m, 3072)
    F = X_imgs.view(X_imgs.size(0), -1).float()
    G = gram_from_features(F)
    return paper_embedding_from_gram(G)


def embedding_target(y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    # one-hot: (m,C)
    Y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    G = gram_from_features(Y)
    return paper_embedding_from_gram(G)

def slerp_path(a: np.ndarray, b: np.ndarray, n_points: int = 100) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) between 2 unit vectors a and b.
    Returns (n_points, dim).
    """
    # normalise (au cas où)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    cos_omega = np.clip(np.dot(a, b), -1.0, 1.0)
    omega = np.arccos(cos_omega)

    if omega < 1e-8:
        # points quasi identiques -> segment dégénéré
        ts = np.linspace(0.0, 1.0, n_points)
        return (1 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]

    ts = np.linspace(0.0, 1.0, n_points)
    sin_omega = np.sin(omega)

    path = []
    for t in ts:
        p = (np.sin((1 - t) * omega) / sin_omega) * a + (np.sin(t * omega) / sin_omega) * b
        p = p / (np.linalg.norm(p) + 1e-12)  # rester sur la sphère
        path.append(p)
    return np.stack(path, axis=0)

# -------------------------
# Plots
# -------------------------
def plot_distance_heatmap(D: np.ndarray, names, outpath: str):
    plt.figure(figsize=(7.8, 6.8))
    plt.imshow(D, aspect="auto")
    plt.colorbar(label="Distance angulaire (radians)")
    plt.title("Heatmap des distances angulaires (steps × steps)")

    step = max(1, len(names) // 12)
    ticks = list(range(0, len(names), step))
    plt.xticks(ticks, [names[i] for i in ticks], rotation=90, fontsize=8)
    plt.yticks(ticks, [names[i] for i in ticks], fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_trajectory(P: np.ndarray, names, outpath: str, shortest_path_2d: np.ndarray = None):
    plt.figure(figsize=(7.8, 6.8))

    # Trajectoire des steps (ordre)
    plt.plot(P[:, 0], P[:, 1], marker="o", label="Trajectoire (steps)")

    # Chemin le plus court Input -> Target (si fourni)
    if shortest_path_2d is not None:
        plt.plot(shortest_path_2d[:, 0], shortest_path_2d[:, 1],
                 linestyle="--", linewidth=2, label="Plus court chemin (Input → Target)")

    # Mettre en évidence input/target si présents
    if "input" in names:
        i = names.index("input")
        plt.scatter(P[i, 0], P[i, 1], s=80, marker="s", label="Input")
        plt.text(P[i, 0], P[i, 1], " input", fontsize=9)
    if "target" in names:
        i = names.index("target")
        plt.scatter(P[i, 0], P[i, 1], s=80, marker="D", label="Target")
        plt.text(P[i, 0], P[i, 1], " target", fontsize=9)

    # Labels légers pour les autres points (index)
    for i, name in enumerate(names):
        if name not in ("input", "target"):
            plt.text(P[i, 0], P[i, 1], str(i), fontsize=7)

    plt.title("Trajectoire (MDS → PCA) : PC1 vs PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def build_steps(model, include_conv1=True):
    steps = []
    if include_conv1:
        steps.append(("conv1", model.conv1))

    for i, b in enumerate(model.layer1):
        steps.append((f"layer1.{i}", b))
    for i, b in enumerate(model.layer2):
        steps.append((f"layer2.{i}", b))
    for i, b in enumerate(model.layer3):
        steps.append((f"layer3.{i}", b))

    return steps


def collect_one_activation(model, X_imgs: torch.Tensor, module, batch_size: int = 32):
    """
    Fait un forward complet mais ne hooke QUE `module`.
    Retourne F de shape (m, d) sur CPU.
    """
    model.eval()
    model.to(device)

    acts = []

    def hook_fn(_module, _inp, out):
        # out: (B,C,H,W) ou (B,C)
        if out.dim() == 4:
            out2 = out.mean(dim=(2, 3))   # GAP -> (B,C)
        else:
            out2 = out
        acts.append(out2.detach().to(torch.float32).cpu().contiguous())

    handle = module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, X_imgs.size(0), batch_size):
            xb = X_imgs[i:i + batch_size].to(device)
            _ = model(xb)

    handle.remove()

    # Ici la liste contient ~m/batch_size blocs, c’est léger.
    F = torch.cat(acts, dim=0)
    return F


def run_all_models_joint(args):
    print("Running ALL models (JOINT embedding)")

    loader = test_loader if args.split == "test" else valid_loader
    X_imgs, y = take_m_examples(loader, args.m)

    models = discover_models("models")  # list of (depth, width, model_path)

    # 1) Construire la liste globale des embeddings (un embedding par "step")
    global_embeddings = []   # list[torch.Tensor]  (1D vectors)
    global_meta = []         # list[(label, names, local_index)]
    # On stocke aussi, pour reconstruire chaque trajectoire :
    traj_slices = {}         # label -> (start, end, names)

    start = 0

    for depth, width, model_path in models:
        label = f"d{depth}_w{width}"
        print("Processing", label, flush=True)

        # build model
        if depth == 20:
            model = resnet20(width=width, num_classes=10).to(device)
        elif depth == 32:
            model = resnet32(width=width, num_classes=10).to(device)
        elif depth == 56:
            model = resnet56(width=width, num_classes=10).to(device)
        else:
            print("Skipping unsupported depth:", depth)
            continue

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # activations
        names, feats = collect_activations(model, X_imgs)

        # local embeddings list for this model (keep names)
        emb = []
        emb_names = []

        # input
        emb.append(embedding_input(X_imgs).reshape(-1).cpu())
        emb_names.append("input")

        # internal steps
        for n in names:
            G = gram_from_features(feats[n])
            v = paper_embedding_from_gram(G).reshape(-1).cpu()
            emb.append(v)
            emb_names.append(n)

            del G
            del feats[n]
            gc.collect()

        # target
        emb.append(embedding_target(y).reshape(-1).cpu())
        emb_names.append("target")

        # append to global
        for i, v in enumerate(emb):
            global_embeddings.append(v)
            global_meta.append((label, emb_names, i))

        end = start + len(emb)
        traj_slices[label] = (start, end, emb_names)
        start = end

        # IMPORTANT: cleanup per model
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print("Done", label, "steps =", len(emb), flush=True)

    # 2) Construire D_all (NxN)
    N = len(global_embeddings)
    print("Total points =", N)

    D_all = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        vi = global_embeddings[i]
        for j in range(i, N):
            d = angular_distance(vi, global_embeddings[j])
            D_all[i, j] = d
            D_all[j, i] = d

    # 3) MDS global + PCA 2D
    k_eff = min(args.mds_dim, N - 1)
    Y = classical_mds(D_all, k=k_eff)
    # si tu utilises la "sphère" comme dans ton code :
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    P2, V, meanY = pca_2d_with_basis(Y)

    # 4) Plot : toutes les trajectoires dans une seule figure
    out_dir = os.path.join(args.out_dir, "ALL_JOINT")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))

    for label, (a, b, names) in traj_slices.items():
        P = P2[a:b]

        # --- Option: orientation U (si tu veux garder)
        P = orient_U(P, names)

        # Trajectoire des steps
        plt.plot(P[:, 0], P[:, 1], marker="o", linewidth=1, label=label, alpha=0.9)

        # Input / Target markers
        if "input" in names:
            ii = names.index("input")
            plt.scatter(P[ii, 0], P[ii, 1], s=120, marker="s")  # carré
        if "target" in names:
            it = names.index("target")
            plt.scatter(P[it, 0], P[it, 1], s=140, marker="*")  # étoile

        # --- Chemin le plus court (géodésique) Input -> Target dans l'espace MDS global
        if "input" in names and "target" in names:
            ii = names.index("input")
            it = names.index("target")

            # Indices globaux dans Y
            Yin = Y[a + ii]
            Ytg = Y[a + it]

            path_high = slerp_path(Yin, Ytg, n_points=120)        # (120, k_eff)
            path_2d = (path_high - meanY) @ V                      # (120, 2)

            # Option: appliquer le même flip que P si tu veux cohérence stricte
            # (simple: recalculer les flips à partir de P original)
            # Ici on le laisse tel quel (souvent OK visuellement).
            plt.plot(path_2d[:, 0], path_2d[:, 1], linestyle="--", linewidth=2, alpha=0.8)

    plt.legend(fontsize=7)
    plt.title(f"JOINT embedding (MDS+PCA), m={args.m}, split={args.split}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    outpath = os.path.join(out_dir, "trajectory_ALL_joint_pc1_pc2.png")
    plt.savefig(outpath, dpi=200)
    plt.close()


    # (optionnel) sauvegarder D_all pour debug
    np.save(os.path.join(out_dir, "D_all.npy"), D_all)

    print("Saved:", outpath)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet","vgg"])
    parser.add_argument("--depth", type=int, default=None, help="e.g. 20/32/56 (required unless --all)")
    parser.add_argument("--width", type=int, default=16, help="base width (default 16)")
    parser.add_argument("--m", type=int, default=512, help="number of samples for Gram")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--out_dir", type=str, default="figures")
    parser.add_argument("--mds_dim", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="plot all available resnets in one figure")
    
    args = parser.parse_args()

    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.all:
        run_all_models_joint(args)
        return

    loader = test_loader if args.split == "test" else valid_loader

    # Fixer m exemples identiques pour tout (input/target + activations)
    X_imgs, y = take_m_examples(loader, args.m)

    # Charger le modèle
    model_path = get_model_path(args.depth, args.width, net=args.model)

    if args.model == "vgg" :
        model = VGG_CIFAR(d= args.depth)
    elif args.depth == 20:
        model = resnet20(width=args.width, num_classes=10).to(device)
    elif args.depth == 32:
        model = resnet32(width=args.width, num_classes=10).to(device)
    elif args.depth == 56:
        model = resnet56(width=args.width, num_classes=10).to(device)
    else:
        raise ValueError("depth must be 20, 32, or 56 for now")

        
    print("args.deepth " , int(args.depth) ,  " args.width " , int(args.width))
    print(model_path )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    

    # Activations
    names, feats = collect_activations(model, X_imgs)

    embeddings = []
    emb_names = []
    
    embeddings.append(embedding_input(X_imgs))
    emb_names.append("input")
    # Embeddings pour chaque step
    for n in names:
        G = gram_from_features(feats[n])
        v = paper_embedding_from_gram(G)
        embeddings.append(v)
        emb_names.append(n)

        # libèrer memeoire au fur et à mesure
        del G
        del feats[n]
        gc.collect()

        
    embeddings.append(embedding_target(y, num_classes=10))
    emb_names.append("target")
    print("embeddings names " ,emb_names)
    #print("embeddings" , embeddings)
    # Distances
    L = len(embeddings)
    D = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            D[i, j] = angular_distance(embeddings[i], embeddings[j])
            
    path = os.path.join(args.out_dir, f"{args.model}_d{args.depth}_w{args.width}_m{args.m}_{args.split}")
    os.makedirs(path, exist_ok=True)

        
    print("path " , path)
    
    os.makedirs(path, exist_ok=True)

    heatmap_path = os.path.join(path , "distances_heatmap.png")
    plot_distance_heatmap(D, emb_names, heatmap_path)
    print("Saved:", heatmap_path)

    # MDS -> sphere -> PCA
    Y = classical_mds(D, k=args.mds_dim)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    # PCA 2D + base PCA
    P, V, meanY = pca_2d_with_basis(Y)
    P = orient_U(P, emb_names)

    traj_path = os.path.join(path, "trajectory_pc1_pc2.png")
    
    shortest_path_2d = None
    # Convention d'orientation : input à gauche, target à droite
    i_in = emb_names.index("input")
    i_tg = emb_names.index("target")


    # Géodésique sur la sphère dans l'espace MDS (dimension mds_dim)
    path_high = slerp_path(Y[i_in], Y[i_tg], n_points=120)  # (120, mds_dim)

    # Projeter ces points via la MÊME PCA (moyenne + base)
    path_centered = path_high - meanY
    shortest_path_2d = path_centered @ V
    
    plot_trajectory(P, emb_names, traj_path, shortest_path_2d=shortest_path_2d)
    print("Saved:", traj_path)

    np.save(os.path.join(path, "D.npy"), D)
    with open(os.path.join(path, "names.txt"), "w", encoding="utf-8") as f:
        for n in emb_names:
            f.write(n + "\n")


if __name__ == "__main__":
    main()
