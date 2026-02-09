import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import cka  # cka.py
from data_prep import valid_loader, test_loader, device
from resnet2 import resnet20 , resnet32


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
    v = cka.to_vector_torch(Gc, kdiag=False, upper=False)                 # cka.py
    v = v.reshape(-1)  # <-- force 1D

    v = v / (torch.norm(v) + 1e-12)
    return v


def angular_distance(v1: torch.Tensor, v2: torch.Tensor) -> float:
    cos = torch.dot(v1, v2).clamp(-1.0, 1.0)
    return float(torch.acos(cos))


# -------------------------
# Extraire un subset FIXE de m exemples (images + labels)
# -------------------------
def take_m_examples(loader, m: int):
    xs, ys = [], []
    seen = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= m:
            break
    X = torch.cat(xs, dim=0)[:m]
    Y = torch.cat(ys, dim=0)[:m]
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
            acts[name].append(out2.detach().cpu())
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in steps]

    # Forward en batchs (on réutilise X_imgs)
    with torch.no_grad():
        # on passe tout d'un coup si ça passe en mémoire, sinon mini-batch
        B = 128
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



# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepth", type=int, required=True, default=20)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--mds_dim", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--split", type=str, default="test", choices=["test", "valid"])

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loader = test_loader if args.split == "test" else valid_loader

    # Fixer m exemples identiques pour tout (input/target + activations)
    X_imgs, y = take_m_examples(loader, args.m)

    # Charger le modèle
    if args.deepth == 20:
        model = resnet20(width=args.width, num_classes=10).to(device)
    elif args.deepth == 32:
        model = resnet32(width=args.width, num_classes=10).to(device)

    state = torch.load(args.model_path, map_location=device)
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
        
    
    embeddings.append(embedding_target(y, num_classes=10))
    emb_names.append("target")
    print("embeddings names " ,emb_names)
    #print("embeddings" , embeddings)
    
    # Distances
    L = len(embeddings)
    print(L)
    D = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            D[i, j] = angular_distance(embeddings[i], embeddings[j])
            
    if args.deepth == 20:
        path = args.out_dir + "20"
    else :
        path = args.out_dir + "32"
        
    heatmap_path = os.path.join(path , "distances_heatmap.png")
    plot_distance_heatmap(D, emb_names, heatmap_path)
    print("Saved:", heatmap_path)

    # MDS -> sphere -> PCA
    Y = classical_mds(D, k=args.mds_dim)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    # PCA 2D + base PCA
    P, V, meanY = pca_2d_with_basis(Y)


    traj_path = os.path.join(path, "trajectory_pc1_pc2.png")
    
    shortest_path_2d = None
        # Convention d'orientation : input à gauche, target à droite
    i_in = emb_names.index("input")
    i_tg = emb_names.index("target")

    # Si target est à gauche de input, on flip PC1
    if P[i_tg, 0] < P[i_in, 0]:
        P[:, 0] *= -1
        V[:, 0] *= -1  # important: pour projeter aussi le plus court chemin

    # Si target est en dessous de input, on flip PC2 (optionnel)
    if P[i_tg, 1] < P[i_in, 1]:
        P[:, 1] *= -1
        V[:, 1] *= -1


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
