import os
import gzip
import re
import itertools
import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score


def load_fasta(path):
    # Lê um arquivo FASTA normal ou .gz e retorna lista de (header, seq)
    open_func = gzip.open if path.endswith(".gz") else open
    seqs = []
    with open_func(path, "rt") as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            seqs.append((header, "".join(seq)))
    return seqs


def get_class(header):
    parts = header.split()
    return parts[1] if len(parts) > 1 else parts[0]


AMINO = list("ACDEFGHIKLMNPQRSTVWY")
PAIR_LIST = list(itertools.product(AMINO, AMINO))

def build_feature_index(skips):
    # Cria um índice para cada (skip, par_de_aminoácidos)
    fmap = {}
    k = 0
    for s in skips:
        for p in PAIR_LIST:
            fmap[(s, p)] = k
            k += 1
    return fmap



def seq_to_features(seq, skips, fmap):
    # Transforma uma sequência em vetor binário indicando presença de kmers
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    L = len(seq)
    vec = np.zeros(len(fmap), dtype=np.uint8)

    for s in skips:
        for i in range(L - (s + 1)):
            a = seq[i]
            b = seq[i + s + 1]
            pair = (a, b)
            if pair in PAIR_LIST:
                idx = fmap[(s, pair)]
                vec[idx] = 1

    return vec


def build_matrix(records, skips):
    fmap = build_feature_index(skips)

    X = []
    y = []
    ids = []

    for h, seq in records:
        X.append(seq_to_features(seq, skips, fmap))
        y.append(get_class(h))
        ids.append(h)

    return np.array(X), y, ids


def reduce_pca(X, n=300):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    p = PCA(n_components=min(n, X.shape[1]), random_state=42)
    return p.fit_transform(Xs)


def internal_metrics(Xpca, labels):
    unique = set(labels)
    k = len(unique) - (1 if -1 in unique else 0)

    if k < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}

    return {
        "silhouette": silhouette_score(Xpca, labels),
        "calinski": calinski_harabasz_score(Xpca, labels),
        "davies": davies_bouldin_score(Xpca, labels)
    }


def f1_macro(y_true, labels):
    uniq = {c: i for i, c in enumerate(set(y_true))}
    y_true_num = np.array([uniq[c] for c in y_true])

    cl_to_class = {}
    for cl in set(labels):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            continue
        majority = Counter(y_true_num[idx]).most_common(1)[0][0]
        cl_to_class[cl] = majority

    y_pred = np.array([cl_to_class.get(cl, -1) for cl in labels])
    valid = y_pred != -1

    return f1_score(y_true_num[valid], y_pred[valid], average="macro")


def run_clusterings(Xpca):
    results = []

    tests = [
        ("KMeans", KMeans(n_clusters=20, n_init="auto", random_state=42)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=20)),
        ("Birch", Birch(n_clusters=20)),
        ("GMM", GaussianMixture(n_components=20, random_state=42)),
        ("DBSCAN", DBSCAN(eps=1.0, min_samples=10))
    ]

    for name, model in tests:
        print(f"\n🔹 Rodando {name} ...")
        labels = model.fit_predict(Xpca)
        metrics = internal_metrics(Xpca, labels)
        results.append((name, labels, metrics))

    return results


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fasta_file = os.path.join(BASE_DIR, "astral.fa")

    print("🔸 Lendo FASTA...")
    records = load_fasta(fasta_file)
    print("✔ Total de sequências:", len(records))

    # skips leves para evitar explosão de memória
    skips = [0, 1]

    print("\n🔸 Construindo matriz de atributos (K-mers)...")
    X, y, ids = build_matrix(records, skips)
    print("✔ Matriz X:", X.shape)

    print("\n🔸 Aplicando PCA (300 componentes)...")
    Xpca = reduce_pca(X)
    print("✔ PCA OK:", Xpca.shape)

    print("\n🔸 Rodando clusterizações...")
    results = run_clusterings(Xpca)

    print("\n\n=========== RESULTADOS ===========\n")
    for name, labels, m in results:
        f1 = f1_macro(y, labels)
        print(f"\nAlgoritmo: {name}")
        print("Métricas internas:")
        print("  Silhouette:", m["silhouette"])
        print("  Calinski:", m["calinski"])
        print("  Davies:", m["davies"])
        print("F1 externo:", f1)
