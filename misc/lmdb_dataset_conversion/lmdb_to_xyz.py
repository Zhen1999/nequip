"""
Convert an LMDB dataset (pickled samples) to an ASE .xyz file.

Species resolution order:
 1) atomic_numbers
 2) one_hot + --type-map (list of atomic numbers per column)
 3) charges (rounded to nearest int, clamped 1..118)

Example:
python -m nequip.misc.lmdb_dataset_conversion.lmdb_to_xyz \
  --in archive/ts1x-train.lmdb \
  --out archive/ts1x-train.xyz \
  --pos pos --forces forces --energy energy --onehot one_hot --charges charges
"""

import argparse
import os
import lmdb
import pickle
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import torch
from ase import Atoms
from ase.io import write
import ase.data


def as_tensor(x, dtype=None):
    if x is None:
        return None
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t


def get_field(sample: Any, name: Optional[str]):
    if not name:
        return None
    if isinstance(sample, dict):
        return sample.get(name, None)
    return getattr(sample, name, None)


def resolve_atomic_numbers(sample: Any, args: argparse.Namespace, N: int) -> np.ndarray:
    # 1) atomic_numbers
    z = get_field(sample, args.numbers)
    if z is not None:
        return np.asarray(z, dtype=np.int64)
    # 2) one_hot + type_map
    oh = get_field(sample, args.onehot)
    if oh is not None:
        oh_t = torch.as_tensor(oh)
        types = oh_t.argmax(dim=-1).tolist()
        if args.type_map:
            z_list = [int(v) for v in args.type_map]
            return np.asarray([z_list[i] for i in types], dtype=np.int64)
        else:
            # fallback: treat indices as pseudo-types; user may post-map later
            return np.asarray(types, dtype=np.int64)
    # 3) charges (interpreted as Z)
    charges = get_field(sample, args.charges)
    if charges is not None:
        z = np.rint(np.asarray(charges)).astype(np.int64)
        z = np.clip(z, 1, 118)
        return z
    raise KeyError("Cannot resolve atomic numbers: provide --numbers or --onehot (+ --type-map) or --charges")


def sample_to_atoms(sample: Any, args: argparse.Namespace) -> Atoms:
    pos = get_field(sample, args.pos)
    if pos is None:
        raise KeyError("positions not found; specify --pos <key>")
    pos = np.asarray(pos, dtype=float)
    N = pos.shape[0]

    z = resolve_atomic_numbers(sample, args, N)
    # Convert to chemical symbols if needed
    syms = [ase.data.chemical_symbols[int(zz)] if 1 <= int(zz) < len(ase.data.chemical_symbols) else f"X{int(zz)}" for zz in z]

    atoms = Atoms(symbols=syms, positions=pos)

    # Optional: cell and PBC
    cell = get_field(sample, args.cell)
    if cell is not None:
        cell_arr = np.asarray(cell, dtype=float)
        cell_arr = cell_arr.reshape(3, 3) if cell_arr.size == 9 else cell_arr
        atoms.cell = cell_arr
    pbc = get_field(sample, args.pbc)
    if pbc is not None:
        atoms.pbc = np.asarray(pbc).astype(bool)

    # Optional arrays/info
    forces = get_field(sample, args.forces)
    if forces is not None:
        atoms.new_array('forces', np.asarray(forces, dtype=float))
    energy = get_field(sample, args.energy)
    if energy is not None:
        atoms.info['energy'] = float(np.asarray(energy).reshape(-1)[0])

    return atoms


def extract_hessian_flat(sample: Any, args: argparse.Namespace, N: int) -> Optional[np.ndarray]:
    """Return a flattened full Hessian (D*D,) as float64 if present; else None.

    Accepts shapes: (D*D,), (1, D*D), (D, D), or (N,3,N,3).
    """
    key = args.hessian
    if not key:
        return None
    H = get_field(sample, key)
    if H is None:
        return None
    H = np.asarray(H)
    D = N * 3
    if H.ndim == 1 and H.size == D * D:
        flat = H
    elif H.ndim == 2 and H.shape == (1, D * D):
        flat = H.reshape(-1)
    elif H.ndim == 2 and H.shape == (D, D):
        flat = H.reshape(-1)
    elif H.ndim == 4 and H.shape == (N, 3, N, 3):
        flat = H.transpose(0, 1, 2, 3).reshape(-1)
    else:
        raise ValueError(f"Unexpected Hessian shape {H.shape} for N={N} (D={D})")
    return flat.astype(np.float64, copy=False)


def extract_hessian_blocks(sample: Any, args: argparse.Namespace, N: int) -> Optional[np.ndarray]:
    """Return full Hessian as blocks with shape (N,3,N,3) if present; else None.

    Accepts shapes: flat (D*D,), (1,D*D), (D,D), or already (N,3,N,3).
    """
    key = args.hessian
    if not key:
        return None
    H = get_field(sample, key)
    if H is None:
        return None
    H = np.asarray(H)
    D = N * 3
    if H.ndim == 4 and H.shape == (N, 3, N, 3):
        return H.astype(np.float64, copy=False)
    if H.ndim == 1 and H.size == D * D:
        HH = H.reshape(D, D)
    elif H.ndim == 2 and H.shape == (1, D * D):
        HH = H.reshape(D, D)
    elif H.ndim == 2 and H.shape == (D, D):
        HH = H
    else:
        raise ValueError(f"Unexpected Hessian shape {H.shape} for N={N} (D={D})")
    # reshape to (N,3,N,3)
    return HH.reshape(N, 3, N, 3).astype(np.float64, copy=False)


def iter_lmdb(path: str):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=False)
    with env.begin() as txn:
        entries = txn.stat()["entries"]
        for idx in range(entries):
            raw = txn.get(f"{idx}".encode("ascii"))
            if raw is None:
                continue
            yield pickle.loads(raw)


def main():
    p = argparse.ArgumentParser(description="Convert LMDB to ASE XYZ")
    p.add_argument("--in", dest="input", required=True, help="Input LMDB path")
    p.add_argument("--out", dest="output", required=True, help="Output .xyz file path")
    p.add_argument("--pos", required=True, help="Key for positions [N,3]")
    p.add_argument("--numbers", default=None, help="Key for atomic_numbers [N]")
    p.add_argument("--onehot", default=None, help="Key for one-hot species [N,C]")
    p.add_argument("--type-map", nargs="*", help="Atomic numbers per one-hot column, e.g., 1 6 7 8 16")
    p.add_argument("--charges", default=None, help="Key for per-atom charges to interpret as atomic numbers")
    p.add_argument("--forces", default=None, help="Key for forces [N,3]")
    p.add_argument("--energy", default=None, help="Key for total energy scalar")
    p.add_argument("--cell", default=None, help="Key for cell (3x3 or 1x3x3)")
    p.add_argument("--pbc", default=None, help="Key for PBC (bool or [3])")
    p.add_argument("--hessian", default=None, help="Key for Hessian labels to export alongside XYZ")
    p.add_argument(
        "--embed-hessian",
        action="store_true",
        help="Embed full Hessian per-atom in EXTXYZ (default: variable width 'hessian:R:9*N' per frame)",
    )
    # New preferred flag: explicitly pad to n_max
    p.add_argument(
        "--pad-hessian",
        action="store_true",
        help=(
            "When embedding Hessian, pad each frame to dataset-global n_max, i.e., write 'hessian:R:9*n_max'. "
            "Default behavior is no padding (variable width '9*N')."
        ),
    )
    # Backward-compat legacy flag
    p.add_argument(
        "--no-pad-hessian",
        action="store_true",
        help=(
            "[Deprecated] Explicitly disable padding; kept for backward compatibility. "
            "Default is already no padding."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many samples (from the start of the LMDB, after sampling), useful for quick subsets",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Take every k-th sample (k>=1). With --limit, stops after picking that many.",
    )
    p.add_argument(
        "--hessian-out",
        default=None,
        help="Path to save Hessians (.npz). Default: change .xyz to _hessian.npz",
    )

    args = p.parse_args()

    # First pass: collect raw fields and determine n_max
    samples: List[Dict[str, Any]] = []
    Ns: List[int] = []
    picked = 0
    for idx, sample in enumerate(iter_lmdb(args.input)):
        if args.every > 1 and (idx % args.every != 0):
            continue
        pos = get_field(sample, args.pos)
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=float)
        N = pos.shape[0]
        entry: Dict[str, Any] = {
            "pos": pos,
            "z": resolve_atomic_numbers(sample, args, N),
            "cell": get_field(sample, args.cell),
            "pbc": get_field(sample, args.pbc),
            "forces": get_field(sample, args.forces),
            "energy": get_field(sample, args.energy),
            "N": N,
        }
        if args.hessian:
            entry["H_blocks"] = extract_hessian_blocks(sample, args, N)
        samples.append(entry)
        Ns.append(N)
        picked += 1
        if args.limit is not None and picked >= args.limit:
            break

    if len(samples) == 0:
        raise RuntimeError("No valid samples found in LMDB")

    n_max = int(max(Ns))

    # Determine padding mode: default to no padding
    use_pad = bool(getattr(args, "pad_hessian", False))
    if getattr(args, "no_pad_hessian", False):
        use_pad = False

    # Second pass: build Atoms with optional embedded per-atom Hessian
    atoms_list: List[Atoms] = []
    for entry in samples:
        pos = entry["pos"]
        z = entry["z"]
        N = entry["N"]
        syms = [ase.data.chemical_symbols[int(zz)] if 1 <= int(zz) < len(ase.data.chemical_symbols) else f"X{int(zz)}" for zz in z]
        atoms = Atoms(symbols=syms, positions=pos)

        # cell / pbc
        if entry["cell"] is not None:
            cell_arr = np.asarray(entry["cell"], dtype=float)
            cell_arr = cell_arr.reshape(3, 3) if cell_arr.size == 9 else cell_arr
            atoms.cell = cell_arr
        if entry["pbc"] is not None:
            atoms.pbc = np.asarray(entry["pbc"]).astype(bool)

        # forces
        if entry["forces"] is not None:
            atoms.new_array("forces", np.asarray(entry["forces"], dtype=float))

        # energy
        if entry["energy"] is not None:
            atoms.info["energy"] = float(np.asarray(entry["energy"]).reshape(-1)[0])

        # embedded Hessian per-atom
        if args.hessian and args.embed_hessian:
            H_blocks = entry.get("H_blocks", None)
            if not use_pad:
                # Default: build (N, 9*N) array (variable width per frame)
                H_embed = np.zeros((N, 9 * N), dtype=np.float64)
                if H_blocks is not None:
                    # H_blocks: (N,3,N,3) -> (N,N,9)
                    NN9 = H_blocks.transpose(0, 2, 1, 3).reshape(N, N, 9)
                    H_embed[:, : N * 9] = NN9.reshape(N, N * 9)
                atoms.new_array("hessian", H_embed)
            else:
                # Explicit padding: build (N, 9*n_max) array; zeros if missing
                H_embed = np.zeros((N, 9 * n_max), dtype=np.float64)
                if H_blocks is not None:
                    # H_blocks: (N,3,N,3) -> (N,N,9)
                    NN9 = H_blocks.transpose(0, 2, 1, 3).reshape(N, N, 9)
                    H_embed[:, : N * 9] = NN9.reshape(N, N * 9)
                atoms.new_array("hessian", H_embed)

        atoms_list.append(atoms)

    # Choose writer based on whether we embed properties
    if args.hessian and args.embed_hessian:
        # Ensure standard .extxyz suffix for EXTXYZ files
        output_path = args.output
        root, ext = os.path.splitext(output_path)
        if ext.lower() != ".extxyz":
            new_path = root + ".extxyz"
            print(f"[INFO] Writing EXTXYZ; changing output suffix to '.extxyz': {new_path} (was {output_path})")
            output_path = new_path
        write(output_path, atoms_list, format="extxyz")
    else:
        # legacy simple xyz (no per-atom hessian inline)
        write(args.output, atoms_list, format="xyz")

    # Optional: also save Hessians separately if requested and not embedded
    if args.hessian and not args.embed_hessian:
        out_path = args.hessian_out or (os.path.splitext(args.output)[0] + "_hessian.npz")
        flat_list = []
        shapes: List[Tuple[int, int]] = []
        for entry in samples:
            N = entry["N"]
            H_blocks = entry.get("H_blocks", None)
            if H_blocks is None:
                flat_list.append(None)
            else:
                flat_list.append(H_blocks.reshape(N * 3, N * 3).reshape(-1))
            shapes.append((N * 3, N * 3))
        np.savez(
            out_path,
            hessian_flat=np.array(flat_list, dtype=object),
            shapes=np.array(shapes, dtype=np.int64),
            num_atoms=np.array(Ns, dtype=np.int64),
        )


if __name__ == "__main__":
    main()
