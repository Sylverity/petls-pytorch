# Topology inside a 160-molecule crystal

This example expands an experimental theobromine–2,4-dihydroxybenzoic-acid
co-crystal into a finite 4×5×1 supercell containing 160 molecules and 1,920
heavy atoms. PETLS-PyTorch analyzes the molecular-center scaffold while the
renderer keeps every molecule visible.

The calculation demonstrates assembly-scale questions PETLS-PyTorch can answer:

- when initially separate molecules become one connected scaffold (`β₀`);
- where packing-scale tunnels appear and disappear (`β₁`);
- where enclosed packing voids persist (`β₂`);
- how the least nonzero 1- and 2-Laplacian eigenvalues change with scale; and
- which molecular-center simplices most strongly support a selected harmonic
  2-cycle.

This is a finite-supercell analysis, not periodic homology. Radius-of-gyration
weights describe molecular size in the power-distance construction; they are
not energies, affinities, or force-field parameters.

## Render it

From the repository root:

```bash
MOLECULE_RENDERER_BIN=/path/to/renderer \
  uv run --extra analysis python examples/theobromine_crystal/render_demo.py
```

The script expects a compatible molecular-renderer executable and checks its
version before rendering. You can use `--renderer /path/to/renderer` instead
of setting `MOLECULE_RENDERER_BIN`.

For a quick smoke render:

```bash
uv run --extra analysis python examples/theobromine_crystal/render_demo.py \
  --renderer /path/to/renderer --frames 4 \
  --output /tmp/petls-crystal.gif --poster /tmp/petls-crystal.png
```

## Structure provenance

The bundled CIF is [Crystallography Open Database entry
7235246](https://www.crystallography.net/cod/7235246.html), an experimental
132 K X-ray structure reported by Gołdyn *et al.* in *CrystEngComm* (2019),
“Synthon hierarchy in theobromine cocrystals with hydroxybenzoic acids as
coformers.” COD data are dedicated to the public domain under CC0; the original
authors and COD are acknowledged here as requested.

The renderer reads the asymmetric unit and four published symmetry operations
directly from the CIF. It does not optimize, dock, or otherwise alter the
experimental intramolecular coordinates before replicating the unit cell.
