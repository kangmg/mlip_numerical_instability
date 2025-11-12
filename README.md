# Numerical Instability in MLIP: Precision Effects

## Overview
The notebook `numerical_instability.ipynb` describes an issue stemming from differences in the default precision settings across MLIP models. 

ASE performs numerical Hessian evaluations by applying slight displacements to atomic positions from the initial positions, resulting in very small energy differences.
When using low-precision data types (e.g., `float32`) to evaluate numerical gradients or Hessians, MLIPs can struggle with these subtle variations, leading to instability such as erratic trajectories or flatlined energy curves.

## Tests
I evaluated three MLIPs (`OrbMol`, `UMA-s-1.1`, and `MACE-OMOL`) under varying precision settings:
- **UMA-s-1.1**: Supports only `float32`
- **MACE-OMOL**: [`float64(default)`, `float32`]
- **OrbMol**: [`float64`, `float32-highest`, `float32-high`]

![final plot](./plot.png)

These tests highlight how precision mismatches exacerbate ASE's numerical approximation challenges.

## Key Observations
- Despite operating exclusively in `float32`, `UMA-s-1.1` generates smoother energy curves compared to `OrbMol` or `MACE-OMOL` under `float32` precision. 
- `OrbMol` and `MACE-OMOL` show horizontal/discrete plot with `float32` precisions.

## Notes
1. `amp` represents the displacement weight, calculated as `mode_vector * amp`
2. Attached animation shows results at `amp=0.5` (vibrational trajectory with perturbations)


<p align="center">
  <img src="./amp0.5.gif" alt="Example GIF" width="400">
</p>
