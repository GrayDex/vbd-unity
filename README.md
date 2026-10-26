# VBD Physics (GPU) for Unity

A compute-shader port of NVIDIA [Newton](https://github.com/newton-physics/newton)'s `SolverVBD`:
[Vertex Block Descent](https://arxiv.org/abs/2403.06321) for particles and tetrahedral FEM (Stable Neo-Hookean),
[Augmented VBD](https://graphics.cs.utah.edu/research/projects/avbd/) for rigid bodies and joints, two-way rigid–soft contact.

Work in progress: particles and tetrahedra step on the GPU; rigid bodies, joints and contacts are not ported yet.

## Use

```csharp
var model = new Vbd.Model { /* particles, tetrahedra, colors */ };
using var solver = new Vbd.SolverVbd(model, iterations: 10);
using var state = new Vbd.State(model);
var cmd = new CommandBuffer();
solver.Step(cmd, state, dt); // records the step; execute the buffer once per frame or replay it
Graphics.ExecuteCommandBuffer(cmd);
```

## Tests against Newton

`Tools~/newton/scene_*.py` runs a scene in Newton and writes its model and every step to `Tests/Fixtures~/*.vbdf`;
the tests step the same model and compare. Regenerate a fixture in an environment with Newton installed:

```
cd Tools~/newton && python scene_hanging_grid.py
```

## License

Apache-2.0, as Newton; see `LICENSE.md` and `NOTICE.md`.
