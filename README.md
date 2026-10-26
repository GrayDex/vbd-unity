# VBD Physics (GPU) for Unity

A compute-shader port of NVIDIA [Newton](https://github.com/newton-physics/newton)'s `SolverVBD`:
[Vertex Block Descent](https://arxiv.org/abs/2403.06321) for particles and tetrahedral FEM (Stable Neo-Hookean),
[Augmented VBD](https://graphics.cs.utah.edu/research/projects/avbd/) for rigid bodies and joints, two-way rigid–soft contact.

Work in progress. Ported:

* particles and tetrahedra (VBD, graph-colored Gauss–Seidel);
* particle–shape contacts with sphere, capsule, box and plane shapes: penalty, damping, friction;
* rigid bodies and joints (compliant-ALM AVBD): ball, fixed, revolute, prismatic, D6, free; drives and limits, kinematic bodies;
* `ModelBuilder`: particles, soft grids and meshes, bodies, shapes, graph coloring.

Not yet: the contact reaction on bodies (particles do not push bodies back; contacts are checked against Newton
with world shapes only), particle–body attachments, edge/face contacts,
joints in `ModelBuilder`, external body and joint forces.

## Use

```csharp
var builder = new Vbd.ModelBuilder(); // gravity -Y by default (Newton's is -Z)
builder.AddSoftGrid(pos, rot, vel, dim, cell, density, kMu, kLambda, kDamp);
builder.AddGroundPlane();
builder.Color();
var model = builder.Finalize();

using var solver = new Vbd.SolverVbd(model, iterations: 10);
using var pipeline = new Vbd.CollisionPipeline(model);
using var contacts = pipeline.CreateContacts();
using var state = new Vbd.State(model);
var cmd = new CommandBuffer();
pipeline.Collide(cmd, state, contacts);
solver.Step(cmd, state, contacts, dt); // records the step; execute the buffer once per frame or replay it
Graphics.ExecuteCommandBuffer(cmd);
```

Kinematic bodies (`Model.BodyKinematic` in `Model.BodyFlags`) are posed by writing `State.BodyQ` before the step;
joint drive targets go to `solver.Control`.

## Tests against Newton

`Tools~/newton/scene_*.py` runs a scene in Newton and writes its model and every step to `Tests/Fixtures~/*.vbdf`;
the tests step the same model and compare step by step. `noise_floor.py` measures how far Newton's own CPU and CUDA
runs drift apart, which sets the rigid tolerances. Regenerate a fixture in an environment with Newton installed:

```
cd Tools~/newton && python scene_hanging_grid.py
```

## License

Apache-2.0, as Newton; see `LICENSE.md` and `NOTICE.md`.
