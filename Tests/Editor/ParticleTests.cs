using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Particle VBD against Newton's SolverVBD step by step, on Newton's own model arrays and coloring.
    public class ParticleTests
    {
        static Model ModelFrom(Fixture fx)
        {
            var colorOffsets = fx.Array<int>("color_offsets");
            return new Model
            {
                Gravity = fx.Array<float3>("gravity")[0],
                ParticleQ = fx.Array<float3>("particle_q"),
                ParticleQd = fx.Array<float3>("particle_qd"),
                ParticleMass = fx.Array<float>("particle_mass"),
                ParticleFlags = fx.Array<int>("particle_flags"),
                TetIndices = fx.Array<int4>("tet_indices"),
                TetPoses = fx.Matrices("tet_poses"),
                TetMaterials = fx.Array<float3>("tet_materials"),
                ColorParticles = fx.Array<int>("color_particles"),
                ColorOffsets = colorOffsets
            };
        }

        [Test]
        public void AdjacencyMatchesNewton()
        {
            var fx = new Fixture("hanging_grid");
            var model = ModelFrom(fx);
            var (pairs, offsets) = Adjacency.VertexTets(model.TetIndices, model.ParticleCount);
            // Newton counts offsets in ints of the flat (tet, order) list: two per pair.
            CollectionAssert.AreEqual(fx.Array<int>("v_adj_tets_offsets"), offsets.Select(o => o * 2).ToArray());
            CollectionAssert.AreEqual(fx.Array<int>("v_adj_tets"), pairs.SelectMany(p => new[] { p.x, p.y }).ToArray());
        }

        [Test]
        public void HangingGridFollowsNewton()
        {
            var fx = new Fixture("hanging_grid");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var reference = fx.Array<float3>("steps_q");
            var n = model.ParticleCount;

            using var solver = new SolverVbd(model, (int)fx.Value("iterations"));
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var q = new float3[n];
            var worst = 0f;
            var worstStep = 0;
            for (var s = 0; s < steps; s++)
            {
                cmd.Clear();
                solver.Step(cmd, state, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.ParticleQ.GetData(q);
                for (var i = 0; i < n; i++)
                {
                    var d = math.length(q[i] - reference[s * n + i]);
                    if (d <= worst) continue;
                    worst = d;
                    worstStep = s;
                }
            }
            var sag = model.ParticleQ.Min(p => p.z) - reference.Skip((steps - 1) * n).Min(p => p.z);
            UnityEngine.Debug.Log($"hanging_grid: worst deviation {worst * 1e3f:F4} mm at step {worstStep}, Newton sag {sag * 1e3f:F1} mm");
            Assert.Less(worst, 1e-4f);
        }
    }
}
