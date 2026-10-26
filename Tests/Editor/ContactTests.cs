using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Particle-shape contacts against Newton's CollisionPipeline + SolverVBD.
    public class ContactTests
    {
        internal static Model ModelFrom(Fixture fx) => new()
        {
            Gravity = fx.Array<float3>("gravity")[0],
            ParticleQ = fx.Array<float3>("particle_q"),
            ParticleQd = fx.Array<float3>("particle_qd"),
            ParticleMass = fx.Array<float>("particle_mass"),
            ParticleFlags = fx.Array<int>("particle_flags"),
            ParticleRadius = fx.Array<float>("particle_radius"),
            TetIndices = fx.Array<int4>("tet_indices"),
            TetPoses = fx.Matrices("tet_poses"),
            TetMaterials = fx.Array<float3>("tet_materials"),
            ColorParticles = fx.Array<int>("color_particles"),
            ColorOffsets = fx.Array<int>("color_offsets"),
            ShapeBody = fx.Array<int>("shape_body"),
            ShapeTransform = fx.Array<Transform>("shape_transform"),
            ShapeType = fx.Array<int>("shape_type"),
            ShapeScale = fx.Array<float3>("shape_scale"),
            ShapeFlags = fx.Array<int>("shape_flags"),
            ShapeMargin = fx.Array<float>("shape_margin"),
            ShapeMaterialKe = fx.Array<float>("shape_material_ke"),
            ShapeMaterialKd = fx.Array<float>("shape_material_kd"),
            ShapeMaterialMu = fx.Array<float>("shape_material_mu"),
            SoftContactKe = fx.Value("soft_contact_ke"),
            SoftContactKd = fx.Value("soft_contact_kd"),
            SoftContactMu = fx.Value("soft_contact_mu")
        };

        // A free run: contacts switch damping and friction branches, so single particles drift apart slowly;
        // the sheet as a whole (center of mass) has to stay with Newton's.
        [Test]
        public void FreeRunStaysWithNewton()
        {
            var fx = new Fixture("soft_contact_static");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var refQ = fx.Array<float3>("steps_q");
            var n = model.ParticleCount;

            using var solver = new SolverVbd(model, (int)fx.Value("iterations"));
            solver.FrictionEpsilon = fx.Value("friction_epsilon");
            using var pipeline = new CollisionPipeline(model);
            pipeline.SoftContactGap = fx.Value("soft_contact_gap");
            using var contacts = pipeline.CreateContacts();
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var q = new float3[n];
            float worstParticle = 0f, worstCenter = 0f;
            for (var s = 0; s < steps; s++)
            {
                cmd.Clear();
                pipeline.Collide(cmd, state, contacts);
                solver.Step(cmd, state, contacts, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.ParticleQ.GetData(q);
                var center = float3.zero;
                for (var i = 0; i < n; i++)
                {
                    worstParticle = math.max(worstParticle, math.length(q[i] - refQ[s * n + i]));
                    center += q[i] - refQ[s * n + i];
                }
                worstCenter = math.max(worstCenter, math.length(center / n));
            }
            UnityEngine.Debug.Log($"soft_contact_static free run: worst particle {worstParticle * 1e3f:F4} mm, worst center of mass {worstCenter * 1e3f:F4} mm");
            Assert.Less(worstParticle, 1e-3f);
            Assert.Less(worstCenter, 1e-4f);
        }

        // Contact records from Newton's state at the start of every step, matched by (particle, shape).
        [Test]
        public void CollideMatchesNewton()
        {
            var fx = new Fixture("soft_contact_static");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var refQ = fx.Array<float3>("steps_q");
            var refParticle = fx.Array<int>("steps_contact_particle");
            var refShape = fx.Array<int>("steps_contact_shape");
            var refPos = fx.Array<float3>("steps_contact_body_pos");
            var refNormal = fx.Array<float3>("steps_contact_normal");
            var records = fx.Shape("steps_contact_particle")[1];
            var n = model.ParticleCount;

            using var pipeline = new CollisionPipeline(model);
            pipeline.SoftContactGap = fx.Value("soft_contact_gap");
            using var contacts = pipeline.CreateContacts();
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var count = new int[1];
            var particle = new int[contacts.SoftContactMax];
            var shape = new int[contacts.SoftContactMax];
            var pos = new float3[contacts.SoftContactMax];
            var normal = new float3[contacts.SoftContactMax];
            int missing = 0, extra = 0;
            float worstPos = 0f, worstNormal = 0f;
            var report = new System.Text.StringBuilder();
            for (var s = 0; s < steps; s++)
            {
                if (s > 0) state.ParticleQ.SetData(refQ, (s - 1) * n, 0, n);
                cmd.Clear();
                pipeline.Collide(cmd, state, contacts);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                contacts.SoftContactCount.GetData(count);
                contacts.SoftContactParticle.GetData(particle);
                contacts.SoftContactShape.GetData(shape);
                contacts.SoftContactBodyPos.GetData(pos);
                contacts.SoftContactNormal.GetData(normal);
                var matched = new bool[count[0]];
                for (var r = 0; r < records && refParticle[s * records + r] >= 0; r++)
                {
                    var j = System.Array.FindIndex(particle, 0, count[0],
                        p => p == refParticle[s * records + r]);
                    while (j >= 0 && shape[j] != refShape[s * records + r])
                        j = j + 1 < count[0] ? System.Array.FindIndex(particle, j + 1, count[0] - j - 1, p => p == refParticle[s * records + r]) : -1;
                    if (j < 0)
                    {
                        if (missing++ < 5) report.AppendLine($"step {s}: missing particle {refParticle[s * records + r]} shape {refShape[s * records + r]}");
                        continue;
                    }
                    matched[j] = true;
                    worstPos = math.max(worstPos, math.length(pos[j] - refPos[s * records + r]));
                    var dn = math.length(normal[j] - refNormal[s * records + r]);
                    if (dn > 1e-3f && worstNormal <= 1e-3f)
                        report.AppendLine($"step {s}: particle {particle[j]} shape {shape[j]} normal {normal[j]} vs {refNormal[s * records + r]}, pos {pos[j]} vs {refPos[s * records + r]}");
                    worstNormal = math.max(worstNormal, dn);
                }
                for (var j = 0; j < count[0]; j++)
                    if (!matched[j] && extra++ < 5) report.AppendLine($"step {s}: extra particle {particle[j]} shape {shape[j]}");
            }
            UnityEngine.Debug.Log($"collide: missing {missing}, extra {extra}, worst body pos {worstPos * 1e3f:F5} mm, worst normal {worstNormal:F6}\n{report}");
            Assert.AreEqual(0, missing + extra);
            Assert.Less(worstPos, 1e-6f);
        }

        // Each step starts from Newton's state of the step before: the error of one step, free of the chaos
        // a free run has once contacts switch damping and friction on and off.
        [Test]
        public void EveryStepFromNewtonStateMatches()
        {
            var fx = new Fixture("soft_contact_static");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var refQ = fx.Array<float3>("steps_q");
            var refQd = fx.Array<float3>("steps_qd");
            var refCounts = fx.Array<int>("steps_contact_count");
            var n = model.ParticleCount;

            using var solver = new SolverVbd(model, (int)fx.Value("iterations"));
            solver.FrictionEpsilon = fx.Value("friction_epsilon");
            using var pipeline = new CollisionPipeline(model);
            pipeline.SoftContactGap = fx.Value("soft_contact_gap");
            using var contacts = pipeline.CreateContacts();
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var q = new float3[n];
            var count = new int[1];
            float worst = 0f, worstFree = 0f;
            int worstStep = 0, countOff = 0;
            for (var s = 0; s < steps; s++)
            {
                if (s > 0)
                {
                    state.ParticleQ.SetData(refQ, (s - 1) * n, 0, n);
                    state.ParticleQd.SetData(refQd, (s - 1) * n, 0, n);
                }
                cmd.Clear();
                pipeline.Collide(cmd, state, contacts);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                contacts.SoftContactCount.GetData(count);
                countOff = math.max(countOff, math.abs(count[0] - refCounts[s]));
                cmd.Clear();
                solver.Step(cmd, state, contacts, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.ParticleQ.GetData(q);
                for (var i = 0; i < n; i++)
                {
                    var d = math.length(q[i] - refQ[s * n + i]);
                    if (refCounts[s] == 0) worstFree = math.max(worstFree, d);
                    if (d <= worst) continue;
                    worst = d;
                    worstStep = s;
                }
            }
            UnityEngine.Debug.Log($"soft_contact_static one step: worst {worst * 1e3f:F5} mm at step {worstStep} " +
                                  $"(contacts {refCounts[worstStep]}), without contacts {worstFree * 1e3f:F5} mm, contact count off by {countOff}");
            Assert.LessOrEqual(countOff, 1);
            Assert.Less(worst, 1e-5f);
        }
    }
}
