using System;
using System.Linq;
using System.Runtime.InteropServices;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // ModelBuilder against newton.ModelBuilder: the same scene built in both, every model array compared.
    public class BuilderTests
    {
        const float Rtol = 1e-5f;

        // Relative tolerance on each value plus an absolute floor of 1e-6 of the array's largest value,
        // for entries that are rounding noise around zero in float32 (rotated inertia products).
        static void AssertNear<T>(string name, T[] expected, T[] actual, float rtol = Rtol) where T : unmanaged
        {
            var e = MemoryMarshal.Cast<T, float>(expected).ToArray();
            var a = MemoryMarshal.Cast<T, float>(actual).ToArray();
            Assert.AreEqual(e.Length, a.Length, $"{name}: length");
            var floor = 1e-6f * (e.Length == 0 ? 0f : e.Max(v => math.abs(v)));
            var worst = 0f;
            for (var i = 0; i < e.Length; i++)
            {
                var err = math.abs(e[i] - a[i]) / (rtol * math.max(math.abs(e[i]), math.abs(a[i])) + floor + 1e-30f);
                worst = math.max(worst, err);
                Assert.LessOrEqual(err, 1f, $"{name}[{i}]: Newton {e[i]:R}, ported {a[i]:R}");
            }
            UnityEngine.Debug.Log($"{name}: worst deviation {worst * rtol:E1} of tolerance {rtol:E0}");
        }

        static void AssertSame(string name, int[] expected, int[] actual) => CollectionAssert.AreEqual(expected, actual, name);

        static float[] InverseMass(float[] mass) => mass.Select(m => m != 0f ? 1f / m : 0f).ToArray();

        // Newton inverts Dm in float32 (LAPACK), which is off by up to a few 1e-5 on sliver tetrahedra; poseRtol covers that.
        static void AssertParticles(Fixture fx, Model model, float poseRtol = Rtol)
        {
            AssertNear("particle_q", fx.Array<float3>("particle_q"), model.ParticleQ);
            AssertNear("particle_qd", fx.Array<float3>("particle_qd"), model.ParticleQd);
            AssertNear("particle_mass", fx.Array<float>("particle_mass"), model.ParticleMass);
            AssertNear("particle_inv_mass", fx.Array<float>("particle_inv_mass"), InverseMass(model.ParticleMass));
            AssertNear("particle_radius", fx.Array<float>("particle_radius"), model.ParticleRadius);
            AssertSame("particle_flags", fx.Array<int>("particle_flags"), model.ParticleFlags);
            AssertSame("tet_indices", fx.Array<int>("tet_indices"), model.TetIndices.SelectMany(t => new[] { t.x, t.y, t.z, t.w }).ToArray());
            AssertNear("tet_poses", fx.Matrices("tet_poses"), model.TetPoses, poseRtol);
            AssertNear("tet_materials", fx.Array<float3>("tet_materials"), model.TetMaterials);
        }

        static quaternion Quat(Fixture fx, string name)
        {
            var q = fx.Array<float>(name);
            return new quaternion(q[0], q[1], q[2], q[3]);
        }

        [Test]
        public void SoftGridMatchesNewton()
        {
            var fx = new Fixture("builder_grid");
            var builder = new ModelBuilder { Gravity = fx.Array<float3>("gravity")[0] };
            builder.AddSoftGrid(new float3(0.5f, -1f, 2f), Quat(fx, "rot"), new float3(0.1f, 0f, -0.2f), new int3(5, 3, 2),
                new float3(0.1f, 0.2f, 0.15f), 800f, 7e4f, 2e4f, 30f, fixLeft: true, fixBottom: true, particleRadius: 0.02f);
            var model = builder.Finalize();
            AssertParticles(fx, model);
            AssertNear("gravity", fx.Array<float3>("gravity"), new[] { model.Gravity });
            Assert.AreEqual(fx.Value("soft_contact_ke"), model.SoftContactKe);
            Assert.AreEqual(fx.Value("soft_contact_kd"), model.SoftContactKd);
            Assert.AreEqual(fx.Value("soft_contact_mu"), model.SoftContactMu);
            Assert.AreEqual(72, model.ParticleCount);
            Assert.AreEqual(150, model.TetCount);
        }

        [Test]
        public void SoftMeshMatchesNewton()
        {
            var fx = new Fixture("builder_mesh");
            var builder = new ModelBuilder();
            UnityEngine.TestTools.LogAssert.Expect(UnityEngine.LogType.Warning, "inverted tetrahedral element");
            builder.AddSoftMesh(fx.Array<float3>("in_pos")[0], Quat(fx, "in_rot"), fx.Value("in_scale"),
                fx.Array<float3>("in_vel")[0], fx.Array<float3>("in_vertices"), fx.Array<int>("in_indices"),
                fx.Value("in_density"), fx.Array<float>("in_k_mu"), fx.Array<float>("in_k_lambda"),
                fx.Array<float>("in_k_damp"), 0.015f);
            var model = builder.Finalize();
            AssertParticles(fx, model, 1e-4f);
            Assert.AreEqual(359, model.TetCount, "the inverted element is dropped");
        }

        static ModelBuilder.ShapeConfig Config(float[] r) => new()
        {
            Density = r[12], Margin = r[13], Ke = r[14], Kd = r[15], Mu = r[16], IsSolid = r[17] != 0f,
            HasParticleCollision = r[18] != 0f, IsVisible = r[19] != 0f, Gap = r[20] < 0f ? null : r[20]
        };

        static Model BuildBodies(Fixture fx)
        {
            var builder = new ModelBuilder();
            var bodies = fx.Array<float>("in_bodies");
            for (var i = 0; i < fx.Shape("in_bodies")[0]; i++)
            {
                var r = bodies.Skip(i * 18).Take(18).ToArray();
                // Inertia as given by Newton's row-major mat33(Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz).
                var inertia = new float3x3(r[11], r[14], r[15], r[14], r[12], r[16], r[15], r[16], r[13]);
                builder.AddBody(new Transform(new float3(r[0], r[1], r[2]), new quaternion(r[3], r[4], r[5], r[6])),
                    new float3(r[7], r[8], r[9]), inertia, r[10], r[17] != 0f);
            }
            var shapes = fx.Array<float>("in_shapes");
            for (var i = 0; i < fx.Shape("in_shapes")[0]; i++)
            {
                var r = shapes.Skip(i * 21).Take(21).ToArray();
                var body = (int)r[0];
                var kind = (int)r[1];
                var xf = new Transform(new float3(r[2], r[3], r[4]), new quaternion(r[5], r[6], r[7], r[8]));
                var cfg = Config(r);
                switch (kind)
                {
                    case GeoType.Sphere: builder.AddShapeSphere(body, xf, r[9], cfg); break;
                    case GeoType.Capsule: builder.AddShapeCapsule(body, xf, r[9], r[10], cfg); break;
                    case GeoType.Box: builder.AddShapeBox(body, xf, r[9], r[10], r[11], cfg); break;
                    case GeoType.Plane: builder.AddShapePlane(null, xf, r[9], r[10], body, cfg); break;
                    case -1: builder.AddShapePlane(new float4(r[2], r[3], r[4], r[9]), null, r[10], r[11], body, cfg); break;
                    case -2: builder.AddGroundPlane(r[9], cfg); break;
                }
            }
            return builder.Finalize();
        }

        [Test]
        public void BodiesAndShapesMatchNewton()
        {
            var fx = new Fixture("builder_bodies");
            var model = BuildBodies(fx);
            AssertNear("body_q", fx.Array<Transform>("body_q"), model.BodyQ);
            AssertNear("body_com", fx.Array<float3>("body_com"), model.BodyCom);
            AssertNear("body_mass", fx.Array<float>("body_mass"), model.BodyMass);
            AssertNear("body_inv_mass", fx.Array<float>("body_inv_mass"), model.BodyInvMass);
            AssertNear("body_inertia", fx.Matrices("body_inertia"), model.BodyInertia);
            AssertNear("body_inv_inertia", fx.Matrices("body_inv_inertia"), model.BodyInvInertia);
            AssertNear("body_qd", fx.Array<float>("body_qd"), MemoryMarshal.Cast<SpatialVelocity, float>(model.BodyQd).ToArray());
            AssertSame("shape_body", fx.Array<int>("shape_body"), model.ShapeBody);
            AssertNear("shape_transform", fx.Array<Transform>("shape_transform"), model.ShapeTransform);
            AssertSame("shape_type", fx.Array<int>("shape_type"), model.ShapeType);
            AssertNear("shape_scale", fx.Array<float3>("shape_scale"), model.ShapeScale);
            AssertSame("shape_flags", fx.Array<int>("shape_flags"), model.ShapeFlags);
            AssertNear("shape_margin", fx.Array<float>("shape_margin"), model.ShapeMargin);
            AssertNear("shape_gap", fx.Array<float>("shape_gap"), model.ShapeGap);
            AssertNear("shape_material_ke", fx.Array<float>("shape_material_ke"), model.ShapeMaterialKe);
            AssertNear("shape_material_kd", fx.Array<float>("shape_material_kd"), model.ShapeMaterialKd);
            AssertNear("shape_material_mu", fx.Array<float>("shape_material_mu"), model.ShapeMaterialMu);
        }

        [Test]
        public void ZeroMassBodyStaysStatic()
        {
            var builder = new ModelBuilder();
            builder.AddBody();
            builder.AddShapeSphere(0, null, 0.5f, new ModelBuilder.ShapeConfig { Density = 0f });
            var model = builder.Finalize();
            Assert.AreEqual(0f, model.BodyMass[0]);
            Assert.AreEqual(0f, model.BodyInvMass[0]);
            Assert.IsFalse(Inertia.AnyNonZero(model.BodyInertia[0]));
            Assert.IsFalse(Inertia.AnyNonZero(model.BodyInvInertia[0]));
        }

        [Test]
        public void HangingGridBuiltByModelBuilderFollowsNewton()
        {
            var fx = new Fixture("hanging_grid");
            var builder = new ModelBuilder { Gravity = fx.Array<float3>("gravity")[0] };
            builder.AddSoftGrid(new float3(0f, 0f, 1f), quaternion.identity, float3.zero, new int3(6, 2, 2),
                new float3(fx.Value("cell")), fx.Value("density"), fx.Value("k_mu"), fx.Value("k_lambda"),
                fx.Value("k_damp"), fixLeft: true);
            builder.Color();
            var model = builder.Finalize();

            AssertNear("particle_q", fx.Array<float3>("particle_q"), model.ParticleQ);
            AssertNear("particle_mass", fx.Array<float>("particle_mass"), model.ParticleMass);
            AssertSame("color_particles", fx.Array<int>("color_particles"), model.ColorParticles);
            AssertSame("color_offsets", fx.Array<int>("color_offsets"), model.ColorOffsets);
            AssertNear("gravity", fx.Array<float3>("gravity"), new[] { model.Gravity });

            var steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var reference = fx.Array<float3>("steps_q");
            var n = model.ParticleCount;
            using var solver = new SolverVbd(model, (int)fx.Value("iterations"));
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var q = new float3[n];
            var worst = 0f;
            for (var s = 0; s < steps; s++)
            {
                cmd.Clear();
                solver.Step(cmd, state, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.ParticleQ.GetData(q);
                for (var i = 0; i < n; i++) worst = math.max(worst, math.length(q[i] - reference[s * n + i]));
            }
            UnityEngine.Debug.Log($"hanging_grid built by ModelBuilder: worst deviation {worst * 1e3f:F5} mm");
            Assert.Less(worst, 1e-4f);
        }
    }
}
