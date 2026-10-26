using System;
using System.Linq;
using System.Runtime.InteropServices;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // ModelBuilder joints against newton.ModelBuilder: the scene of Tools~/newton/scene_builder_joints.py built here,
    // every finalized array compared, then the built model stepped against Newton's run of its own model.
    public class JointBuilderTests
    {
        const float Hz = 0.15f;
        const float Rtol = 1e-5f;
        // As RigidTests. Newton's CPU and CUDA runs of this scene part by 0.014 mm and 0.0033° over its 120 steps
        // (scene_builder_joints.py --noise).
        const float PositionTolerance = 1e-4f;
        const double RotationToleranceDeg = 0.01;
        const int Steps = 120;

        static quaternion Rot(float3 axis, float angle) => quaternion.AxisAngle(math.normalize(axis), angle);
        static Transform At(float x, float y, float z, quaternion? q = null) => new(new float3(x, y, z), q ?? quaternion.identity);

        // rigid_scene.box_inertia: solid box about its center, computed in double as Python does.
        static float3x3 BoxInertia(double m, double hx, double hy, double hz) =>
            float3x3.Scale((float)(m / 3.0 * (hy * hy + hz * hz)), (float)(m / 3.0 * (hx * hx + hz * hz)),
                (float)(m / 3.0 * (hx * hx + hy * hy)));

        static ModelBuilder Build(float3 gravity)
        {
            var b = new ModelBuilder { Gravity = gravity };
            int Link(float3 c, float mass, quaternion? q = null, bool kinematic = false, float hx = 0.05f, float hy = 0.05f,
                float hz = Hz) =>
                b.AddLink(new Transform(c, q ?? quaternion.identity), null, BoxInertia(mass, hx, hy, hz), mass,
                    isKinematic: kinematic);
            ModelBuilder.JointDofConfig Cfg(float3 axis, float limitLower = -Model.MaxVal, float limitUpper = Model.MaxVal,
                float limitKe = 1e4f, float limitKd = 1e1f, float targetPos = 0f, float targetKe = 0f, float targetKd = 0f) =>
                new(axis, limitLower, limitUpper, limitKe, limitKd, targetPos, 0f, targetKe, targetKd);
            var x = new float3(1f, 0f, 0f);
            var y = new float3(0f, 1f, 0f);
            var z = new float3(0f, 0f, 1f);

            var b0 = Link(new float3(0f, 0f, 2f - Hz), 1f);
            var b1 = Link(new float3(0f, 0f, 2f - 3 * Hz), 0.8f);
            var b2 = Link(new float3(0f, 0f, 2f - 5 * Hz), 0.6f, Rot(z, 0.4f), hx: 0.12f, hy: 0.12f);
            var b3 = Link(new float3(0f, 0f, 2f - 7 * Hz), 0.5f, hx: 0.12f, hy: 0.12f);
            var b4 = Link(new float3(0f, 0f, 2f - 9 * Hz), 0.4f, hx: 0.12f, hy: 0.12f);
            var b5 = Link(new float3(0.3f, 0.1f, 2f - 9 * Hz), 0.3f, Rot(new float3(1f, 1f, 0f), 0.5f));
            b.AddArticulation(new[]
            {
                b.AddJointRevolute(-1, b0, At(0f, 0f, 2f), At(0f, 0f, Hz), x, targetPos: 0.3f, targetKe: 50f,
                    targetKd: 2f, limitLower: -1f, limitUpper: 1f),
                b.AddJointPrismatic(b0, b1, At(0f, 0f, -Hz), At(0f, 0f, Hz), z, limitLower: -0.05f, limitUpper: 0.05f,
                    limitKe: 2e4f, limitKd: 5f),
                b.AddJointFixed(b1, b2, At(0f, 0f, -Hz), At(0f, 0f, Hz, Rot(z, -0.4f))),
                b.AddJointD6(b2, b3, new[] { Cfg(x, -0.02f, 0.02f) },
                    new[] { Cfg(y, targetPos: 0.2f, targetKe: 20f, targetKd: 1f), Cfg(x, -0.5f, 0.5f, 1e3f, 2f) },
                    At(0f, 0f, -Hz, Rot(z, -0.4f)), At(0f, 0f, Hz)),
                b.AddJointBall(b3, b4, At(0f, 0f, -Hz), At(0f, 0f, Hz)),
                b.AddJointFree(b5, At(0.1f, 0f, 0f), parent: b4)
            });

            var b6 = b.AddBody(At(1f, 0f, 1f, Rot(y, 0.3f)), new float3(0.02f, 0f, 0.01f), BoxInertia(2.0, 0.1, 0.1, 0.1), 2f);
            b.AddShapeSphere(b6, At(0.05f, 0f, 0f), 0.1f, new ModelBuilder.ShapeConfig { Density = 300f });

            var b7 = Link(new float3(-1f, 0f, 1.5f), 1f, kinematic: true, hz: 0.05f);
            var b8 = Link(new float3(-1f, 0f, 1.5f - 0.05f - Hz), 0.5f);
            var b9 = Link(new float3(-1f, 0f, 1.5f - 0.05f - 3 * Hz), 0.4f);
            b.AddArticulation(new[]
            {
                b.AddJointFree(b7),
                b.AddJointD6(b7, b8, null,
                    new[]
                    {
                        Cfg(x, -0.6f, 0.6f), Cfg(y, targetPos: 0.1f, targetKe: 30f, targetKd: 1f),
                        Cfg(z, targetKe: 5f, targetKd: 0.5f)
                    },
                    At(0f, 0f, -0.05f), At(0f, 0f, Hz)),
                b.AddJointRevolute(b8, b9, Cfg(new float3(0f, 2f, 0f), -1f, 0.8f, targetPos: 1.5f, targetKe: 10f),
                    At(0f, 0f, -Hz), At(0f, 0f, Hz))
            });

            var b10 = Link(new float3(0f, 1f, 1f), 0.7f);
            b.AddJointPrismatic(-1, b10, At(0f, 1f, 1f), axis: z, targetPos: 0.1f, targetVel: 0.2f, targetKe: 100f,
                targetKd: 5f);
            var b11 = Link(new float3(0f, -1f, 1f - Hz), 0.6f);
            b.AddJointBall(-1, b11, At(0f, -1f, 1f), At(0f, 0f, Hz), enabled: false);
            b.Color();
            return b;
        }

        // Relative tolerance per value with an absolute floor of 1e-6 for values that are rounding noise around zero.
        static void AssertNear<T>(string name, T[] expected, T[] actual) where T : unmanaged
        {
            var e = MemoryMarshal.Cast<T, float>(expected).ToArray();
            var a = MemoryMarshal.Cast<T, float>(actual).ToArray();
            Assert.AreEqual(e.Length, a.Length, $"{name}: length");
            for (var i = 0; i < e.Length; i++)
                Assert.LessOrEqual(math.abs(e[i] - a[i]), Rtol * math.max(math.abs(e[i]), math.abs(a[i])) + 1e-6f,
                    $"{name}[{i}]: Newton {e[i]:R}, ported {a[i]:R}");
        }

        static void AssertSame(string name, int[] expected, int[] actual) => CollectionAssert.AreEqual(expected, actual, name);

        [Test]
        public void JointedModelMatchesNewton()
        {
            var fx = new Fixture("builder_joints");
            var model = Build(fx.Array<float3>("gravity")[0]).Finalize();
            var joints = fx.Shape("joint_type")[0];

            AssertNear("body_q", fx.Array<Transform>("body_q"), model.BodyQ);
            AssertNear("body_qd", fx.Array<SpatialVelocity>("body_qd"), model.BodyQd);
            AssertNear("body_com", fx.Array<float3>("body_com"), model.BodyCom);
            AssertNear("body_mass", fx.Array<float>("body_mass"), model.BodyMass);
            AssertNear("body_inv_mass", fx.Array<float>("body_inv_mass"), model.BodyInvMass);
            AssertNear("body_inertia", fx.Matrices("body_inertia"), model.BodyInertia);
            AssertNear("body_inv_inertia", fx.Matrices("body_inv_inertia"), model.BodyInvInertia);
            AssertSame("body_flags", fx.Array<int>("body_flags"), model.BodyFlags);
            AssertSame("body_color_bodies", fx.Array<int>("body_color_bodies"), model.BodyColorBodies);
            AssertSame("body_color_offsets", fx.Array<int>("body_color_offsets"), model.BodyColorOffsets);

            AssertSame("joint_type", fx.Array<int>("joint_type"), model.JointType);
            AssertSame("joint_enabled", fx.Array<int>("joint_enabled"), model.JointEnabled.Select(e => e ? 1 : 0).ToArray());
            AssertSame("joint_parent", fx.Array<int>("joint_parent"), model.JointParent);
            AssertSame("joint_child", fx.Array<int>("joint_child"), model.JointChild);
            AssertSame("joint_q_start", fx.Array<int>("joint_q_start").Take(joints).ToArray(), model.JointQStart);
            AssertSame("joint_qd_start", fx.Array<int>("joint_qd_start").Take(joints).ToArray(), model.JointQdStart);
            Assert.AreEqual(model.JointQ.Length, fx.Array<int>("joint_q_start")[joints], "joint coordinate count");
            Assert.AreEqual(model.JointDofCount, fx.Array<int>("joint_qd_start")[joints], "joint DOF count");
            AssertSame("joint_dof_dim", fx.Array<int>("joint_dof_dim"), MemoryMarshal.Cast<int2, int>(model.JointDofDim).ToArray());
            AssertNear("joint_X_p", fx.Array<Transform>("joint_X_p"), model.JointXP);
            AssertNear("joint_X_c", fx.Array<Transform>("joint_X_c"), model.JointXC);
            AssertNear("joint_axis", fx.Array<float3>("joint_axis"), model.JointAxis);
            AssertNear("joint_q", fx.Array<float>("joint_q"), model.JointQ);
            AssertNear("joint_qd", fx.Array<float>("joint_qd"), model.JointQd);
            AssertNear("joint_f", fx.Array<float>("joint_f"), model.JointF);
            AssertNear("joint_target_ke", fx.Array<float>("joint_target_ke"), model.JointTargetKe);
            AssertNear("joint_target_kd", fx.Array<float>("joint_target_kd"), model.JointTargetKd);
            AssertNear("joint_limit_lower", fx.Array<float>("joint_limit_lower"), model.JointLimitLower);
            AssertNear("joint_limit_upper", fx.Array<float>("joint_limit_upper"), model.JointLimitUpper);
            AssertNear("joint_limit_ke", fx.Array<float>("joint_limit_ke"), model.JointLimitKe);
            AssertNear("joint_limit_kd", fx.Array<float>("joint_limit_kd"), model.JointLimitKd);
            AssertNear("joint_target_q", fx.Array<float>("joint_target_q"), model.JointTargetQ);
            AssertNear("joint_target_qd", fx.Array<float>("joint_target_qd"), model.JointTargetQd);
        }

        // Angle of the relative rotation, in degrees, in double precision.
        static double AngleDeg(quaternion a, quaternion b)
        {
            double4 p = a.value, r = b.value;
            var v = r.w * p.xyz - p.w * r.xyz - math.cross(r.xyz, p.xyz);
            var w = r.w * p.w + math.dot(r.xyz, p.xyz);
            return math.degrees(2.0 * math.atan2(math.length(v), math.abs(w)));
        }

        [Test]
        public void BuiltModelFollowsNewton()
        {
            var fx = new Fixture("builder_joints");
            var model = Build(fx.Array<float3>("gravity")[0]).Finalize();
            var dt = fx.Value("dt");
            var reference = fx.Array<Transform>("steps_body_q");
            var n = model.BodyCount;
            using var solver = new SolverVbd(model, (int)fx.Value("iterations"));
            solver.RigidJointLinearKe = fx.Value("rigid_joint_linear_ke");
            solver.RigidJointAngularKe = fx.Value("rigid_joint_angular_ke");
            solver.RigidJointLinearKd = fx.Value("rigid_joint_linear_kd");
            solver.RigidJointAngularKd = fx.Value("rigid_joint_angular_kd");
            solver.RigidAvbdGamma = fx.Value("rigid_avbd_gamma");
            solver.RigidJointAlpha = fx.Value("rigid_joint_alpha");
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var q = new Transform[n];
            var bodyPos = new float[n];
            var bodyRot = new double[n];
            for (var s = 0; s < Steps; s++)
            {
                cmd.Clear();
                solver.Step(cmd, state, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.BodyQ.GetData(q);
                for (var i = 0; i < n; i++)
                {
                    var r = reference[s * n + i];
                    bodyPos[i] = math.max(bodyPos[i], math.length(q[i].P - r.P));
                    bodyRot[i] = math.max(bodyRot[i], AngleDeg(q[i].Q, r.Q));
                }
            }
            UnityEngine.Debug.Log($"builder_joints: {Steps} steps, worst position {bodyPos.Max() * 1e3f:F5} mm, worst " +
                                  $"rotation {bodyRot.Max():F5} deg; per body mm " +
                                  $"{string.Join(" ", bodyPos.Select(p => (p * 1e3f).ToString("F5")))}, deg " +
                                  $"{string.Join(" ", bodyRot.Select(r => r.ToString("F5")))}");
            Assert.Less(bodyPos.Max(), PositionTolerance, "position");
            Assert.Less(bodyRot.Max(), RotationToleranceDeg, "rotation");
        }

        [Test]
        public void ArticulationRulesAsNewton()
        {
            var b = new ModelBuilder();
            var a = b.AddLink();
            var c = b.AddLink();
            var d = b.AddLink(isKinematic: true);
            var j0 = b.AddJointRevolute(-1, a);
            var j1 = b.AddJointRevolute(a, c);
            var j2 = b.AddJointBall(c, d);
            Assert.Throws<ArgumentException>(() => b.AddArticulation(new[] { j1, j0 }), "not ascending");
            Assert.Throws<ArgumentException>(() => b.AddArticulation(new[] { j0, j2 }), "not contiguous");
            Assert.Throws<ArgumentException>(() => b.AddArticulation(new[] { j0, j1, j2 }), "kinematic body on a body");
            b.AddArticulation(new[] { j0, j1 });
            Assert.Throws<ArgumentException>(() => b.AddArticulation(new[] { j1 }), "joint in two articulations");
            Assert.Throws<InvalidOperationException>(() => b.Finalize(), "joint outside any articulation");
            Assert.Throws<ArgumentException>(() => b.AddJoint(5, -1, a), "DISTANCE is not solved by SolverVbd");
        }
    }
}
