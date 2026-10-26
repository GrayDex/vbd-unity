using System;
using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Rigid AVBD with joints against Newton's SolverVBD step by step, on Newton's own model arrays and body coloring.
    public class RigidTests
    {
        const float PositionTolerance = 1e-4f;
        const double RotationToleranceDeg = 0.01;

        static Model ModelFrom(Fixture fx)
        {
            var joints = fx.Shape("joint_type")[0];
            return new Model
            {
                Gravity = fx.Array<float3>("gravity")[0],
                BodyQ = fx.Array<Transform>("body_q"),
                BodyQd = fx.Array<SpatialVelocity>("body_qd"),
                BodyCom = fx.Array<float3>("body_com"),
                BodyMass = fx.Array<float>("body_mass"),
                BodyInvMass = fx.Array<float>("body_inv_mass"),
                BodyInertia = fx.Matrices("body_inertia"),
                BodyInvInertia = fx.Matrices("body_inv_inertia"),
                BodyFlags = fx.Array<int>("body_flags"),
                BodyColorBodies = fx.Array<int>("body_color_bodies"),
                BodyColorOffsets = fx.Array<int>("body_color_offsets"),
                JointType = fx.Array<int>("joint_type"),
                JointEnabled = fx.Array<int>("joint_enabled").Select(e => e != 0).ToArray(),
                JointParent = fx.Array<int>("joint_parent"),
                JointChild = fx.Array<int>("joint_child"),
                JointXP = fx.Array<Transform>("joint_X_p"),
                JointXC = fx.Array<Transform>("joint_X_c"),
                JointQStart = fx.Array<int>("joint_q_start").Take(joints).ToArray(),
                JointQdStart = fx.Array<int>("joint_qd_start").Take(joints).ToArray(),
                JointDofDim = fx.Array<int2>("joint_dof_dim"),
                JointQ = fx.Array<float>("joint_q"),
                JointAxis = fx.Array<float3>("joint_axis"),
                JointTargetKe = fx.Array<float>("joint_target_ke"),
                JointTargetKd = fx.Array<float>("joint_target_kd"),
                JointLimitLower = fx.Array<float>("joint_limit_lower"),
                JointLimitUpper = fx.Array<float>("joint_limit_upper"),
                JointLimitKe = fx.Array<float>("joint_limit_ke"),
                JointLimitKd = fx.Array<float>("joint_limit_kd"),
                JointTargetQ = fx.Array<float>("joint_target_q"),
                JointTargetQd = fx.Array<float>("joint_target_qd")
            };
        }

        // Angle of the relative rotation, in degrees, in double precision.
        static double AngleDeg(quaternion a, quaternion b)
        {
            double4 p = a.value, r = b.value;
            // Vector and scalar part of conj(b) * a.
            var v = r.w * p.xyz - p.w * r.xyz - math.cross(r.xyz, p.xyz);
            var w = r.w * p.w + math.dot(r.xyz, p.xyz);
            return math.degrees(2.0 * math.atan2(math.length(v), math.abs(w)));
        }

        static void Run(string scene, bool replay = false, int steps = -1, bool kinematic = false)
        {
            var fx = new Fixture(scene);
            var model = ModelFrom(fx);
            if (steps < 0) steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var reference = fx.Array<Transform>("steps_body_q");
            // Kinematic scenes: body 0 is posed by the caller before each step.
            var prescribed = kinematic ? fx.Array<Transform>("prescribed_q") : null;
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
            float worstPos = 0f, early = 0f, travel = 0f;
            double worstRot = 0.0, earlyRot = 0.0;
            int worstPosStep = 0, worstRotStep = 0;
            for (var s = 0; s < steps; s++)
            {
                // Replay: the step recorded once runs every frame, as a caller replaying one command buffer would.
                if (!replay || s == 0)
                {
                    cmd.Clear();
                    if (prescribed != null) cmd.SetBufferData(state.BodyQ, new[] { prescribed[s] }, 0, 0, 1);
                    solver.Step(cmd, state, dt);
                }
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.BodyQ.GetData(q);
                for (var i = 0; i < n; i++)
                {
                    var r = reference[s * n + i];
                    var dp = math.length(q[i].P - r.P);
                    var dr = AngleDeg(q[i].Q, r.Q);
                    travel = math.max(travel, math.length(r.P - model.BodyQ[i].P));
                    if (s < 10)
                    {
                        early = math.max(early, dp);
                        earlyRot = math.max(earlyRot, dr);
                    }
                    if (dp > worstPos)
                    {
                        worstPos = dp;
                        worstPosStep = s;
                    }
                    if (dr > worstRot)
                    {
                        worstRot = dr;
                        worstRotStep = s;
                    }
                }
            }
            UnityEngine.Debug.Log($"{scene}{(replay ? " replayed" : "")}: {steps} steps, worst position {worstPos * 1e3f:F5} mm at step {worstPosStep}, " +
                                  $"worst rotation {worstRot:F5} deg at step {worstRotStep}; first 10 steps " +
                                  $"{early * 1e3f:F5} mm, {earlyRot:F5} deg; Newton travel {travel * 1e3f:F1} mm");
            Assert.Less(worstPos, PositionTolerance, $"{scene}: position");
            Assert.Less(worstRot, RotationToleranceDeg, $"{scene}: rotation");
        }

        [Test] public void RevolutePendulumFollowsNewton() => Run("revolute_pendulum");
        // Five links amplify float32 rounding: Newton's own CPU and CUDA runs part by more than 0.01° after step 78.
        [Test] public void BallChainFollowsNewton() => Run("ball_chain", steps: 60);
        [Test] public void FixedPairFollowsNewton() => Run("fixed_pair");
        [Test] public void PrismaticLimitFollowsNewton() => Run("prismatic_limit");
        [Test] public void D6LimitsFollowNewton() => Run("d6_limits");
        [Test] public void RevoluteDriveFollowsNewton() => Run("revolute_drive");
        [Test] public void RecordedStepReplays() => Run("revolute_drive", replay: true);
        [Test] public void KinematicChildFollowsNewton() => Run("kinematic_child", kinematic: true);
        [Test] public void FingerOnMovingPalmFollowsNewton() => Run("finger", kinematic: true);
        [Test] public void FreeSpinFollowsNewton() => Run("free_spin");

        [Test]
        public void AdjacencyExcludesFreeJointsAndWorld()
        {
            var model = ModelFrom(new Fixture("kinematic_child"));
            var (joints, offsets) = Adjacency.BodyJoints(model);
            CollectionAssert.AreEqual(new[] { 0, 1, 2 }, offsets);
            CollectionAssert.AreEqual(new[] { 1, 1 }, joints);
            Assert.That(model.JointType[0], Is.EqualTo(Model.JointFree), "joint 0 is the kinematic root's free joint");
            Assert.That(Array.IndexOf(model.BodyFlags, Model.BodyKinematic), Is.EqualTo(0));
        }
    }
}
