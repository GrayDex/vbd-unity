using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // State.BodyF and Control.JointF against Newton's SolverVBD step by step: forces change every step and are
    // written into the recorded step, as a caller driving the bodies would.
    public class ExternalForceTests
    {
        // As RigidTests (scene_external_forces.py --noise). Over 120 steps Newton's CPU and CUDA runs part by 0.006 mm
        // and 0.0023°, and its CUDA run moves by 0.034 mm and 0.013° when the forces grow by 1e-6: the free slide
        // and swing axes keep any rounding of the solve. The port lands within that response.
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
            var v = r.w * p.xyz - p.w * r.xyz - math.cross(r.xyz, p.xyz);
            var w = r.w * p.w + math.dot(r.xyz, p.xyz);
            return math.degrees(2.0 * math.atan2(math.length(v), math.abs(w)));
        }

        [Test]
        public void BodyAndJointForcesFollowNewton()
        {
            var fx = new Fixture("external_forces");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var dt = fx.Value("dt");
            var reference = fx.Array<Transform>("steps_body_q");
            var bodyF = fx.Array<SpatialVelocity>("applied_body_f");
            var jointF = fx.Array<float>("applied_joint_f");
            var n = model.BodyCount;
            var dofs = model.JointDofCount;

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
            float worstPos = 0f;
            double worstRot = 0.0;
            int worstPosStep = 0, worstRotStep = 0;
            var bodyPos = new float[n];
            var bodyRot = new double[n];
            float early = 0f;
            double earlyRot = 0.0;
            for (var s = 0; s < steps; s++)
            {
                cmd.Clear();
                cmd.SetBufferData(state.BodyF, bodyF, s * n, 0, n);
                cmd.SetBufferData(solver.Control.JointF, jointF, s * dofs, 0, dofs);
                solver.Step(cmd, state, dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                state.BodyQ.GetData(q);
                for (var i = 0; i < n; i++)
                {
                    var r = reference[s * n + i];
                    var dp = math.length(q[i].P - r.P);
                    var dr = AngleDeg(q[i].Q, r.Q);
                    if (dp > worstPos) (worstPos, worstPosStep) = (dp, s);
                    if (dr > worstRot) (worstRot, worstRotStep) = (dr, s);
                    bodyPos[i] = math.max(bodyPos[i], dp);
                    bodyRot[i] = math.max(bodyRot[i], dr);
                    if (s < 10) (early, earlyRot) = (math.max(early, dp), math.max(earlyRot, dr));
                }
            }
            UnityEngine.Debug.Log($"external_forces: {steps} steps, worst position {worstPos * 1e3f:F5} mm at step " +
                                  $"{worstPosStep}, worst rotation {worstRot:F5} deg at step {worstRotStep}; first 10 steps "+
                                  $"{early * 1e3f:F5} mm, {earlyRot:F5} deg; per body mm "+
                                  $"{string.Join(" ", bodyPos.Select(p => (p * 1e3f).ToString("F5")))}, deg "+
                                  $"{string.Join(" ", bodyRot.Select(r => r.ToString("F5")))}");
            Assert.Less(worstPos, PositionTolerance, "position");
            Assert.Less(worstRot, RotationToleranceDeg, "rotation");

            // Newton's step reads body_f and leaves it to the caller.
            var after = new SpatialVelocity[n];
            state.BodyF.GetData(after);
            for (var i = 0; i < n; i++)
                Assert.AreEqual(bodyF[(steps - 1) * n + i].Linear, after[i].Linear, $"body_f[{i}] kept");
        }
    }
}
