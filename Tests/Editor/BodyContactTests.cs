using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Two-way body-particle contacts against Newton: free bodies fall onto a soft block, a kinematic capsule presses in.
    public class BodyContactTests
    {
        static Model ModelFrom(Fixture fx)
        {
            var model = ContactTests.ModelFrom(fx);
            var bodies = RigidTests.ModelFrom(fx);
            foreach (var f in typeof(Model).GetFields().Where(f => !f.IsLiteral && (f.Name.StartsWith("Body") || f.Name.StartsWith("Joint"))))
                f.SetValue(model, f.GetValue(bodies));
            return model;
        }

        sealed class Run : System.IDisposable
        {
            public readonly Fixture Fx = new("soft_body_contact");
            public readonly Model Model;
            public readonly SolverVbd Solver;
            readonly CollisionPipeline _pipeline;
            readonly Contacts _contacts;
            public readonly State State;
            readonly CommandBuffer _cmd = new();
            public readonly int Steps;
            readonly int _kinematic;
            readonly float _dt;
            readonly Transform[] _prescribed;

            public Run()
            {
                Model = ModelFrom(Fx);
                Steps = (int)Fx.Value("steps");
                _dt = Fx.Value("dt");
                _kinematic = (int)Fx.Value("kinematic_body");
                _prescribed = Fx.Array<Transform>("prescribed_q");
                Solver = new SolverVbd(Model, (int)Fx.Value("iterations"))
                {
                    FrictionEpsilon = Fx.Value("friction_epsilon"),
                    RigidJointLinearKe = Fx.Value("rigid_joint_linear_ke"),
                    RigidJointAngularKe = Fx.Value("rigid_joint_angular_ke"),
                    RigidAvbdGamma = Fx.Value("rigid_avbd_gamma")
                };
                _pipeline = new CollisionPipeline(Model) { SoftContactGap = Fx.Value("soft_contact_gap") };
                _contacts = _pipeline.CreateContacts();
                State = new State(Model);
            }

            // Poses the kinematic capsule, collides and steps, as the Newton scene does.
            public void Step(int s)
            {
                _cmd.Clear();
                _cmd.SetBufferData(State.BodyQ, new[] { _prescribed[s] }, 0, _kinematic, 1);
                _pipeline.Collide(_cmd, State, _contacts);
                Solver.Step(_cmd, State, _contacts, _dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(_cmd);
            }

            public void Dispose()
            {
                Solver.Dispose();
                _pipeline.Dispose();
                _contacts.Dispose();
                State.Dispose();
                _cmd.Dispose();
            }
        }

        // Each step starts from Newton's particles and bodies of the step before: the error of one step.
        [Test]
        public void EveryStepFromNewtonStateMatches()
        {
            using var run = new Run();
            var refQ = run.Fx.Array<float3>("steps_q");
            var refQd = run.Fx.Array<float3>("steps_qd");
            var refBodyQ = run.Fx.Array<Transform>("steps_body_q");
            var refBodyQd = run.Fx.Array<SpatialVelocity>("steps_body_qd");
            int n = run.Model.ParticleCount, nb = run.Model.BodyCount;
            var q = new float3[n];
            var bq = new Transform[nb];
            float worst = 0f, worstBody = 0f;
            for (var s = 0; s < run.Steps; s++)
            {
                if (s > 0)
                {
                    run.State.ParticleQ.SetData(refQ, (s - 1) * n, 0, n);
                    run.State.ParticleQd.SetData(refQd, (s - 1) * n, 0, n);
                    run.State.BodyQ.SetData(refBodyQ, (s - 1) * nb, 0, nb);
                    run.State.BodyQd.SetData(refBodyQd, (s - 1) * nb, 0, nb);
                    run.Solver.BodyQPrev.SetData(refBodyQ, (s - 1) * nb, 0, nb);
                }
                run.Step(s);
                run.State.ParticleQ.GetData(q);
                run.State.BodyQ.GetData(bq);
                for (var i = 0; i < n; i++) worst = math.max(worst, math.length(q[i] - refQ[s * n + i]));
                for (var b = 0; b < nb; b++) worstBody = math.max(worstBody, math.length(bq[b].P - refBodyQ[s * nb + b].P));
            }
            UnityEngine.Debug.Log($"soft_body_contact one step: worst particle {worst * 1e3f:F5} mm, worst body {worstBody * 1e3f:F5} mm");
            Assert.Less(worst, 1e-5f);
            Assert.Less(worstBody, 1e-5f);
        }

        // A free run: contacts switch damping and friction branches, so the run drifts slowly; the bodies have to
        // land, bounce and settle with Newton's.
        [Test]
        public void FreeRunFollowsNewton()
        {
            using var run = new Run();
            var refQ = run.Fx.Array<float3>("steps_q");
            var refBodyQ = run.Fx.Array<Transform>("steps_body_q");
            int n = run.Model.ParticleCount, nb = run.Model.BodyCount;
            var q = new float3[n];
            var bq = new Transform[nb];
            float worstParticle = 0f, worstCenter = 0f, worstBody = 0f;
            var worstBodyStep = 0;
            for (var s = 0; s < run.Steps; s++)
            {
                run.Step(s);
                run.State.ParticleQ.GetData(q);
                run.State.BodyQ.GetData(bq);
                var center = float3.zero;
                for (var i = 0; i < n; i++)
                {
                    worstParticle = math.max(worstParticle, math.length(q[i] - refQ[s * n + i]));
                    center += q[i] - refQ[s * n + i];
                }
                worstCenter = math.max(worstCenter, math.length(center / n));
                for (var b = 0; b < nb; b++)
                {
                    var d = math.length(bq[b].P - refBodyQ[s * nb + b].P);
                    if (d <= worstBody) continue;
                    worstBody = d;
                    worstBodyStep = s;
                }
            }
            UnityEngine.Debug.Log($"soft_body_contact free run: worst particle {worstParticle * 1e3f:F4} mm, " +
                                  $"center of mass {worstCenter * 1e3f:F4} mm, worst body {worstBody * 1e3f:F4} mm at step {worstBodyStep}");
            Assert.Less(worstParticle, 1e-3f);
            Assert.Less(worstCenter, 1e-4f);
            Assert.Less(worstBody, 1e-4f);
        }
    }
}
