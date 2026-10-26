using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Body-body contacts against Newton's CollisionPipeline + SolverVBD: bodies land on the ground and on each other,
    // a kinematic box pushes a body (scene_body_contact.py: analytic pairs, scene_body_stack.py: GJK/MPR pairs).
    public class BodyBodyContactTests
    {
        static Model ModelFrom(Fixture fx)
        {
            var model = RigidTests.ModelFrom(fx);
            model.ShapeBody = fx.Array<int>("shape_body");
            model.ShapeTransform = fx.Array<Transform>("shape_transform");
            model.ShapeType = fx.Array<int>("shape_type");
            model.ShapeScale = fx.Array<float3>("shape_scale");
            model.ShapeFlags = fx.Array<int>("shape_flags");
            model.ShapeMargin = fx.Array<float>("shape_margin");
            model.ShapeGap = fx.Array<float>("shape_gap");
            model.ShapeCollisionGroup = fx.Array<int>("shape_collision_group");
            model.ShapeCollisionFilterPairs = fx.Array<int2>("shape_collision_filter_pairs");
            model.ShapeMaterialKe = fx.Array<float>("shape_material_ke");
            model.ShapeMaterialKd = fx.Array<float>("shape_material_kd");
            model.ShapeMaterialMu = fx.Array<float>("shape_material_mu");
            return model;
        }

        // Angle of the relative rotation, in degrees, in double precision (as RigidTests).
        static double AngleDeg(quaternion a, quaternion b)
        {
            double4 p = a.value, r = b.value;
            var v = r.w * p.xyz - p.w * r.xyz - math.cross(r.xyz, p.xyz);
            var w = r.w * p.w + math.dot(r.xyz, p.xyz);
            return math.degrees(2.0 * math.atan2(math.length(v), math.abs(w)));
        }

        sealed class Run : System.IDisposable
        {
            public readonly Fixture Fx;
            public readonly Model Model;
            public readonly SolverVbd Solver;
            readonly CollisionPipeline _pipeline;
            public readonly Contacts Contacts;
            public readonly State State;
            readonly CommandBuffer _cmd = new();
            public readonly int Steps;
            readonly int _kinematic;
            readonly float _dt;
            readonly Transform[] _prescribed;

            public Run(string scene)
            {
                Fx = new Fixture(scene);
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
                    RigidAvbdGamma = Fx.Value("rigid_avbd_gamma"),
                    RigidContactAlpha = Fx.Value("rigid_contact_alpha")
                };
                _pipeline = new CollisionPipeline(Model);
                Contacts = _pipeline.CreateContacts();
                State = new State(Model);
            }

            // Poses the kinematic box, collides and steps, as the Newton scene does.
            public void Step(int s)
            {
                _cmd.Clear();
                _cmd.SetBufferData(State.BodyQ, new[] { _prescribed[s] }, 0, _kinematic, 1);
                _pipeline.Collide(_cmd, State, Contacts);
                Solver.Step(_cmd, State, Contacts, _dt);
                UnityEngine.Graphics.ExecuteCommandBuffer(_cmd);
            }

            public void Dispose()
            {
                Solver.Dispose();
                _pipeline.Dispose();
                Contacts.Dispose();
                State.Dispose();
                _cmd.Dispose();
            }
        }

        // Each step starts from Newton's bodies of the step before: the error of one step. Steps whose contact count
        // differs are counted apart: there a contact sits on a float threshold and rounding decides it.
        static (float position, double rotation, int countSteps, int countOff) OneStep(string scene)
        {
            using var run = new Run(scene);
            var refQ = run.Fx.Array<Transform>("steps_body_q");
            var refQd = run.Fx.Array<SpatialVelocity>("steps_body_qd");
            var refCounts = run.Fx.Array<int>("steps_contact_count");
            var nb = run.Model.BodyCount;
            var q = new Transform[nb];
            var count = new int[1];
            float worst = 0f;
            double worstRot = 0.0;
            int worstStep = 0, worstRotStep = 0, worstRotBody = 0, countSteps = 0, countOff = 0;
            for (var s = 0; s < run.Steps; s++)
            {
                if (s > 0)
                {
                    run.State.BodyQ.SetData(refQ, (s - 1) * nb, 0, nb);
                    run.State.BodyQd.SetData(refQd, (s - 1) * nb, 0, nb);
                    run.Solver.BodyQPrev.SetData(refQ, (s - 1) * nb, 0, nb);
                }
                run.Step(s);
                run.State.BodyQ.GetData(q);
                run.Contacts.RigidContactCount.GetData(count);
                if (count[0] != refCounts[s])
                {
                    countSteps++;
                    countOff = math.max(countOff, math.abs(count[0] - refCounts[s]));
                    continue;
                }
                for (var b = 0; b < nb; b++)
                {
                    var rot = AngleDeg(q[b].Q, refQ[s * nb + b].Q);
                    if (rot > worstRot) { worstRot = rot; worstRotStep = s; worstRotBody = b; }
                    var d = math.length(q[b].P - refQ[s * nb + b].P);
                    if (d <= worst) continue;
                    worst = d;
                    worstStep = s;
                }
            }
            UnityEngine.Debug.Log($"{scene} one step: worst {worst * 1e3f:F5} mm at step {worstStep}, rotation " +
                                  $"{worstRot:F5} deg at step {worstRotStep} body {worstRotBody}; {countSteps} of " +
                                  $"{run.Steps} steps with another contact count, off by up to {countOff}");
            return (worst, worstRot, countSteps, countOff);
        }

        static (float position, double rotation) FreeRun(string scene)
        {
            using var run = new Run(scene);
            var refQ = run.Fx.Array<Transform>("steps_body_q");
            var nb = run.Model.BodyCount;
            var q = new Transform[nb];
            float worst = 0f;
            double worstRot = 0.0;
            int worstStep = 0, worstBody = 0;
            for (var s = 0; s < run.Steps; s++)
            {
                run.Step(s);
                run.State.BodyQ.GetData(q);
                for (var b = 0; b < nb; b++)
                {
                    worstRot = math.max(worstRot, AngleDeg(q[b].Q, refQ[s * nb + b].Q));
                    var d = math.length(q[b].P - refQ[s * nb + b].P);
                    if (d <= worst) continue;
                    worst = d;
                    worstStep = s;
                    worstBody = b;
                }
            }
            UnityEngine.Debug.Log($"{scene} free run: worst {worst * 1e3f:F4} mm at step {worstStep} body {worstBody}, " +
                                  $"rotation {worstRot:F4} deg; final z " +
                                  string.Join(" ", q.Select(t => t.P.z.ToString("F4"))));
            return (worst, worstRot);
        }

        // One step from Newton's state, steps with the same contact count: rounding of one contact search and solve,
        // the first ten steps of Newton's CPU against its CUDA run agree to 0.00000 mm; 1e-5 m, 1e-3 deg. A contact
        // on a float threshold (the manifold adds its deepest point when two faces differ by more than 0.1 degrees)
        // may flip with rounding, by one contact; that may not happen on more than 5% of the steps.
        [TestCase("body_contact")]
        [TestCase("body_stack")]
        public void EveryStepFromNewtonStateMatches(string scene)
        {
            var (position, rotation, countSteps, countOff) = OneStep(scene);
            Assert.LessOrEqual(countOff, 1);
            Assert.Less(countSteps, 15);
            Assert.Less(position, 1e-5f);
            Assert.Less(rotation, 1e-3);
        }

        // Free run over 0.5 s: contacts switch branches, so rounding grows with the horizon. Tolerance is four times
        // the noise floor of Newton's CPU against its CUDA run (noise_floor_body.py): body_contact 0.044 mm and
        // 0.076 deg, body_stack 0.165 mm and 0.021 deg (rounded up).
        [TestCase("body_contact", 0.2f, 0.3)]
        [TestCase("body_stack", 0.7f, 0.1)]
        public void FreeRunFollowsNewton(string scene, float positionMm, double rotationDeg)
        {
            var (position, rotation) = FreeRun(scene);
            Assert.Less(position, positionMm * 1e-3f);
            Assert.Less(rotation, rotationDeg);
        }
    }
}
