using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace Vbd.Tests.Editor
{
    // Shape-shape contacts against Newton's CollisionPipeline on scripted body poses, every shape pair type.
    public class BodyCollideTests
    {
        const int Fields = 19;

        static Model ModelFrom(Fixture fx) => new()
        {
            BodyQ = fx.Array<Transform>("body_q"),
            ShapeBody = fx.Array<int>("shape_body"),
            ShapeTransform = fx.Array<Transform>("shape_transform"),
            ShapeType = fx.Array<int>("shape_type"),
            ShapeScale = fx.Array<float3>("shape_scale"),
            ShapeFlags = fx.Array<int>("shape_flags"),
            ShapeMargin = fx.Array<float>("shape_margin"),
            ShapeGap = fx.Array<float>("shape_gap"),
            ShapeCollisionGroup = fx.Array<int>("shape_collision_group"),
            ShapeCollisionFilterPairs = fx.Array<int2>("shape_collision_filter_pairs")
        };

        static HashSet<int2> PairSet(IEnumerable<int2> pairs) =>
            new(pairs.Select(p => new int2(math.min(p.x, p.y), math.max(p.x, p.y))));

        // The explicit pair list from Newton's model arrays, and the filters our ModelBuilder derives from the
        // scene's joints, groups and flags, against Newton's.
        [Test]
        public void FilteringMatchesNewton()
        {
            var fx = new Fixture("body_collide");
            var newtonPairs = PairSet(fx.Array<int2>("shape_contact_pairs"));
            var pairs = PairSet(CollisionPipeline.ShapeContactPairs(ModelFrom(fx)));
            Assert.IsTrue(newtonPairs.SetEquals(pairs), $"pairs: {pairs.Count} vs Newton {newtonPairs.Count}");

            var built = Build(fx);
            var newtonFilters = PairSet(fx.Array<int2>("shape_collision_filter_pairs"));
            var filters = PairSet(built.ShapeCollisionFilterPairs);
            Assert.IsTrue(newtonFilters.SetEquals(filters),
                $"filters {string.Join(" ", filters)} vs Newton {string.Join(" ", newtonFilters)}");
            CollectionAssert.AreEqual(fx.Array<int>("shape_collision_group"), built.ShapeCollisionGroup);
            CollectionAssert.AreEqual(fx.Array<int>("shape_flags"), built.ShapeFlags);
            Assert.IsTrue(newtonPairs.SetEquals(PairSet(CollisionPipeline.ShapeContactPairs(built))));
        }

        // The scene of scene_body_collide.py through our ModelBuilder.
        static Model Build(Fixture fx)
        {
            var b = new ModelBuilder { Gravity = new float3(0f, 0f, -9.81f) };
            foreach (var q in fx.Array<Transform>("body_q")) b.AddLink(q, mass: 1f);
            var rows = fx.Array<float>("shape_rows");
            for (var r = 0; r < rows.Length / 12; r++)
            {
                var s = rows.Skip(r * 12).Take(12).ToArray();
                var cfg = new ModelBuilder.ShapeConfig
                {
                    Margin = s[5], Gap = s[6] < 0f ? null : s[6], CollisionGroup = (int)s[7],
                    HasShapeCollision = s[8] != 0f, Density = 0f
                };
                var body = (int)s[0];
                var pose = new Transform(new float3(s[9], s[10], s[11]), quaternion.identity);
                switch ((int)s[1])
                {
                    case GeoType.Plane: b.AddGroundPlane(cfg: cfg); break;
                    case GeoType.Sphere: b.AddShapeSphere(body, pose, s[2], cfg); break;
                    case GeoType.Capsule: b.AddShapeCapsule(body, pose, s[2], s[3], cfg); break;
                    default: b.AddShapeBox(body, pose, s[2], s[3], s[4], cfg); break;
                }
            }
            var filterRows = fx.Array<int2>("filter_rows");
            foreach (var f in filterRows) b.AddShapeCollisionFilterPair(f.x, f.y);
            var joints = fx.Array<int4>("joint_rows");
            var articulations = new SortedDictionary<int, List<int>>();
            foreach (var j in joints)
            {
                var index = j.x switch
                {
                    Model.JointRevolute => b.AddJointRevolute(j.y, j.z),
                    Model.JointFree => b.AddJointFree(j.z, parent: j.y),
                    _ => b.AddJointFixed(j.y, j.z)
                };
                if (!articulations.TryGetValue(j.w, out var list)) articulations[j.w] = list = new List<int>();
                list.Add(index);
            }
            foreach (var a in articulations.Values) b.AddArticulation(a);
            return b.Finalize();
        }

        // Contact records on Newton's poses at every step, matched within each shape pair by the nearest points.
        [Test]
        public void CollideMatchesNewton()
        {
            var fx = new Fixture("body_collide");
            var model = ModelFrom(fx);
            var steps = (int)fx.Value("steps");
            var poses = fx.Array<Transform>("steps_body_q");
            var table = fx.Array<float>("steps_contacts");
            var width = fx.Shape("steps_contacts")[1];
            var refCounts = fx.Array<int>("steps_contact_count");
            var nb = model.BodyCount;

            using var pipeline = new CollisionPipeline(model);
            using var contacts = pipeline.CreateContacts();
            using var state = new State(model);
            using var cmd = new CommandBuffer();
            var max = contacts.RigidContactMax;
            var count = new int[1];
            var shape0 = new int[max];
            var shape1 = new int[max];
            var point0 = new float3[max];
            var point1 = new float3[max];
            var offset0 = new float3[max];
            var offset1 = new float3[max];
            var normal = new float3[max];
            var margin0 = new float[max];
            var margin1 = new float[max];
            int missing = 0, extra = 0;
            var worst = new SortedDictionary<string, float2>();
            var report = new System.Text.StringBuilder();
            for (var s = 0; s < steps; s++)
            {
                state.BodyQ.SetData(poses, s * nb, 0, nb);
                cmd.Clear();
                pipeline.Collide(cmd, state, contacts);
                UnityEngine.Graphics.ExecuteCommandBuffer(cmd);
                contacts.RigidContactCount.GetData(count);
                contacts.RigidContactShape0.GetData(shape0);
                contacts.RigidContactShape1.GetData(shape1);
                contacts.RigidContactPoint0.GetData(point0);
                contacts.RigidContactPoint1.GetData(point1);
                contacts.RigidContactOffset0.GetData(offset0);
                contacts.RigidContactOffset1.GetData(offset1);
                contacts.RigidContactNormal.GetData(normal);
                contacts.RigidContactMargin0.GetData(margin0);
                contacts.RigidContactMargin1.GetData(margin1);
                var n = math.min(count[0], max);
                var matched = new bool[n];
                for (var r = 0; r < refCounts[s]; r++)
                {
                    var row = new System.ReadOnlySpan<float>(table, (s * width + r) * Fields, Fields);
                    int a = (int)row[0], b = (int)row[1];
                    var p0 = new float3(row[2], row[3], row[4]);
                    var p1 = new float3(row[5], row[6], row[7]);
                    var best = -1;
                    var bestD = float.MaxValue;
                    for (var j = 0; j < n; j++)
                    {
                        if (matched[j] || shape0[j] != a || shape1[j] != b) continue;
                        var d = math.length(point0[j] - p0) + math.length(point1[j] - p1);
                        if (d >= bestD) continue;
                        bestD = d;
                        best = j;
                    }
                    if (best < 0)
                    {
                        if (missing++ < 10) report.AppendLine($"step {s}: missing {a}-{b} at {p0} {p1}");
                        continue;
                    }
                    matched[best] = true;
                    var dp = math.max(math.max(math.length(point0[best] - p0), math.length(point1[best] - p1)),
                        math.max(math.length(offset0[best] - new float3(row[8], row[9], row[10])),
                            math.length(offset1[best] - new float3(row[11], row[12], row[13]))));
                    dp = math.max(dp, math.max(math.abs(margin0[best] - row[17]), math.abs(margin1[best] - row[18])));
                    var dn = math.length(normal[best] - new float3(row[14], row[15], row[16]));
                    var key = $"{model.ShapeType[a]}-{model.ShapeType[b]}";
                    worst[key] = math.max(worst.TryGetValue(key, out var w) ? w : 0f, new float2(dp, dn));
                }
                for (var j = 0; j < n; j++)
                    if (!matched[j] && extra++ < 10)
                        report.AppendLine($"step {s}: extra {shape0[j]}-{shape1[j]} at {point0[j]} {point1[j]}");
            }
            var summary = string.Join(", ", worst.Select(kv => $"{kv.Key}: {kv.Value.x * 1e3f:F5} mm, normal {kv.Value.y:E1}"));
            UnityEngine.Debug.Log($"body collide: missing {missing}, extra {extra}; per type pair {summary}\n{report}");
            Assert.AreEqual(0, missing + extra);
            foreach (var w in worst.Values)
            {
                Assert.Less(w.x, 1e-5f);
                Assert.Less(w.y, 1e-5f);
            }
        }
    }
}
