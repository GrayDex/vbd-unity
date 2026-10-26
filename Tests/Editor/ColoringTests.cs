using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;

namespace Vbd.Tests.Editor
{
    // Graph coloring against Newton / Warp: valid, and the same colors node by node.
    public class ColoringTests
    {
        static readonly (string name, ColoringAlgorithm algorithm, bool balance)[] Variants =
        {
            ("mcs_bal", ColoringAlgorithm.Mcs, true), ("mcs_raw", ColoringAlgorithm.Mcs, false),
            ("greedy_bal", ColoringAlgorithm.Greedy, true), ("greedy_raw", ColoringAlgorithm.Greedy, false)
        };

        static int[] ColorsOf(System.Collections.Generic.List<int[]> groups, int count)
        {
            var colors = Enumerable.Repeat(-1, count).ToArray();
            for (var c = 0; c < groups.Count; c++)
                foreach (var node in groups[c])
                    colors[node] = c;
            return colors;
        }

        static string Sizes(System.Collections.Generic.List<int[]> groups) => string.Join("/", groups.Select(g => g.Length));

        static ModelBuilder GridBuilder(Fixture fx)
        {
            var builder = new ModelBuilder();
            builder.AddSoftGrid(new float3(0.5f, -1f, 2f), new quaternion(fx.Array<float>("rot")[0], fx.Array<float>("rot")[1],
                    fx.Array<float>("rot")[2], fx.Array<float>("rot")[3]), new float3(0.1f, 0f, -0.2f), new int3(5, 3, 2),
                new float3(0.1f, 0.2f, 0.15f), 800f, 7e4f, 2e4f, 30f, fixLeft: true, fixBottom: true, particleRadius: 0.02f);
            return builder;
        }

        static ModelBuilder MeshBuilder(Fixture fx)
        {
            var builder = new ModelBuilder();
            var r = fx.Array<float>("in_rot");
            UnityEngine.TestTools.LogAssert.Expect(UnityEngine.LogType.Warning, "inverted tetrahedral element");
            builder.AddSoftMesh(fx.Array<float3>("in_pos")[0], new quaternion(r[0], r[1], r[2], r[3]), fx.Value("in_scale"),
                fx.Array<float3>("in_vel")[0], fx.Array<float3>("in_vertices"), fx.Array<int>("in_indices"),
                fx.Value("in_density"), fx.Array<float>("in_k_mu"), fx.Array<float>("in_k_lambda"),
                fx.Array<float>("in_k_damp"), 0.015f);
            return builder;
        }

        static ModelBuilder LargeBuilder()
        {
            var builder = new ModelBuilder();
            builder.AddSoftGrid(float3.zero, quaternion.identity, float3.zero, new int3(30, 10, 10), new float3(0.1f), 1000f,
                1e5f, 1e5f, 0f);
            return builder;
        }

        static void CheckAgainstNewton(string label, ModelBuilder builder, Fixture fx)
        {
            var active = builder.TetMaterials.Select(m => m.x != 0f || m.y != 0f).ToArray();
            var edges = Coloring.ConstructTetmeshGraphEdges(builder.TetIndices, active);
            foreach (var (name, algorithm, balance) in Variants)
            {
                builder.Color(balance, 1.1f, algorithm);
                var colors = ColorsOf(builder.ParticleColorGroups, builder.ParticleCount);
                Assert.IsTrue(Coloring.IsValid(edges, colors), $"{label} {name}: two neighbors share a color");
                // No tetrahedron has two vertices of one color.
                foreach (var t in builder.TetIndices.Where((_, i) => active[i]))
                    Assert.AreEqual(4, new[] { colors[t.x], colors[t.y], colors[t.z], colors[t.w] }.Distinct().Count(), $"{label} {name}");
                var expected = fx.Array<int>("colors_" + name);
                var differing = Enumerable.Range(0, expected.Length).Count(i => expected[i] != colors[i]);
                UnityEngine.Debug.Log($"{label} {name}: {builder.ParticleColorGroups.Count} colors, sizes {Sizes(builder.ParticleColorGroups)}, " +
                                      $"{differing} of {expected.Length} nodes differ from Newton");
                CollectionAssert.AreEqual(expected, colors, $"{label} {name}");
            }
        }

        [Test]
        public void SoftGridColorsMatchNewton()
        {
            var fx = new Fixture("builder_grid");
            CheckAgainstNewton("grid", GridBuilder(fx), fx);
        }

        [Test]
        public void SoftMeshColorsMatchNewton()
        {
            var fx = new Fixture("builder_mesh");
            CheckAgainstNewton("mesh", MeshBuilder(fx), fx);
        }

        [Test]
        public void LargeGridColorsMatchNewton()
        {
            CheckAgainstNewton("large", LargeBuilder(), new Fixture("builder_large"));
        }

        [Test]
        public void LargeGridColorsFast()
        {
            var builder = LargeBuilder();
            foreach (var (name, algorithm, balance) in Variants)
            {
                var watch = Stopwatch.StartNew();
                builder.Color(balance, 1.1f, algorithm);
                watch.Stop();
                UnityEngine.Debug.Log($"large 30x10x10 grid ({builder.ParticleCount} particles, {builder.TetCount} tets) {name}: " +
                                      $"{watch.Elapsed.TotalMilliseconds:F1} ms, {builder.ParticleColorGroups.Count} colors, sizes {Sizes(builder.ParticleColorGroups)}");
                Assert.Less(watch.Elapsed.TotalMilliseconds, 2000);
            }
        }

        [Test]
        public void BodyColorsMatchNewton()
        {
            var fx = new Fixture("builder_bodies");
            var parents = fx.Array<int>("joint_parent");
            var children = fx.Array<int>("joint_child");
            const int n = 40;
            var edges = parents.Zip(children, (p, c) => new int2(p, c)).Where(e => e.x != -1 && e.x != e.y).ToArray();
            foreach (var (name, algorithm, balance) in Variants)
            {
                var groups = Coloring.ColorRigidBodies(n, parents, children, balance, 1.1f, algorithm);
                var colors = ColorsOf(groups, n);
                Assert.IsTrue(Coloring.IsValid(edges, colors), $"bodies {name}: two bodies of one joint share a color");
                var expected = fx.Array<int>("body_colors_" + name);
                UnityEngine.Debug.Log($"bodies {name}: {groups.Count} colors, sizes {Sizes(groups)}, " +
                                      $"{Enumerable.Range(0, n).Count(i => expected[i] != colors[i])} of {n} differ from Newton");
                CollectionAssert.AreEqual(expected, colors, $"bodies {name}");
            }
        }

        [Test]
        public void BodiesWithoutJointsShareOneColor()
        {
            var groups = Coloring.ColorRigidBodies(5, new[] { -1, -1 }, new[] { 0, 1 });
            Assert.AreEqual(1, groups.Count);
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4 }, groups[0]);
        }

        [Test]
        public void BuilderColorsParticlesAndBodiesIntoModel()
        {
            var builder = new ModelBuilder();
            builder.AddSoftGrid(float3.zero, quaternion.identity, float3.zero, new int3(3, 2, 2), new float3(0.1f), 1000f, 1e4f, 1e4f, 0f);
            for (var i = 0; i < 4; i++) builder.AddBody();
            builder.Color(jointParent: new[] { -1, 0, 1, 2 }, jointChild: new[] { 0, 1, 2, 3 });
            var model = builder.Finalize();
            Assert.AreEqual(model.ParticleCount, model.ColorParticles.Length);
            Assert.AreEqual(model.ParticleCount, model.ColorOffsets[model.ColorCount]);
            Assert.AreEqual(4, model.BodyColorBodies.Length);
            Assert.AreEqual(2, model.BodyColorCount, "a chain needs two colors");
        }
    }
}
