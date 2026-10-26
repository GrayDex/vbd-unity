using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Vbd
{
    // Port of the part of newton.ModelBuilder that SolverVbd needs: particles, tetrahedra, rigid bodies and shapes
    // (ModelBuilder.Rigid.cs), graph coloring and Finalize. Axes stay as given; the default gravity is Newton's, -Z.
    // Joints are not here yet: Color takes the joint parent/child arrays instead.
    public sealed partial class ModelBuilder
    {
        public float3 Gravity = new(0f, 0f, -9.81f);
        public float DefaultParticleRadius = 0.1f;
        public float DefaultTetDensity = 1f;
        public float DefaultTetKMu = 1e3f;
        public float DefaultTetKLambda = 1e3f;
        public float DefaultTetKDamp = 0f;
        // Newton's Model defaults for soft contacts.
        public float SoftContactKe = 1e3f;
        public float SoftContactKd = 10f;
        public float SoftContactMu = 0.5f;

        public readonly List<float3> ParticleQ = new();
        public readonly List<float3> ParticleQd = new();
        public readonly List<float> ParticleMass = new();
        public readonly List<float> ParticleRadius = new();
        public readonly List<int> ParticleFlags = new();

        public readonly List<int4> TetIndices = new();
        // Inverse rest edge matrix Dm⁻¹, as Model.TetPoses.
        public readonly List<float3x3> TetPoses = new();
        public readonly List<float3> TetMaterials = new();

        public List<int[]> ParticleColorGroups = new();
        public List<int[]> BodyColorGroups = new();

        public int ParticleCount => ParticleQ.Count;
        public int TetCount => TetIndices.Count;

        // add_particle; a negative radius means DefaultParticleRadius.
        public int AddParticle(float3 pos, float3 vel, float mass, float radius = -1f, int flags = Model.ParticleActive)
        {
            ParticleQ.Add(pos);
            ParticleQd.Add(vel);
            ParticleMass.Add(mass);
            ParticleRadius.Add(radius < 0f ? DefaultParticleRadius : radius);
            ParticleFlags.Add(flags);
            return ParticleQ.Count - 1;
        }

        // add_tetrahedron: returns the rest volume; an inverted or flat element is skipped, as Newton does.
        public float AddTetrahedron(int i, int j, int k, int l, float kMu = 1e3f, float kLambda = 1e3f, float kDamp = 0f)
        {
            double3 p = ParticleQ[i];
            var dm = new double3x3((double3)ParticleQ[j] - p, (double3)ParticleQ[k] - p, (double3)ParticleQ[l] - p);
            var volume = math.determinant(dm) / 6.0;
            if (volume <= 0.0)
            {
                Debug.LogWarning("inverted tetrahedral element");
                return (float)volume;
            }
            TetIndices.Add(new int4(i, j, k, l));
            var inv = math.inverse(dm);
            TetPoses.Add(new float3x3((float3)inv.c0, (float3)inv.c1, (float3)inv.c2));
            TetMaterials.Add(new float3(kMu, kLambda, kDamp));
            return (float)volume;
        }

        // add_soft_grid: dim cells per axis, five tetrahedra per cell, vertices ordered x fastest then y then z.
        // Every vertex carries one cell's mass; fixed sides get zero mass.
        public void AddSoftGrid(float3 pos, quaternion rot, float3 vel, int3 dim, float3 cell, float density,
            float kMu, float kLambda, float kDamp, bool fixLeft = false, bool fixRight = false, bool fixTop = false,
            bool fixBottom = false, float particleRadius = -1f)
        {
            var startVertex = ParticleQ.Count;
            var mass = (float)((double)cell.x * cell.y * cell.z * density);
            for (var z = 0; z <= dim.z; z++)
            for (var y = 0; y <= dim.y; y++)
            for (var x = 0; x <= dim.x; x++)
            {
                var m = mass;
                if (fixLeft && x == 0) m = 0f;
                if (fixRight && x == dim.x) m = 0f;
                if (fixTop && y == dim.y) m = 0f;
                if (fixBottom && y == 0) m = 0f;
                var p = math.rotate(rot, new float3(x * cell.x, y * cell.y, z * cell.z)) + pos;
                AddParticle(p, vel, m, particleRadius);
            }

            int GridIndex(int x, int y, int z) => (dim.x + 1) * (dim.y + 1) * z + (dim.x + 1) * y + x + startVertex;
            void AddTet(int i, int j, int k, int l) => AddTetrahedron(i, j, k, l, kMu, kLambda, kDamp);

            for (var z = 0; z < dim.z; z++)
            for (var y = 0; y < dim.y; y++)
            for (var x = 0; x < dim.x; x++)
            {
                var v0 = GridIndex(x, y, z);
                var v1 = GridIndex(x + 1, y, z);
                var v2 = GridIndex(x + 1, y, z + 1);
                var v3 = GridIndex(x, y, z + 1);
                var v4 = GridIndex(x, y + 1, z);
                var v5 = GridIndex(x + 1, y + 1, z);
                var v6 = GridIndex(x + 1, y + 1, z + 1);
                var v7 = GridIndex(x, y + 1, z + 1);
                if (((x & 1) ^ (y & 1) ^ (z & 1)) != 0)
                {
                    AddTet(v0, v1, v4, v3);
                    AddTet(v2, v3, v6, v1);
                    AddTet(v5, v4, v1, v6);
                    AddTet(v7, v6, v3, v4);
                    AddTet(v4, v1, v6, v3);
                }
                else
                {
                    AddTet(v1, v2, v5, v0);
                    AddTet(v3, v0, v7, v2);
                    AddTet(v4, v7, v0, v5);
                    AddTet(v6, v5, v2, v7);
                    AddTet(v5, v2, v7, v0);
                }
            }
        }

        // add_soft_mesh: vertices are scaled, rotated, moved; indices are 4 per tetrahedron. kMu, kLambda, kDamp hold
        // one value for all elements or one per element. A quarter of each element's mass goes to each corner.
        public void AddSoftMesh(float3 pos, quaternion rot, float scale, float3 vel, IReadOnlyList<float3> vertices,
            IReadOnlyList<int> indices, float? density = null, IReadOnlyList<float> kMu = null,
            IReadOnlyList<float> kLambda = null, IReadOnlyList<float> kDamp = null, float? particleRadius = null)
        {
            var rho = density ?? DefaultTetDensity;
            var numTets = indices.Count / 4;
            float Pick(IReadOnlyList<float> values, float fallback, int t) =>
                values == null ? fallback : values.Count == 1 ? values[0] : values.Count == numTets
                    ? values[t]
                    : throw new ArgumentException($"Material array length {values.Count} is neither 1 nor {numTets}");

            var startVertex = ParticleQ.Count;
            foreach (var v in vertices)
                AddParticle(math.rotate(rot, v * scale) + pos, vel, 0f, particleRadius ?? -1f);

            for (var t = 0; t < numTets; t++)
            {
                var v0 = startVertex + indices[t * 4];
                var v1 = startVertex + indices[t * 4 + 1];
                var v2 = startVertex + indices[t * 4 + 2];
                var v3 = startVertex + indices[t * 4 + 3];
                var volume = AddTetrahedron(v0, v1, v2, v3, Pick(kMu, DefaultTetKMu, t),
                    Pick(kLambda, DefaultTetKLambda, t), Pick(kDamp, DefaultTetKDamp, t));
                if (!(volume > 0f)) continue;
                var share = (float)(rho * (double)volume / 4.0);
                ParticleMass[v0] += share;
                ParticleMass[v1] += share;
                ParticleMass[v2] += share;
                ParticleMass[v3] += share;
            }
        }

        public void AddSoftMesh(float3 pos, quaternion rot, float scale, float3 vel, IReadOnlyList<float3> vertices,
            IReadOnlyList<int> indices, float density, float kMu, float kLambda, float kDamp, float? particleRadius = null) =>
            AddSoftMesh(pos, rot, scale, vel, vertices, indices, density, new[] { kMu }, new[] { kLambda }, new[] { kDamp },
                particleRadius);

        // color(): fills ParticleColorGroups from the active tetrahedra (Lamé μ or λ not zero, as SolverVbd) and
        // BodyColorGroups from the joints. Call before Finalize.
        public void Color(bool balanceColors = true, float targetMaxMinColorRatio = 1.1f,
            ColoringAlgorithm algorithm = ColoringAlgorithm.Mcs, IReadOnlyList<int> jointParent = null,
            IReadOnlyList<int> jointChild = null, IReadOnlyList<int2> additionalBodyEdges = null)
        {
            if (ParticleCount != 0)
            {
                var active = new bool[TetCount];
                for (var t = 0; t < active.Length; t++) active[t] = TetMaterials[t].x != 0f || TetMaterials[t].y != 0f;
                var edges = Coloring.ConstructTetmeshGraphEdges(TetIndices, active);
                if (edges.Length > 0)
                    ParticleColorGroups = Coloring.ColorGraph(ParticleCount, edges, balanceColors, targetMaxMinColorRatio, algorithm);
                else
                {
                    var all = new int[ParticleCount];
                    for (var i = 0; i < all.Length; i++) all[i] = i;
                    ParticleColorGroups = new List<int[]> { all };
                }
            }
            BodyColorGroups = Coloring.ColorRigidBodies(BodyCount, jointParent, jointChild, balanceColors,
                targetMaxMinColorRatio, algorithm, additionalBodyEdges);
        }

        static void Flatten(List<int[]> groups, out int[] items, out int[] offsets)
        {
            offsets = new int[groups.Count + 1];
            for (var g = 0; g < groups.Count; g++) offsets[g + 1] = offsets[g] + groups[g].Length;
            items = new int[offsets[groups.Count]];
            for (var g = 0; g < groups.Count; g++) groups[g].CopyTo(items, offsets[g]);
        }

        // finalize: copies the arrays into a Model; rigid body mass properties pass through the same validation and
        // correction as Newton's (negative or non-finite mass, non-positive-definite or inconsistent inertia).
        public Model Finalize()
        {
            var model = new Model
            {
                Gravity = Gravity,
                ParticleQ = ParticleQ.ToArray(),
                ParticleQd = ParticleQd.ToArray(),
                ParticleMass = ParticleMass.ToArray(),
                ParticleRadius = ParticleRadius.ToArray(),
                ParticleFlags = ParticleFlags.ToArray(),
                TetIndices = TetIndices.ToArray(),
                TetPoses = TetPoses.ToArray(),
                TetMaterials = TetMaterials.ToArray(),
                SoftContactKe = SoftContactKe,
                SoftContactKd = SoftContactKd,
                SoftContactMu = SoftContactMu
            };
            if (ParticleColorGroups.Count > 0)
            {
                Flatten(ParticleColorGroups, out model.ColorParticles, out model.ColorOffsets);
            }
            if (BodyColorGroups.Count > 0)
            {
                Flatten(BodyColorGroups, out model.BodyColorBodies, out model.BodyColorOffsets);
            }
            FinalizeRigid(model);
            return model;
        }
    }
}
