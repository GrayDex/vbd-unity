using Unity.Mathematics;

namespace Vbd
{
    static class Adjacency
    {
        // Per particle, the (tet, vertex order) pairs in ascending tet order, as Newton's MeshAdjacency.
        public static (int2[] pairs, int[] offsets) VertexTets(int4[] tets, int particleCount)
        {
            var offsets = new int[particleCount + 1];
            foreach (var t in tets)
                for (var k = 0; k < 4; k++)
                    offsets[t[k] + 1]++;
            for (var i = 0; i < particleCount; i++) offsets[i + 1] += offsets[i];
            var fill = (int[])offsets.Clone();
            var pairs = new int2[offsets[particleCount]];
            for (var tet = 0; tet < tets.Length; tet++)
                for (var k = 0; k < 4; k++)
                    pairs[fill[tets[tet][k]]++] = new int2(tet, k);
            return (pairs, offsets);
        }

        // Per body, its constrained joints in ascending order, as Newton's _compute_rigid_force_element_adjacency:
        // FREE joints and world endpoints are left out, disabled joints kept.
        public static (int[] joints, int[] offsets) BodyJoints(Model model)
        {
            var offsets = new int[model.BodyCount + 1];
            for (var j = 0; j < model.JointCount; j++)
                if (model.JointType[j] != Model.JointFree)
                    foreach (var b in new[] { model.JointParent[j], model.JointChild[j] })
                        if (b >= 0) offsets[b + 1]++;
            for (var b = 0; b < model.BodyCount; b++) offsets[b + 1] += offsets[b];
            var fill = (int[])offsets.Clone();
            var joints = new int[offsets[model.BodyCount]];
            for (var j = 0; j < model.JointCount; j++)
                if (model.JointType[j] != Model.JointFree)
                    foreach (var b in new[] { model.JointParent[j], model.JointChild[j] })
                        if (b >= 0) joints[fill[b]++] = j;
            return (joints, offsets);
        }
    }
}
