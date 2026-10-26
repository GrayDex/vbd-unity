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
    }
}
