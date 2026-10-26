using Unity.Mathematics;

namespace Vbd
{
    // Newton's Model: everything the solver reads but never changes.
    public sealed class Model
    {
        public const int ParticleActive = 1;

        public float3 Gravity = new(0f, -9.81f, 0f);

        public float3[] ParticleQ = { };
        public float3[] ParticleQd = { };
        public float[] ParticleMass = { };
        public int[] ParticleFlags = { };

        public int4[] TetIndices = { };
        // Inverse rest edge matrix Dm⁻¹ per tetrahedron; rest volume is 1 / (6 det Dm⁻¹).
        public float3x3[] TetPoses = { };
        // Lamé μ, λ and damping per tetrahedron.
        public float3[] TetMaterials = { };

        // Particles grouped by color: no two particles of one color share an element.
        public int[] ColorParticles = { };
        public int[] ColorOffsets = { 0 };

        public int ParticleCount => ParticleQ.Length;
        public int TetCount => TetIndices.Length;
        public int ColorCount => ColorOffsets.Length - 1;
    }
}
