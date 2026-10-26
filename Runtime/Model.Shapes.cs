using Unity.Mathematics;

namespace Vbd
{
    // Newton's GeoType values for the shapes this port handles.
    public static class GeoType
    {
        public const int Plane = 1;
        public const int Sphere = 3;
        public const int Capsule = 4;
        public const int Box = 7;
    }

    public sealed partial class Model
    {
        public const int ShapeCollideShapes = 1 << 1;
        public const int ShapeCollideParticles = 1 << 2;

        // Body each shape is attached to, -1 for the world.
        public int[] ShapeBody = { };
        // Shape frame in its body's frame (world frame for world shapes).
        public Transform[] ShapeTransform = { };
        public int[] ShapeType = { };
        // As Newton: sphere radius in x; capsule radius in x, half height along local z in y; box half extents.
        public float3[] ShapeScale = { };
        public int[] ShapeFlags = { };
        public float[] ShapeMargin = { };
        public float[] ShapeMaterialKe = { };
        public float[] ShapeMaterialKd = { };
        public float[] ShapeMaterialMu = { };
        // Newton's shape_collision_group: a positive group collides with itself and negative groups, a negative one
        // with every group but itself, 0 with none. Empty puts every shape in group 1, the builder's default.
        public int[] ShapeCollisionGroup = { };
        // Shape pairs, smaller index first, that never collide (Newton's shape_collision_filter_pairs).
        public int2[] ShapeCollisionFilterPairs = { };

        public float[] ParticleRadius = { };
        public float SoftContactKe = 1e3f;
        public float SoftContactKd = 10f;
        public float SoftContactMu = 0.5f;

        public int ShapeCount => ShapeType.Length;
    }
}
