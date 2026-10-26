using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Vbd
{
    // Newton's transform: body origin and rotation (x, y, z, w); 28 bytes on the GPU.
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform
    {
        public float3 P;
        public quaternion Q;

        public Transform(float3 p, quaternion q)
        {
            P = p;
            Q = q;
        }
    }

    // Newton's spatial velocity: linear velocity of the center of mass and angular velocity, both in world frame.
    [StructLayout(LayoutKind.Sequential)]
    public struct SpatialVelocity
    {
        public float3 Linear;
        public float3 Angular;
    }

    public sealed partial class Model
    {
        public Transform[] BodyQ = { };
        public SpatialVelocity[] BodyQd = { };
        // Center of mass in the body frame.
        public float3[] BodyCom = { };
        public float[] BodyMass = { };
        public float[] BodyInvMass = { };
        // Inertia about the center of mass, body frame.
        public float3x3[] BodyInertia = { };
        public float3x3[] BodyInvInertia = { };

        // Bodies grouped by color: no two bodies of one color share a joint.
        public int[] BodyColorBodies = { };
        public int[] BodyColorOffsets = { 0 };

        public int BodyCount => BodyQ.Length;
        public int BodyColorCount => BodyColorOffsets.Length - 1;
    }
}
