using Unity.Mathematics;

namespace Vbd
{
    // Newton's joint arrays. Per joint: type, bodies, anchor frames, start of its coordinates (JointQ) and
    // degrees of freedom (axes, drives, limits). Coordinates and DOFs differ in count only for BALL (4/3) and FREE (7/6).
    public sealed partial class Model
    {
        public const int JointPrismatic = 0;
        public const int JointRevolute = 1;
        public const int JointBall = 2;
        public const int JointFixed = 3;
        public const int JointFree = 4;
        public const int JointD6 = 6;

        // Newton's MAXVAL: a limit at ±MaxVal is no limit.
        public const float MaxVal = 1e10f;

        public int[] JointType = { };
        public bool[] JointEnabled = { };
        // Body indices, -1 for the world.
        public int[] JointParent = { };
        public int[] JointChild = { };
        // Joint frame in the parent's and in the child's body frame.
        public Transform[] JointXp = { };
        public Transform[] JointXc = { };
        public int[] JointQStart = { };
        public int[] JointQdStart = { };
        // Linear and angular DOF counts; linear DOFs come first.
        public int2[] JointDofDim = { };
        // Joint coordinates at BodyQ: drive targets and limits are measured from them.
        public float[] JointQ = { };
        // Joint velocities per DOF; the solver does not read them (body velocities live in State).
        public float[] JointQd = { };

        // Per DOF, in the joint's parent frame.
        public float3[] JointAxis = { };
        public float[] JointTargetKe = { };
        public float[] JointTargetKd = { };
        public float[] JointLimitLower = { };
        public float[] JointLimitUpper = { };
        public float[] JointLimitKe = { };
        public float[] JointLimitKd = { };

        // Initial drive targets for Control: positions per coordinate, velocities per DOF.
        public float[] JointTargetQ = { };
        public float[] JointTargetQd = { };
        // Initial generalized joint forces for Control, per DOF; empty means zero.
        public float[] JointF = { };

        public int JointCount => JointType.Length;
        public int JointDofCount => JointAxis.Length;
    }
}
