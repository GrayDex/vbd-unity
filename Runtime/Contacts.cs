using System;
using UnityEngine;

namespace Vbd
{
    // Newton's Contacts: records written by CollisionPipeline, read by the solver.
    // The counts live on the GPU; kernels run over the capacity and skip records past them.
    public sealed class Contacts : IDisposable
    {
        public readonly int SoftContactMax;
        public readonly GraphicsBuffer SoftContactCount;
        public readonly GraphicsBuffer SoftContactParticle;
        public readonly GraphicsBuffer SoftContactShape;
        // Closest shape surface point in the body frame (world frame for world shapes).
        public readonly GraphicsBuffer SoftContactBodyPos;
        public readonly GraphicsBuffer SoftContactBodyVel;
        public readonly GraphicsBuffer SoftContactNormal;

        // Shape-shape contacts. The count may exceed RigidContactMax: records past the capacity are dropped.
        public readonly int RigidContactMax;
        public readonly GraphicsBuffer RigidContactCount;
        public readonly GraphicsBuffer RigidContactShape0;
        public readonly GraphicsBuffer RigidContactShape1;
        // Contact points on the shapes' core geometry (sphere center, capsule axis), each in its body's frame.
        public readonly GraphicsBuffer RigidContactPoint0;
        public readonly GraphicsBuffer RigidContactPoint1;
        // From the core point to the surface, in the body frame: (effective radius + margin) along the normal.
        public readonly GraphicsBuffer RigidContactOffset0;
        public readonly GraphicsBuffer RigidContactOffset1;
        // World frame, from shape 0 to shape 1.
        public readonly GraphicsBuffer RigidContactNormal;
        // Effective radius + margin of each shape.
        public readonly GraphicsBuffer RigidContactMargin0;
        public readonly GraphicsBuffer RigidContactMargin1;

        public Contacts(int softContactMax, int rigidContactMax = 0)
        {
            SoftContactMax = Math.Max(softContactMax, 1);
            SoftContactCount = Counter();
            SoftContactParticle = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 4);
            SoftContactShape = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 4);
            SoftContactBodyPos = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);
            SoftContactBodyVel = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);
            SoftContactNormal = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);

            RigidContactMax = Math.Max(rigidContactMax, 1);
            RigidContactCount = Counter();
            RigidContactShape0 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 4);
            RigidContactShape1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 4);
            RigidContactPoint0 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 12);
            RigidContactPoint1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 12);
            RigidContactOffset0 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 12);
            RigidContactOffset1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 12);
            RigidContactNormal = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 12);
            RigidContactMargin0 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 4);
            RigidContactMargin1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, RigidContactMax, 4);
        }

        static GraphicsBuffer Counter()
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, 4);
            b.SetData(new[] { 0 });
            return b;
        }

        public void Dispose()
        {
            SoftContactCount.Dispose();
            SoftContactParticle.Dispose();
            SoftContactShape.Dispose();
            SoftContactBodyPos.Dispose();
            SoftContactBodyVel.Dispose();
            SoftContactNormal.Dispose();
            RigidContactCount.Dispose();
            RigidContactShape0.Dispose();
            RigidContactShape1.Dispose();
            RigidContactPoint0.Dispose();
            RigidContactPoint1.Dispose();
            RigidContactOffset0.Dispose();
            RigidContactOffset1.Dispose();
            RigidContactNormal.Dispose();
            RigidContactMargin0.Dispose();
            RigidContactMargin1.Dispose();
        }
    }
}
