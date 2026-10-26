using System;
using UnityEngine;

namespace Vbd
{
    // Newton's Contacts, particle-shape part: records written by CollisionPipeline, read by the solver.
    // The count lives on the GPU; kernels run over the capacity and skip records past it.
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

        public Contacts(int softContactMax)
        {
            SoftContactMax = Math.Max(softContactMax, 1);
            SoftContactCount = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, 4);
            SoftContactCount.SetData(new[] { 0 });
            SoftContactParticle = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 4);
            SoftContactShape = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 4);
            SoftContactBodyPos = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);
            SoftContactBodyVel = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);
            SoftContactNormal = new GraphicsBuffer(GraphicsBuffer.Target.Structured, SoftContactMax, 12);
        }

        public void Dispose()
        {
            SoftContactCount.Dispose();
            SoftContactParticle.Dispose();
            SoftContactShape.Dispose();
            SoftContactBodyPos.Dispose();
            SoftContactBodyVel.Dispose();
            SoftContactNormal.Dispose();
        }
    }
}
