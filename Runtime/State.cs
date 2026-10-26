using System;
using UnityEngine;

namespace Vbd
{
    // Newton's State on the GPU: what one step advances.
    public sealed class State : IDisposable
    {
        public readonly GraphicsBuffer ParticleQ;
        public readonly GraphicsBuffer ParticleQd;

        public State(Model model)
        {
            var n = Math.Max(model.ParticleCount, 1);
            ParticleQ = new GraphicsBuffer(GraphicsBuffer.Target.Structured, n, 12);
            ParticleQd = new GraphicsBuffer(GraphicsBuffer.Target.Structured, n, 12);
            if (model.ParticleCount == 0) return;
            ParticleQ.SetData(model.ParticleQ);
            ParticleQd.SetData(model.ParticleQd);
        }

        public void Dispose()
        {
            ParticleQ.Dispose();
            ParticleQd.Dispose();
        }
    }
}
