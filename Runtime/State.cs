using System;
using UnityEngine;

namespace Vbd
{
    // Newton's State on the GPU: what one step advances.
    public sealed class State : IDisposable
    {
        public readonly GraphicsBuffer ParticleQ;
        public readonly GraphicsBuffer ParticleQd;
        public readonly GraphicsBuffer BodyQ;
        public readonly GraphicsBuffer BodyQd;

        public State(Model model)
        {
            ParticleQ = Upload(model.ParticleQ, 12);
            ParticleQd = Upload(model.ParticleQd, 12);
            BodyQ = Upload(model.BodyQ, 28);
            BodyQd = Upload(model.BodyQd, 24);
        }

        public void Dispose()
        {
            ParticleQ.Dispose();
            ParticleQd.Dispose();
            BodyQ.Dispose();
            BodyQd.Dispose();
        }

        static GraphicsBuffer Upload<T>(T[] data, int stride) where T : struct
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(data.Length, 1), stride);
            if (data.Length > 0) b.SetData(data);
            return b;
        }
    }
}
