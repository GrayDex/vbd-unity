using System;
using UnityEngine;

namespace Vbd
{
    // Newton's Control on the GPU: joint drive targets, written by the caller between steps.
    public sealed class Control : IDisposable
    {
        // Per joint coordinate (Model.JointQStart), and per DOF (Model.JointQdStart).
        public readonly GraphicsBuffer JointTargetQ;
        public readonly GraphicsBuffer JointTargetQd;

        public Control(Model model)
        {
            JointTargetQ = Upload(model.JointTargetQ);
            JointTargetQd = Upload(model.JointTargetQd);
        }

        public void Dispose()
        {
            JointTargetQ.Dispose();
            JointTargetQd.Dispose();
        }

        static GraphicsBuffer Upload(float[] data)
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(data.Length, 1), sizeof(float));
            if (data.Length > 0) b.SetData(data);
            return b;
        }
    }
}
