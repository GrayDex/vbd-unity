using System;
using UnityEngine;

namespace Vbd
{
    // Newton's Control on the GPU: joint drive targets and joint forces, written by the caller between steps.
    public sealed class Control : IDisposable
    {
        // Per joint coordinate (Model.JointQStart), and per DOF (Model.JointQdStart).
        public readonly GraphicsBuffer JointTargetQ;
        public readonly GraphicsBuffer JointTargetQd;
        // Newton's joint_f per DOF: force along a linear axis, torque about an angular one; FREE takes a world-frame
        // force and torque at the child's center of mass, BALL a world-frame torque.
        public readonly GraphicsBuffer JointF;

        public Control(Model model)
        {
            JointTargetQ = Upload(model.JointTargetQ);
            JointTargetQd = Upload(model.JointTargetQd);
            JointF = Upload(model.JointF.Length == model.JointDofCount ? model.JointF : new float[model.JointDofCount]);
        }

        public void Dispose()
        {
            JointTargetQ.Dispose();
            JointTargetQd.Dispose();
            JointF.Dispose();
        }

        static GraphicsBuffer Upload(float[] data)
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(data.Length, 1), sizeof(float));
            b.SetData(data.Length > 0 ? data : new float[1]);
            return b;
        }
    }
}
