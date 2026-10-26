using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Newton's CollisionPipeline, particle-shape part: every step, before SolverVbd.Step.
    public sealed class CollisionPipeline : IDisposable
    {
        static class Id
        {
            public static readonly int ParticleQ = Shader.PropertyToID("_ParticleQ");
            public static readonly int ParticleRadius = Shader.PropertyToID("_ParticleRadius");
            public static readonly int ParticleFlags = Shader.PropertyToID("_ParticleFlags");
            public static readonly int BodyQ = Shader.PropertyToID("_BodyQ");
            public static readonly int ShapeBody = Shader.PropertyToID("_ShapeBody");
            public static readonly int ShapeTransform = Shader.PropertyToID("_ShapeTransform");
            public static readonly int ShapeType = Shader.PropertyToID("_ShapeType");
            public static readonly int ShapeScale = Shader.PropertyToID("_ShapeScale");
            public static readonly int ShapeFlags = Shader.PropertyToID("_ShapeFlags");
            public static readonly int ShapeMargin = Shader.PropertyToID("_ShapeMargin");
            public static readonly int ContactCount = Shader.PropertyToID("_ContactCount");
            public static readonly int ContactParticle = Shader.PropertyToID("_ContactParticle");
            public static readonly int ContactShape = Shader.PropertyToID("_ContactShape");
            public static readonly int ContactBodyPos = Shader.PropertyToID("_ContactBodyPos");
            public static readonly int ContactBodyVel = Shader.PropertyToID("_ContactBodyVel");
            public static readonly int ContactNormal = Shader.PropertyToID("_ContactNormal");
            public static readonly int ParticleCount = Shader.PropertyToID("_ParticleCount");
            public static readonly int ShapeCount = Shader.PropertyToID("_ShapeCount");
            public static readonly int ContactMax = Shader.PropertyToID("_ContactMax");
            public static readonly int Gap = Shader.PropertyToID("_Gap");
        }

        // Detection-only distance added to particle radius and shape margin, as Newton's soft_contact_gap.
        public float SoftContactGap = 0.01f;

        readonly Model _model;
        readonly ComputeShader _collide;
        readonly int _clearCount, _createSoftContacts;
        readonly List<GraphicsBuffer> _buffers = new();

        public CollisionPipeline(Model model)
        {
            _model = model;
            _collide = Resources.Load<ComputeShader>("Vbd/Collide");
            _clearCount = _collide.FindKernel("ClearCount");
            _createSoftContacts = _collide.FindKernel("CreateSoftContacts");
            var k = _createSoftContacts;
            _collide.SetBuffer(k, Id.ParticleRadius, Upload(model.ParticleRadius));
            _collide.SetBuffer(k, Id.ParticleFlags, Upload(model.ParticleFlags));
            _collide.SetBuffer(k, Id.ShapeBody, Upload(model.ShapeBody));
            _collide.SetBuffer(k, Id.ShapeTransform, Upload(model.ShapeTransform));
            _collide.SetBuffer(k, Id.ShapeType, Upload(model.ShapeType));
            _collide.SetBuffer(k, Id.ShapeScale, Upload(model.ShapeScale));
            _collide.SetBuffer(k, Id.ShapeFlags, Upload(model.ShapeFlags));
            _collide.SetBuffer(k, Id.ShapeMargin, Upload(model.ShapeMargin));
        }

        // Capacity for every particle × shape pair, as Newton's default soft_contact_max.
        public Contacts CreateContacts() => new(_model.ParticleCount * _model.ShapeCount);

        public void Collide(CommandBuffer cmd, State state, Contacts contacts)
        {
            cmd.SetComputeBufferParam(_collide, _clearCount, Id.ContactCount, contacts.SoftContactCount);
            cmd.DispatchCompute(_collide, _clearCount, 1, 1, 1);
            var pairs = _model.ParticleCount * _model.ShapeCount;
            if (pairs == 0) return;
            var k = _createSoftContacts;
            cmd.SetComputeIntParam(_collide, Id.ParticleCount, _model.ParticleCount);
            cmd.SetComputeIntParam(_collide, Id.ShapeCount, _model.ShapeCount);
            cmd.SetComputeIntParam(_collide, Id.ContactMax, contacts.SoftContactMax);
            cmd.SetComputeFloatParam(_collide, Id.Gap, SoftContactGap);
            cmd.SetComputeBufferParam(_collide, k, Id.ParticleQ, state.ParticleQ);
            cmd.SetComputeBufferParam(_collide, k, Id.BodyQ, state.BodyQ);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactCount, contacts.SoftContactCount);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactParticle, contacts.SoftContactParticle);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactShape, contacts.SoftContactShape);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactBodyPos, contacts.SoftContactBodyPos);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactBodyVel, contacts.SoftContactBodyVel);
            cmd.SetComputeBufferParam(_collide, k, Id.ContactNormal, contacts.SoftContactNormal);
            cmd.DispatchCompute(_collide, k, (pairs + 63) / 64, 1, 1);
        }

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _buffers.Clear();
        }

        GraphicsBuffer Upload<T>(T[] data) where T : struct
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(data.Length, 1), Marshal.SizeOf<T>());
            if (data.Length > 0) b.SetData(data);
            _buffers.Add(b);
            return b;
        }
    }
}
