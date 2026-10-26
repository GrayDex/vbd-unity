using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Newton's SolverVBD: one step is initialize, `Iterations` passes over the colors, finalize.
    // Records into a command buffer instead of launching: the caller executes it once per frame or replays it.
    public sealed class SolverVbd : IDisposable
    {
        const int GroupSize = 64;
        const int ParticlesPerGroup = 4;

        static class Id
        {
            public static readonly int Q = Shader.PropertyToID("_Q");
            public static readonly int Qd = Shader.PropertyToID("_Qd");
            public static readonly int QPrev = Shader.PropertyToID("_QPrev");
            public static readonly int Inertia = Shader.PropertyToID("_Inertia");
            public static readonly int Mass = Shader.PropertyToID("_Mass");
            public static readonly int Flags = Shader.PropertyToID("_Flags");
            public static readonly int TetIndices = Shader.PropertyToID("_TetIndices");
            public static readonly int TetPoses = Shader.PropertyToID("_TetPoses");
            public static readonly int TetMaterials = Shader.PropertyToID("_TetMaterials");
            public static readonly int AdjTets = Shader.PropertyToID("_AdjTets");
            public static readonly int AdjTetsOffsets = Shader.PropertyToID("_AdjTetsOffsets");
            public static readonly int ColorParticles = Shader.PropertyToID("_ColorParticles");
            public static readonly int ParticleCount = Shader.PropertyToID("_ParticleCount");
            public static readonly int ColorStart = Shader.PropertyToID("_ColorStart");
            public static readonly int ColorSize = Shader.PropertyToID("_ColorSize");
            public static readonly int Dt = Shader.PropertyToID("_Dt");
            public static readonly int Gravity = Shader.PropertyToID("_Gravity");
        }

        public int Iterations;

        readonly Model _model;
        readonly ComputeShader _particles;
        readonly int _forwardStep, _solveElasticity, _updateVelocity;
        readonly List<GraphicsBuffer> _buffers = new();

        public SolverVbd(Model model, int iterations = 10)
        {
            _model = model;
            Iterations = iterations;
            _particles = Resources.Load<ComputeShader>("Vbd/ParticleVbd");
            _forwardStep = _particles.FindKernel("ForwardStep");
            _solveElasticity = _particles.FindKernel("SolveElasticity");
            _updateVelocity = _particles.FindKernel("UpdateVelocity");

            var n = model.ParticleCount;
            var qPrev = Buffer<float3>(n);
            var inertia = Buffer<float3>(n);
            var (adjPairs, adjOffsets) = Adjacency.VertexTets(model.TetIndices, n);
            // Rows of Dm⁻¹: Unity.Mathematics stores matrices by column.
            var poseRows = new float3x3[model.TetCount];
            for (var i = 0; i < poseRows.Length; i++) poseRows[i] = math.transpose(model.TetPoses[i]);

            foreach (var k in new[] { _forwardStep, _solveElasticity, _updateVelocity })
            {
                _particles.SetBuffer(k, Id.QPrev, qPrev);
                _particles.SetBuffer(k, Id.Inertia, inertia);
                _particles.SetBuffer(k, Id.Mass, Upload(model.ParticleMass));
                _particles.SetBuffer(k, Id.Flags, Upload(model.ParticleFlags));
            }
            _particles.SetBuffer(_solveElasticity, Id.TetIndices, Upload(model.TetIndices));
            _particles.SetBuffer(_solveElasticity, Id.TetPoses, Upload(poseRows));
            _particles.SetBuffer(_solveElasticity, Id.TetMaterials, Upload(model.TetMaterials));
            _particles.SetBuffer(_solveElasticity, Id.AdjTets, Upload(adjPairs));
            _particles.SetBuffer(_solveElasticity, Id.AdjTetsOffsets, Upload(adjOffsets));
            _particles.SetBuffer(_solveElasticity, Id.ColorParticles, Upload(model.ColorParticles));
        }

        public void Step(CommandBuffer cmd, State state, float dt)
        {
            var n = _model.ParticleCount;
            if (n == 0) return;
            cmd.SetComputeIntParam(_particles, Id.ParticleCount, n);
            cmd.SetComputeFloatParam(_particles, Id.Dt, dt);
            cmd.SetComputeVectorParam(_particles, Id.Gravity, (Vector3)_model.Gravity);
            foreach (var k in new[] { _forwardStep, _solveElasticity, _updateVelocity })
            {
                cmd.SetComputeBufferParam(_particles, k, Id.Q, state.ParticleQ);
                cmd.SetComputeBufferParam(_particles, k, Id.Qd, state.ParticleQd);
            }

            var particleGroups = (n + GroupSize - 1) / GroupSize;
            cmd.DispatchCompute(_particles, _forwardStep, particleGroups, 1, 1);
            for (var iter = 0; iter < Iterations; iter++)
                for (var c = 0; c < _model.ColorCount; c++)
                {
                    var start = _model.ColorOffsets[c];
                    var size = _model.ColorOffsets[c + 1] - start;
                    if (size == 0) continue;
                    cmd.SetComputeIntParam(_particles, Id.ColorStart, start);
                    cmd.SetComputeIntParam(_particles, Id.ColorSize, size);
                    cmd.DispatchCompute(_particles, _solveElasticity, (size + ParticlesPerGroup - 1) / ParticlesPerGroup, 1, 1);
                }
            cmd.DispatchCompute(_particles, _updateVelocity, particleGroups, 1, 1);
        }

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _buffers.Clear();
        }

        GraphicsBuffer Buffer<T>(int count) where T : struct
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(count, 1), Marshal.SizeOf<T>());
            _buffers.Add(b);
            return b;
        }

        GraphicsBuffer Upload<T>(T[] data) where T : struct
        {
            var b = Buffer<T>(data.Length);
            if (data.Length > 0) b.SetData(data);
            return b;
        }
    }
}
