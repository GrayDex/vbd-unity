using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Particles: Newton's _initialize_particles, _solve_particle_iteration, _finalize_particles.
    public sealed partial class SolverVbd
    {
        const int ParticlesPerGroup = 4;

        static class ParticleId
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

        ComputeShader _particles;
        int _forwardStep, _solveElasticity, _updateVelocity;
        // Newton's particle_q_prev: positions at the start of the step.
        GraphicsBuffer _particleQPrev;

        void CreateParticles()
        {
            _particles = Resources.Load<ComputeShader>("Vbd/ParticleVbd");
            _forwardStep = _particles.FindKernel("ForwardStep");
            _solveElasticity = _particles.FindKernel("SolveElasticity");
            _updateVelocity = _particles.FindKernel("UpdateVelocity");

            var n = _model.ParticleCount;
            _particleQPrev = Buffer<float3>(n);
            var inertia = Buffer<float3>(n);
            var (adjPairs, adjOffsets) = Adjacency.VertexTets(_model.TetIndices, n);
            // Rows of Dm⁻¹: Unity.Mathematics stores matrices by column.
            var poseRows = new float3x3[_model.TetCount];
            for (var i = 0; i < poseRows.Length; i++) poseRows[i] = math.transpose(_model.TetPoses[i]);

            foreach (var k in new[] { _forwardStep, _solveElasticity, _updateVelocity })
            {
                _particles.SetBuffer(k, ParticleId.QPrev, _particleQPrev);
                _particles.SetBuffer(k, ParticleId.Inertia, inertia);
                _particles.SetBuffer(k, ParticleId.Mass, Upload(_model.ParticleMass));
                _particles.SetBuffer(k, ParticleId.Flags, Upload(_model.ParticleFlags));
            }
            _particles.SetBuffer(_solveElasticity, ParticleId.TetIndices, Upload(_model.TetIndices));
            _particles.SetBuffer(_solveElasticity, ParticleId.TetPoses, Upload(poseRows));
            _particles.SetBuffer(_solveElasticity, ParticleId.TetMaterials, Upload(_model.TetMaterials));
            _particles.SetBuffer(_solveElasticity, ParticleId.AdjTets, Upload(adjPairs));
            _particles.SetBuffer(_solveElasticity, ParticleId.AdjTetsOffsets, Upload(adjOffsets));
            _particles.SetBuffer(_solveElasticity, ParticleId.ColorParticles, Upload(_model.ColorParticles));
        }

        void InitializeParticles(CommandBuffer cmd, State state, float dt)
        {
            var n = _model.ParticleCount;
            if (n == 0) return;
            cmd.SetComputeIntParam(_particles, ParticleId.ParticleCount, n);
            cmd.SetComputeFloatParam(_particles, ParticleId.Dt, dt);
            cmd.SetComputeVectorParam(_particles, ParticleId.Gravity, (Vector3)_model.Gravity);
            foreach (var k in new[] { _forwardStep, _solveElasticity, _updateVelocity })
            {
                cmd.SetComputeBufferParam(_particles, k, ParticleId.Q, state.ParticleQ);
                cmd.SetComputeBufferParam(_particles, k, ParticleId.Qd, state.ParticleQd);
            }
            cmd.DispatchCompute(_particles, _forwardStep, Groups(n), 1, 1);
        }

        void SolveParticleIteration(CommandBuffer cmd)
        {
            for (var c = 0; c < _model.ColorCount; c++)
            {
                var start = _model.ColorOffsets[c];
                var size = _model.ColorOffsets[c + 1] - start;
                if (size == 0) continue;
                cmd.SetComputeIntParam(_particles, ParticleId.ColorStart, start);
                cmd.SetComputeIntParam(_particles, ParticleId.ColorSize, size);
                cmd.DispatchCompute(_particles, _solveElasticity, Groups(size, ParticlesPerGroup), 1, 1);
            }
        }

        void FinalizeParticles(CommandBuffer cmd)
        {
            if (_model.ParticleCount == 0) return;
            cmd.DispatchCompute(_particles, _updateVelocity, Groups(_model.ParticleCount), 1, 1);
        }
    }
}
