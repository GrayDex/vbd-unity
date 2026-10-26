using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Newton's SolverVBD: one step is initialize, `Iterations` passes (rigid bodies, then particles), finalize.
    // Records into a command buffer instead of launching: the caller executes it once per frame or replays it.
    // Each part lives in its own file and plugs into the step through the partial methods below.
    public sealed partial class SolverVbd : IDisposable
    {
        const int GroupSize = 64;

        public int Iterations;

        readonly Model _model;
        readonly List<GraphicsBuffer> _buffers = new();
        // Newton's body_q_prev: body poses at the start of the step, written by the rigid part, read by joints and contacts.
        readonly GraphicsBuffer _bodyQPrev;
        internal GraphicsBuffer BodyQPrev => _bodyQPrev;

        public SolverVbd(Model model, int iterations = 10)
        {
            _model = model;
            Iterations = iterations;
            _bodyQPrev = Upload(model.BodyQ);
            CreateParticles();
            CreateContacts();
            CreateRigidBodies();
            CreateBodyContacts();
            CreateBodyBodyContacts();
        }

        partial void CreateRigidBodies();
        partial void InitializeRigidBodies(CommandBuffer cmd, State state, float dt);
        partial void SolveRigidBodyIteration(CommandBuffer cmd, State state, Contacts contacts, float dt);
        // Per body color, before its solve: contact and attachment forces into the body scratch (Newton's accumulate_*_per_body).
        partial void AccumulateBodyForces(CommandBuffer cmd, Contacts contacts, int colorStart, int colorSize, float dt);
        partial void FinalizeRigidBodies(CommandBuffer cmd, State state, float dt);

        public void Step(CommandBuffer cmd, State state, float dt) => Step(cmd, state, null, dt);

        // `contacts` from CollisionPipeline.Collide on the same state, or null for none.
        public void Step(CommandBuffer cmd, State state, Contacts contacts, float dt)
        {
            InitializeBodyBodyContacts(cmd, state, contacts, dt);
            InitializeRigidBodies(cmd, state, dt);
            InitializeParticles(cmd, state, dt);
            InitializeContacts(cmd, state, contacts);
            InitializeBodyContacts(cmd, state, contacts);
            for (var iter = 0; iter < Iterations; iter++)
            {
                SolveRigidBodyIteration(cmd, state, contacts, dt);
                UpdateBodyBodyContactDuals(cmd, contacts);
                SolveParticleIteration(cmd);
            }
            FinalizeRigidBodies(cmd, state, dt);
            FinalizeParticles(cmd);
        }

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _buffers.Clear();
            _noContacts.Dispose();
        }

        static int Groups(int threads, int perGroup = GroupSize) => (threads + perGroup - 1) / perGroup;

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
