using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Body-particle contacts, body side: the reaction of every particle contact on its dynamic body, gathered per
    // body color right before the body's solve, as Newton's _solve_rigid_body_iteration does.
    public sealed partial class SolverVbd
    {
        static class BodyContactId
        {
            public static readonly int ParticleQ = Shader.PropertyToID("_ParticleQ");
            public static readonly int ParticleQPrev = Shader.PropertyToID("_ParticleQPrev");
            public static readonly int ParticleRadius = Shader.PropertyToID("_ParticleRadius");
            public static readonly int ContactCount = Shader.PropertyToID("_ContactCount");
            public static readonly int ContactParticle = Shader.PropertyToID("_ContactParticle");
            public static readonly int ContactShape = Shader.PropertyToID("_ContactShape");
            public static readonly int ContactBodyPos = Shader.PropertyToID("_ContactBodyPos");
            public static readonly int ContactBodyVel = Shader.PropertyToID("_ContactBodyVel");
            public static readonly int ContactNormal = Shader.PropertyToID("_ContactNormal");
            public static readonly int ContactMaterial = Shader.PropertyToID("_ContactMaterial");
            public static readonly int ShapeBody = Shader.PropertyToID("_ShapeBody");
            public static readonly int ShapeMargin = Shader.PropertyToID("_ShapeMargin");
            public static readonly int BodyQ = Shader.PropertyToID("_BodyQ");
            public static readonly int BodyQPrev = Shader.PropertyToID("_BodyQPrev");
            public static readonly int BodyCom = Shader.PropertyToID("_BodyCom");
            public static readonly int BodyInvMass = Shader.PropertyToID("_BodyInvMass");
            public static readonly int ColorBodies = Shader.PropertyToID("_ColorBodies");
            public static readonly int BodyContactHead = Shader.PropertyToID("_BodyContactHead");
            public static readonly int BodyContactNext = Shader.PropertyToID("_BodyContactNext");
            public static readonly int BodyForces = Shader.PropertyToID("_BodyForces");
            public static readonly int HasBodyForces = Shader.PropertyToID("_HasBodyForces");
            public static readonly int BodyCount = Shader.PropertyToID("_BodyCount");
            public static readonly int ContactMax = Shader.PropertyToID("_ContactMax");
            public static readonly int ColorStart = Shader.PropertyToID("_ColorStart");
            public static readonly int ColorSize = Shader.PropertyToID("_ColorSize");
            public static readonly int Dt = Shader.PropertyToID("_Dt");
            public static readonly int FrictionEpsilon = Shader.PropertyToID("_FrictionEpsilon");
        }

        // Wrench: force, torque, and the linear, angular-linear and angular 3x3 Hessian blocks.
        const int WrenchStride = (3 + 3 + 9 + 9 + 9) * sizeof(float);

        ComputeShader _bodyContacts;
        int _clearBodyContactHeads, _buildBodyContactLists, _accumulateBodyContacts;
        GraphicsBuffer _bodyContactHead, _bodyContactNext, _bodyForces;

        void CreateBodyContacts()
        {
            var bodies = _model.BodyCount;
            if (bodies == 0) return;
            _bodyForces = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodies, WrenchStride);
            _buffers.Add(_bodyForces);
            if (_model.ParticleCount == 0) return;

            _bodyContacts = Resources.Load<ComputeShader>("Vbd/BodyContacts");
            _clearBodyContactHeads = _bodyContacts.FindKernel("ClearBodyContactHeads");
            _buildBodyContactLists = _bodyContacts.FindKernel("BuildBodyContactLists");
            _accumulateBodyContacts = _bodyContacts.FindKernel("AccumulateBodyParticleContacts");
            _bodyContactHead = Buffer<int>(bodies);

            var build = _buildBodyContactLists;
            var accumulate = _accumulateBodyContacts;
            _bodyContacts.SetBuffer(_clearBodyContactHeads, BodyContactId.BodyContactHead, _bodyContactHead);
            _bodyContacts.SetBuffer(build, BodyContactId.BodyContactHead, _bodyContactHead);
            _bodyContacts.SetBuffer(build, BodyContactId.ShapeBody, Upload(_model.ShapeBody));
            _bodyContacts.SetBuffer(build, BodyContactId.BodyInvMass, _bodyInvMassEffective);
            _bodyContacts.SetBuffer(accumulate, BodyContactId.BodyContactHead, _bodyContactHead);
            _bodyContacts.SetBuffer(accumulate, BodyContactId.ParticleQPrev, _particleQPrev);
            _bodyContacts.SetBuffer(accumulate, BodyContactId.ParticleRadius, Upload(_model.ParticleRadius));
            _bodyContacts.SetBuffer(accumulate, BodyContactId.ShapeMargin, Upload(_model.ShapeMargin));
            _bodyContacts.SetBuffer(accumulate, BodyContactId.BodyQPrev, _bodyQPrev);
            _bodyContacts.SetBuffer(accumulate, BodyContactId.BodyCom, Upload(_model.BodyCom));
            _bodyContacts.SetBuffer(accumulate, BodyContactId.ColorBodies, Upload(_model.BodyColorBodies));
            _bodyContacts.SetBuffer(accumulate, BodyContactId.BodyForces, _bodyForces);
        }

        // After InitializeContacts: lists each dynamic body's contacts once per step.
        void InitializeBodyContacts(CommandBuffer cmd, State state, Contacts contacts)
        {
            if (_bodyContacts == null || contacts == null) return;
            if (_bodyContactNext == null || _bodyContactNext.count < contacts.SoftContactMax)
                _bodyContactNext = Buffer<int>(contacts.SoftContactMax);

            var build = _buildBodyContactLists;
            var accumulate = _accumulateBodyContacts;
            cmd.SetComputeIntParam(_bodyContacts, BodyContactId.BodyCount, _model.BodyCount);
            cmd.SetComputeIntParam(_bodyContacts, BodyContactId.ContactMax, contacts.SoftContactMax);
            cmd.SetComputeFloatParam(_bodyContacts, BodyContactId.FrictionEpsilon, FrictionEpsilon);
            cmd.SetComputeBufferParam(_bodyContacts, build, BodyContactId.BodyContactNext, _bodyContactNext);
            cmd.SetComputeBufferParam(_bodyContacts, build, BodyContactId.ContactCount, contacts.SoftContactCount);
            cmd.SetComputeBufferParam(_bodyContacts, build, BodyContactId.ContactShape, contacts.SoftContactShape);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.BodyContactNext, _bodyContactNext);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactParticle, contacts.SoftContactParticle);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactShape, contacts.SoftContactShape);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactBodyPos, contacts.SoftContactBodyPos);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactBodyVel, contacts.SoftContactBodyVel);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactNormal, contacts.SoftContactNormal);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ContactMaterial, _contactMaterial);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.ParticleQ, state.ParticleQ);
            cmd.SetComputeBufferParam(_bodyContacts, accumulate, BodyContactId.BodyQ, state.BodyQ);

            cmd.DispatchCompute(_bodyContacts, _clearBodyContactHeads, Groups(_model.BodyCount), 1, 1);
            cmd.DispatchCompute(_bodyContacts, build, Groups(contacts.SoftContactMax), 1, 1);
        }

        // Newton's order: body-particle contacts write the color's wrenches, body-body contacts add to them (or write
        // them when no particle contacts were gathered). Particle-body attachments, first in Newton, are not ported.
        partial void AccumulateBodyForces(CommandBuffer cmd, Contacts contacts, int colorStart, int colorSize, float dt)
        {
            var particles = _bodyContacts != null && contacts != null;
            var bodies = _bodyBody != null && contacts != null;
            // Without contacts reaching bodies the solve skips the gathered forces.
            cmd.SetComputeBufferParam(_rigid, _solveRigid, BodyContactId.BodyForces, _bodyForces);
            cmd.SetComputeIntParam(_rigid, BodyContactId.HasBodyForces, particles || bodies ? 1 : 0);
            if (particles)
            {
                cmd.SetComputeIntParam(_bodyContacts, BodyContactId.ColorStart, colorStart);
                cmd.SetComputeIntParam(_bodyContacts, BodyContactId.ColorSize, colorSize);
                cmd.SetComputeFloatParam(_bodyContacts, BodyContactId.Dt, dt);
                cmd.DispatchCompute(_bodyContacts, _accumulateBodyContacts, Groups(colorSize), 1, 1);
            }
            if (bodies) AccumulateBodyBodyContacts(cmd, colorStart, colorSize, particles);
        }
    }
}
