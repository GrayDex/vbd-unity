using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Body-particle contacts, particle side: Newton's _refresh_body_particle_contact_state and the gathered
    // contact force inside the particle solve.
    public sealed partial class SolverVbd
    {
        static class ContactId
        {
            public static readonly int ParticleRadius = Shader.PropertyToID("_ParticleRadius");
            public static readonly int ContactCount = Shader.PropertyToID("_ContactCount");
            public static readonly int ContactParticle = Shader.PropertyToID("_ContactParticle");
            public static readonly int ContactShape = Shader.PropertyToID("_ContactShape");
            public static readonly int ContactBodyPos = Shader.PropertyToID("_ContactBodyPos");
            public static readonly int ContactBodyVel = Shader.PropertyToID("_ContactBodyVel");
            public static readonly int ContactNormal = Shader.PropertyToID("_ContactNormal");
            public static readonly int ContactHead = Shader.PropertyToID("_ContactHead");
            public static readonly int ContactNext = Shader.PropertyToID("_ContactNext");
            public static readonly int ContactMaterial = Shader.PropertyToID("_ContactMaterial");
            public static readonly int ShapeBody = Shader.PropertyToID("_ShapeBody");
            public static readonly int ShapeMargin = Shader.PropertyToID("_ShapeMargin");
            public static readonly int ShapeMaterialKe = Shader.PropertyToID("_ShapeMaterialKe");
            public static readonly int ShapeMaterialKd = Shader.PropertyToID("_ShapeMaterialKd");
            public static readonly int ShapeMaterialMu = Shader.PropertyToID("_ShapeMaterialMu");
            public static readonly int BodyQ = Shader.PropertyToID("_BodyQ");
            public static readonly int BodyQPrev = Shader.PropertyToID("_BodyQPrev");
            public static readonly int ContactMax = Shader.PropertyToID("_ContactMax");
            public static readonly int SoftContactKe = Shader.PropertyToID("_SoftContactKe");
            public static readonly int SoftContactKd = Shader.PropertyToID("_SoftContactKd");
            public static readonly int SoftContactMu = Shader.PropertyToID("_SoftContactMu");
            public static readonly int FrictionEpsilon = Shader.PropertyToID("_FrictionEpsilon");
        }

        // Slip speed [m/s] below which friction ramps linearly, as Newton's friction_epsilon.
        public float FrictionEpsilon = 1e-2f;

        int _clearContactHeads, _initBodyParticleContacts;
        GraphicsBuffer _contactHead;
        GraphicsBuffer _contactNext, _contactMaterial;
        Contacts _noContacts;

        void CreateContacts()
        {
            _clearContactHeads = _particles.FindKernel("ClearContactHeads");
            _initBodyParticleContacts = _particles.FindKernel("InitBodyParticleContacts");
            _contactHead = Buffer<int>(_model.ParticleCount);
            _noContacts = new Contacts(1);

            var solve = _solveElasticity;
            _particles.SetBuffer(solve, ContactId.ParticleRadius, Upload(_model.ParticleRadius));
            _particles.SetBuffer(solve, ContactId.ShapeBody, Upload(_model.ShapeBody));
            _particles.SetBuffer(solve, ContactId.ShapeMargin, Upload(_model.ShapeMargin));
            _particles.SetBuffer(solve, ContactId.BodyQPrev, _bodyQPrev);
            var init = _initBodyParticleContacts;
            _particles.SetBuffer(init, ContactId.ShapeMaterialKe, Upload(_model.ShapeMaterialKe));
            _particles.SetBuffer(init, ContactId.ShapeMaterialKd, Upload(_model.ShapeMaterialKd));
            _particles.SetBuffer(init, ContactId.ShapeMaterialMu, Upload(_model.ShapeMaterialMu));
            foreach (var k in new[] { _clearContactHeads, init, solve })
                _particles.SetBuffer(k, ContactId.ContactHead, _contactHead);
        }

        void InitializeContacts(CommandBuffer cmd, State state, Contacts contacts)
        {
            if (_model.ParticleCount == 0) return;
            contacts ??= _noContacts;
            if (_contactNext == null || _contactNext.count < contacts.SoftContactMax)
            {
                _contactNext = Buffer<int>(contacts.SoftContactMax);
                _contactMaterial = Buffer<float3>(contacts.SoftContactMax);
            }

            cmd.SetComputeIntParam(_particles, ContactId.ContactMax, contacts.SoftContactMax);
            cmd.SetComputeFloatParam(_particles, ContactId.SoftContactKe, _model.SoftContactKe);
            cmd.SetComputeFloatParam(_particles, ContactId.SoftContactKd, _model.SoftContactKd);
            cmd.SetComputeFloatParam(_particles, ContactId.SoftContactMu, _model.SoftContactMu);
            cmd.SetComputeFloatParam(_particles, ContactId.FrictionEpsilon, FrictionEpsilon);
            foreach (var k in new[] { _initBodyParticleContacts, _solveElasticity })
            {
                cmd.SetComputeBufferParam(_particles, k, ContactId.ContactNext, _contactNext);
                cmd.SetComputeBufferParam(_particles, k, ContactId.ContactMaterial, _contactMaterial);
                cmd.SetComputeBufferParam(_particles, k, ContactId.ContactShape, contacts.SoftContactShape);
            }
            cmd.SetComputeBufferParam(_particles, _initBodyParticleContacts, ContactId.ContactCount, contacts.SoftContactCount);
            cmd.SetComputeBufferParam(_particles, _initBodyParticleContacts, ContactId.ContactParticle, contacts.SoftContactParticle);
            cmd.SetComputeBufferParam(_particles, _solveElasticity, ContactId.ContactBodyPos, contacts.SoftContactBodyPos);
            cmd.SetComputeBufferParam(_particles, _solveElasticity, ContactId.ContactBodyVel, contacts.SoftContactBodyVel);
            cmd.SetComputeBufferParam(_particles, _solveElasticity, ContactId.ContactNormal, contacts.SoftContactNormal);
            cmd.SetComputeBufferParam(_particles, _solveElasticity, ContactId.BodyQ, state.BodyQ);

            cmd.DispatchCompute(_particles, _clearContactHeads, Groups(_model.ParticleCount), 1, 1);
            if (contacts == _noContacts) return;
            cmd.DispatchCompute(_particles, _initBodyParticleContacts, Groups(contacts.SoftContactMax), 1, 1);
        }
    }
}
