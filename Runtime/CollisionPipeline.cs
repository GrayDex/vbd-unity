using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Newton's CollisionPipeline with its default explicit broad phase: particle-shape and shape-shape contacts,
    // every step, before SolverVbd.Step.
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

        static class RigidId
        {
            public static readonly int ShapeGap = Shader.PropertyToID("_ShapeGap");
            public static readonly int ShapePairs = Shader.PropertyToID("_ShapePairs");
            public static readonly int AabbLower = Shader.PropertyToID("_AabbLower");
            public static readonly int AabbUpper = Shader.PropertyToID("_AabbUpper");
            public static readonly int GeomData = Shader.PropertyToID("_GeomData");
            public static readonly int GeomXform = Shader.PropertyToID("_GeomXform");
            public static readonly int CandidatePairs = Shader.PropertyToID("_CandidatePairs");
            public static readonly int ConvexPairs = Shader.PropertyToID("_ConvexPairs");
            public static readonly int Counters = Shader.PropertyToID("_Counters");
            public static readonly int ContactShape0 = Shader.PropertyToID("_ContactShape0");
            public static readonly int ContactShape1 = Shader.PropertyToID("_ContactShape1");
            public static readonly int ContactPoint0 = Shader.PropertyToID("_ContactPoint0");
            public static readonly int ContactPoint1 = Shader.PropertyToID("_ContactPoint1");
            public static readonly int ContactOffset0 = Shader.PropertyToID("_ContactOffset0");
            public static readonly int ContactOffset1 = Shader.PropertyToID("_ContactOffset1");
            public static readonly int ContactNormal = Shader.PropertyToID("_ContactNormal");
            public static readonly int ContactMargin0 = Shader.PropertyToID("_ContactMargin0");
            public static readonly int ContactMargin1 = Shader.PropertyToID("_ContactMargin1");
            public static readonly int Records = Shader.PropertyToID("_Records");
            public static readonly int PairCount = Shader.PropertyToID("_PairCount");
        }

        // Newton's _RIGID_CONTACTS_PER_PRIMITIVE_PAIR and _RIGID_CONTACT_MIN_CAPACITY.
        const int RigidContactsPerPair = 5;
        const int RigidContactMinCapacity = 1000;

        // Detection-only distance added to particle radius and shape margin, as Newton's soft_contact_gap.
        public float SoftContactGap = 0.01f;

        readonly Model _model;
        readonly ComputeShader _collide;
        readonly int _clearCount, _createSoftContacts;
        readonly List<GraphicsBuffer> _buffers = new();

        readonly ComputeShader _rigid;
        readonly int _shapeAabbs, _broadPhase, _narrowPrimitive, _narrowConvex, _unpackA, _unpackB;
        readonly int[] _rigidKernels;
        GraphicsBuffer _records;
        readonly int _pairCount;

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

            var pairs = ShapeContactPairs(model);
            _pairCount = pairs.Length;
            if (_pairCount == 0) return;
            if (model.ShapeGap.Length != model.ShapeCount)
                throw new ArgumentException("Model.ShapeGap must hold a gap per shape for shape-shape contacts.");
            _rigid = Resources.Load<ComputeShader>("Vbd/CollideRigid");
            _shapeAabbs = _rigid.FindKernel("ComputeShapeAabbs");
            _broadPhase = _rigid.FindKernel("BroadPhaseExplicit");
            _narrowPrimitive = _rigid.FindKernel("NarrowPhasePrimitive");
            _narrowConvex = _rigid.FindKernel("NarrowPhaseConvex");
            _unpackA = _rigid.FindKernel("UnpackContactsA");
            _unpackB = _rigid.FindKernel("UnpackContactsB");
            _rigidKernels = new[] { _shapeAabbs, _broadPhase, _narrowPrimitive, _narrowConvex, _unpackA, _unpackB };
            var n = model.ShapeCount;
            var shared = new (int id, GraphicsBuffer buffer)[]
            {
                (Id.ShapeBody, Upload(model.ShapeBody)), (Id.ShapeTransform, Upload(model.ShapeTransform)),
                (Id.ShapeType, Upload(model.ShapeType)), (Id.ShapeScale, Upload(model.ShapeScale)),
                (Id.ShapeMargin, Upload(model.ShapeMargin)), (RigidId.ShapeGap, Upload(model.ShapeGap)),
                (RigidId.ShapePairs, Upload(pairs)), (RigidId.AabbLower, Buffer<float3>(n)),
                (RigidId.AabbUpper, Buffer<float3>(n)), (RigidId.GeomData, Buffer<float4>(n)),
                (RigidId.GeomXform, Buffer<Transform>(n)), (RigidId.CandidatePairs, Buffer<int2>(_pairCount)),
                (RigidId.ConvexPairs, Buffer<int2>(_pairCount)), (RigidId.Counters, Buffer<int>(2))
            };
            foreach (var kernel in _rigidKernels)
            foreach (var (id, buffer) in shared)
                _rigid.SetBuffer(kernel, id, buffer);
        }

        // Particle × shape pairs, as Newton's default soft_contact_max; Newton's neighbor estimate for shape pairs,
        // here bounded by the explicit pair count.
        public Contacts CreateContacts() => new(_model.ParticleCount * _model.ShapeCount,
            _pairCount == 0 ? 0 : Math.Max(RigidContactMinCapacity, _pairCount * RigidContactsPerPair));

        public void Collide(CommandBuffer cmd, State state, Contacts contacts)
        {
            CollideRigid(cmd, state, contacts);
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

        void CollideRigid(CommandBuffer cmd, State state, Contacts contacts)
        {
            if (_pairCount == 0)
            {
                cmd.SetBufferData(contacts.RigidContactCount, NoContacts);
                return;
            }
            cmd.SetComputeIntParam(_rigid, Id.ShapeCount, _model.ShapeCount);
            cmd.SetComputeIntParam(_rigid, RigidId.PairCount, _pairCount);
            // Newton's contact writer, packed: shapes, points, offsets, normal, margins (see ContactRecord).
            if (_records == null || _records.count < contacts.RigidContactMax)
            {
                if (_records != null) _buffers.Remove(_records);
                _records?.Dispose();
                _records = new GraphicsBuffer(GraphicsBuffer.Target.Structured, contacts.RigidContactMax, RecordStride);
                _buffers.Add(_records);
            }
            cmd.SetComputeIntParam(_rigid, Id.ContactMax, contacts.RigidContactMax);
            foreach (var k in _rigidKernels)
            {
                cmd.SetComputeBufferParam(_rigid, k, Id.BodyQ, state.BodyQ);
                cmd.SetComputeBufferParam(_rigid, k, Id.ContactCount, contacts.RigidContactCount);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactShape0, contacts.RigidContactShape0);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactShape1, contacts.RigidContactShape1);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactPoint0, contacts.RigidContactPoint0);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactPoint1, contacts.RigidContactPoint1);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactOffset0, contacts.RigidContactOffset0);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactOffset1, contacts.RigidContactOffset1);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactNormal, contacts.RigidContactNormal);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactMargin0, contacts.RigidContactMargin0);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.ContactMargin1, contacts.RigidContactMargin1);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.Records, _records);
            }
            cmd.DispatchCompute(_rigid, _shapeAabbs, (_model.ShapeCount + 63) / 64, 1, 1);
            var pairGroups = (_pairCount + 63) / 64;
            cmd.DispatchCompute(_rigid, _broadPhase, pairGroups, 1, 1);
            cmd.DispatchCompute(_rigid, _narrowPrimitive, pairGroups, 1, 1);
            cmd.DispatchCompute(_rigid, _narrowConvex, pairGroups, 1, 1);
            var contactGroups = (contacts.RigidContactMax + 63) / 64;
            cmd.DispatchCompute(_rigid, _unpackA, contactGroups, 1, 1);
            cmd.DispatchCompute(_rigid, _unpackB, contactGroups, 1, 1);
        }

        static readonly int[] NoContacts = { 0 };
        const int RecordStride = 76;

        // Newton's model.shape_contact_pairs (_ShapeContactPairs.build_pairs, one world): every pair of shapes that
        // collide with shapes, are on different bodies, have compatible groups and are not filtered.
        internal static int2[] ShapeContactPairs(Model model)
        {
            var n = model.ShapeCount;
            int Group(int s) => s < model.ShapeCollisionGroup.Length ? model.ShapeCollisionGroup[s] : 1;
            bool Active(int s) => (model.ShapeFlags[s] & Model.ShapeCollideShapes) != 0 && Group(s) != 0;
            var filtered = new HashSet<int2>(model.ShapeCollisionFilterPairs);
            var pairs = new List<int2>();
            for (var a = 0; a < n; a++)
            {
                if (!Active(a)) continue;
                for (var b = a + 1; b < n; b++)
                {
                    if (!Active(b) || model.ShapeBody[a] == model.ShapeBody[b]) continue;
                    int ga = Group(a), gb = Group(b);
                    if (ga > 0 ? ga != gb && gb >= 0 : ga == gb) continue;
                    if (!filtered.Contains(new int2(a, b))) pairs.Add(new int2(a, b));
                }
            }
            return pairs.ToArray();
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
