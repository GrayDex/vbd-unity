using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Body-body contacts on Newton's compliant-ALM path: per step the contact state on the collide-frame poses,
    // per body color the contact wrenches (after body-particle contacts, through AccumulateBodyForces), after each
    // rigid iteration the dual update. Newton's defaults: rigid_contact_history off, so multipliers start at zero
    // every step; the legacy hard/soft contact switch and penalty ramp have no effect on this path and are left out.
    public sealed partial class SolverVbd
    {
        // Newton's rigid_contact_alpha: C0 stabilization of contacts, 0 on compliant ALM.
        public float RigidContactAlpha;

        static class BodyBodyId
        {
            public static readonly int ContactCount = Shader.PropertyToID("_ContactCount");
            public static readonly int ContactShape0 = Shader.PropertyToID("_ContactShape0");
            public static readonly int ContactShape1 = Shader.PropertyToID("_ContactShape1");
            public static readonly int ContactPoint0 = Shader.PropertyToID("_ContactPoint0");
            public static readonly int ContactPoint1 = Shader.PropertyToID("_ContactPoint1");
            public static readonly int ContactOffset0 = Shader.PropertyToID("_ContactOffset0");
            public static readonly int ContactOffset1 = Shader.PropertyToID("_ContactOffset1");
            public static readonly int ContactNormal = Shader.PropertyToID("_ContactNormal");
            public static readonly int ContactMargin0 = Shader.PropertyToID("_ContactMargin0");
            public static readonly int ContactMargin1 = Shader.PropertyToID("_ContactMargin1");
            public static readonly int ShapeBody = Shader.PropertyToID("_ShapeBody");
            public static readonly int ShapeMaterialKe = Shader.PropertyToID("_ShapeMaterialKe");
            public static readonly int ShapeMaterialKd = Shader.PropertyToID("_ShapeMaterialKd");
            public static readonly int ShapeMaterialMu = Shader.PropertyToID("_ShapeMaterialMu");
            public static readonly int BodyQ = Shader.PropertyToID("_BodyQ");
            public static readonly int BodyQPrev = Shader.PropertyToID("_BodyQPrev");
            public static readonly int BodyCom = Shader.PropertyToID("_BodyCom");
            public static readonly int BodyInvMass = Shader.PropertyToID("_BodyInvMass");
            public static readonly int BodyInvInertia = Shader.PropertyToID("_BodyInvInertia");
            public static readonly int BodyStructural = Shader.PropertyToID("_BodyStructural");
            public static readonly int ColorBodies = Shader.PropertyToID("_ColorBodies");
            public static readonly int BodyContactHead = Shader.PropertyToID("_BodyContactHead");
            public static readonly int BodyContactNext = Shader.PropertyToID("_BodyContactNext");
            public static readonly int ContactMaterial = Shader.PropertyToID("_ContactMaterial");
            public static readonly int ContactRho = Shader.PropertyToID("_ContactRho");
            public static readonly int ContactLambda = Shader.PropertyToID("_ContactLambda");
            public static readonly int ContactC0 = Shader.PropertyToID("_ContactC0");
            public static readonly int BodyForces = Shader.PropertyToID("_BodyForces");
            public static readonly int BodyCount = Shader.PropertyToID("_BodyCount");
            public static readonly int ContactMax = Shader.PropertyToID("_ContactMax");
            public static readonly int ColorStart = Shader.PropertyToID("_ColorStart");
            public static readonly int ColorSize = Shader.PropertyToID("_ColorSize");
            public static readonly int AddToBodyForces = Shader.PropertyToID("_AddToBodyForces");
            public static readonly int Dt = Shader.PropertyToID("_Dt");
            public static readonly int FrictionEpsilon = Shader.PropertyToID("_FrictionEpsilon");
            public static readonly int StructuralKe = Shader.PropertyToID("_StructuralKe");
            public static readonly int Alpha = Shader.PropertyToID("_Alpha");
        }

        ComputeShader _bodyBody;
        int _clearBodyBodyHeads, _initBodyBody, _accumulateBodyBody, _updateDualsBodyBody;
        int[] _bodyBodyKernels;
        GraphicsBuffer _bodyBodyHead, _bodyBodyNext, _bodyBodyMaterial, _bodyBodyRho, _bodyBodyLambda, _bodyBodyC0;

        void CreateBodyBodyContacts()
        {
            var n = _model.BodyCount;
            if (n == 0 || _model.ShapeCount == 0) return;
            _bodyBody = Resources.Load<ComputeShader>("Vbd/BodyBodyContacts");
            _clearBodyBodyHeads = _bodyBody.FindKernel("ClearBodyBodyHeads");
            _initBodyBody = _bodyBody.FindKernel("InitBodyBodyContacts");
            _accumulateBodyBody = _bodyBody.FindKernel("AccumulateBodyBodyContacts");
            _updateDualsBodyBody = _bodyBody.FindKernel("UpdateDualsBodyBodyContacts");
            _bodyBodyKernels = new[] { _clearBodyBodyHeads, _initBodyBody, _accumulateBodyBody, _updateDualsBodyBody };

            // Kinematic bodies get zero inverse inertia, as Newton's body_inv_inertia_effective.
            var invInertiaRows = new float3x3[n];
            for (var b = 0; b < n; b++)
            {
                var kinematic = b < _model.BodyFlags.Length && (_model.BodyFlags[b] & Model.BodyKinematic) != 0;
                invInertiaRows[b] = kinematic ? float3x3.zero : math.transpose(_model.BodyInvInertia[b]);
            }
            _bodyBodyHead = Buffer<int>(n);
            var shared = new (int id, GraphicsBuffer buffer)[]
            {
                (BodyBodyId.ShapeBody, Upload(_model.ShapeBody)),
                (BodyBodyId.ShapeMaterialKe, Upload(_model.ShapeMaterialKe)),
                (BodyBodyId.ShapeMaterialKd, Upload(_model.ShapeMaterialKd)),
                (BodyBodyId.ShapeMaterialMu, Upload(_model.ShapeMaterialMu)),
                (BodyBodyId.BodyQPrev, _bodyQPrev), (BodyBodyId.BodyCom, Upload(_model.BodyCom)),
                (BodyBodyId.BodyInvMass, _bodyInvMassEffective), (BodyBodyId.BodyInvInertia, Upload(invInertiaRows)),
                (BodyBodyId.BodyStructural, Upload(StructuralBodies(_model))),
                (BodyBodyId.ColorBodies, Upload(_model.BodyColorBodies)), (BodyBodyId.BodyContactHead, _bodyBodyHead),
                (BodyBodyId.BodyForces, _bodyForces)
            };
            foreach (var k in _bodyBodyKernels)
            foreach (var (id, buffer) in shared)
                _bodyBody.SetBuffer(k, id, buffer);
        }

        // Newton's body_structural_k is the largest structural linear stiffness of the enabled joints on a body;
        // every such row has rigid_joint_linear_ke, so a flag per body carries it. FREE joints and D6 joints with
        // three free linear axes have no structural linear row.
        static int[] StructuralBodies(Model m)
        {
            var flags = new int[m.BodyCount];
            for (var j = 0; j < m.JointCount; j++)
            {
                if (!m.JointEnabled[j] || m.JointType[j] == Model.JointFree) continue;
                if (m.JointType[j] == Model.JointD6 && m.JointDofDim[j].x >= 3) continue;
                if (m.JointParent[j] >= 0) flags[m.JointParent[j]] = 1;
                if (m.JointChild[j] >= 0) flags[m.JointChild[j]] = 1;
            }
            return flags;
        }

        // Before the forward step: lists, materials, C0 and rho on the poses the contacts were found at.
        void InitializeBodyBodyContacts(CommandBuffer cmd, State state, Contacts contacts, float dt)
        {
            if (_bodyBody == null || contacts == null) return;
            var max = contacts.RigidContactMax;
            if (_bodyBodyMaterial == null || _bodyBodyMaterial.count < max)
            {
                _bodyBodyNext = Buffer<int>(2 * max);
                _bodyBodyMaterial = Buffer<float3>(max);
                _bodyBodyRho = Buffer<float2>(max);
                _bodyBodyLambda = Buffer<float3>(max);
                _bodyBodyC0 = Buffer<float3>(max);
            }
            cmd.SetComputeIntParam(_bodyBody, BodyBodyId.BodyCount, _model.BodyCount);
            cmd.SetComputeIntParam(_bodyBody, BodyBodyId.ContactMax, max);
            cmd.SetComputeFloatParam(_bodyBody, BodyBodyId.Dt, dt);
            cmd.SetComputeFloatParam(_bodyBody, BodyBodyId.FrictionEpsilon, FrictionEpsilon);
            cmd.SetComputeFloatParam(_bodyBody, BodyBodyId.StructuralKe, RigidJointLinearKe);
            cmd.SetComputeFloatParam(_bodyBody, BodyBodyId.Alpha, RigidContactAlpha);
            foreach (var k in _bodyBodyKernels)
            {
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.BodyQ, state.BodyQ);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactCount, contacts.RigidContactCount);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactShape0, contacts.RigidContactShape0);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactShape1, contacts.RigidContactShape1);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactPoint0, contacts.RigidContactPoint0);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactPoint1, contacts.RigidContactPoint1);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactOffset0, contacts.RigidContactOffset0);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactOffset1, contacts.RigidContactOffset1);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactNormal, contacts.RigidContactNormal);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactMargin0, contacts.RigidContactMargin0);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactMargin1, contacts.RigidContactMargin1);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.BodyContactNext, _bodyBodyNext);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactMaterial, _bodyBodyMaterial);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactRho, _bodyBodyRho);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactLambda, _bodyBodyLambda);
                cmd.SetComputeBufferParam(_bodyBody, k, BodyBodyId.ContactC0, _bodyBodyC0);
            }
            cmd.DispatchCompute(_bodyBody, _clearBodyBodyHeads, Groups(_model.BodyCount), 1, 1);
            cmd.DispatchCompute(_bodyBody, _initBodyBody, Groups(max), 1, 1);
        }

        // Adds to the body-particle wrench of the color when that was gathered, else replaces last iteration's.
        void AccumulateBodyBodyContacts(CommandBuffer cmd, int colorStart, int colorSize, bool add)
        {
            cmd.SetComputeIntParam(_bodyBody, BodyBodyId.ColorStart, colorStart);
            cmd.SetComputeIntParam(_bodyBody, BodyBodyId.ColorSize, colorSize);
            cmd.SetComputeIntParam(_bodyBody, BodyBodyId.AddToBodyForces, add ? 1 : 0);
            cmd.DispatchCompute(_bodyBody, _accumulateBodyBody, Groups(colorSize), 1, 1);
        }

        void UpdateBodyBodyContactDuals(CommandBuffer cmd, Contacts contacts)
        {
            if (_bodyBody == null || contacts == null) return;
            cmd.DispatchCompute(_bodyBody, _updateDualsBodyBody, Groups(contacts.RigidContactMax), 1, 1);
        }
    }
}
