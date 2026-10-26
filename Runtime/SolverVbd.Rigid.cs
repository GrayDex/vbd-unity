using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Vbd
{
    // Rigid bodies and joints: Newton's _initialize_rigid_bodies, _solve_rigid_body_iteration, _finalize_rigid_bodies
    // on the compliant-ALM path (Newton's default). No contacts yet.
    public sealed partial class SolverVbd
    {
        // Newton's SolverVBD constructor arguments with its defaults; read at every step.
        public float RigidJointLinearKe = 1e5f;
        public float RigidJointAngularKe = 1e5f;
        public float RigidJointLinearKd;
        public float RigidJointAngularKd;
        public float RigidAvbdGamma = 0.999f;
        public float RigidJointAlpha;

        // Drive targets; the solver's own, filled from the model, unless the caller sets another.
        public Control Control;

        [StructLayout(LayoutKind.Sequential)]
        struct JointData
        {
            public int Type, Parent, Child, Enabled, QStart, QdStart, LinCount, AngCount;
            public Transform Xp, Xc;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct DofData
        {
            public float3 Axis;
            public float TargetKe, TargetKd, LimitLower, LimitUpper, LimitKe, LimitKd, RestAngle;
        }

        static class RigidId
        {
            public static readonly int BodyQ = Shader.PropertyToID("_BodyQ");
            public static readonly int BodyQd = Shader.PropertyToID("_BodyQd");
            public static readonly int BodyQPrev = Shader.PropertyToID("_BodyQPrev");
            public static readonly int BodyInertiaQ = Shader.PropertyToID("_BodyInertiaQ");
            public static readonly int JointState = Shader.PropertyToID("_JointState");
            public static readonly int DofState = Shader.PropertyToID("_DofState");
            public static readonly int Rebaseline = Shader.PropertyToID("_Rebaseline");
            public static readonly int BodyQRest = Shader.PropertyToID("_BodyQRest");
            public static readonly int BodyCom = Shader.PropertyToID("_BodyCom");
            public static readonly int BodyMass = Shader.PropertyToID("_BodyMass");
            public static readonly int BodyInvMass = Shader.PropertyToID("_BodyInvMass");
            public static readonly int BodyInertia = Shader.PropertyToID("_BodyInertia");
            public static readonly int BodyInvInertia = Shader.PropertyToID("_BodyInvInertia");
            public static readonly int ColorBodies = Shader.PropertyToID("_ColorBodies");
            public static readonly int AdjJoints = Shader.PropertyToID("_AdjJoints");
            public static readonly int AdjJointsOffsets = Shader.PropertyToID("_AdjJointsOffsets");
            public static readonly int Joints = Shader.PropertyToID("_Joints");
            public static readonly int Dofs = Shader.PropertyToID("_Dofs");
            public static readonly int TargetQ = Shader.PropertyToID("_TargetQ");
            public static readonly int TargetQd = Shader.PropertyToID("_TargetQd");
            public static readonly int BodyCount = Shader.PropertyToID("_BodyCount");
            public static readonly int JointCount = Shader.PropertyToID("_JointCount");
            public static readonly int ColorStart = Shader.PropertyToID("_ColorStart");
            public static readonly int ColorSize = Shader.PropertyToID("_ColorSize");
            public static readonly int Dt = Shader.PropertyToID("_Dt");
            public static readonly int Gravity = Shader.PropertyToID("_Gravity");
            public static readonly int LinearKe = Shader.PropertyToID("_LinearKe");
            public static readonly int AngularKe = Shader.PropertyToID("_AngularKe");
            public static readonly int LinearKd = Shader.PropertyToID("_LinearKd");
            public static readonly int AngularKd = Shader.PropertyToID("_AngularKd");
            public static readonly int Alpha = Shader.PropertyToID("_Alpha");
            public static readonly int Retention = Shader.PropertyToID("_Retention");
        }

        static readonly int[] RebaselineDone = { 0 };

        ComputeShader _rigid;
        int _forwardStepRigid, _stepJoint, _solveRigid, _updateDualsJoint, _updateBodyVelocity;
        int[] _rigidKernels;
        GraphicsBuffer _rebaseline;
        bool _hasJointRows;

        partial void CreateRigidBodies()
        {
            var n = _model.BodyCount;
            if (n == 0) return;
            if (_model.BodyColorCount == 0)
                throw new ArgumentException("Model.BodyColorBodies is empty: rigid bodies need a coloring, as in Newton.");
            _rigid = Resources.Load<ComputeShader>("Vbd/RigidVbd");
            _forwardStepRigid = _rigid.FindKernel("ForwardStepRigidBodies");
            _stepJoint = _rigid.FindKernel("StepJointC0LambdaRho");
            _solveRigid = _rigid.FindKernel("SolveRigidBody");
            _updateDualsJoint = _rigid.FindKernel("UpdateDualsJoint");
            _updateBodyVelocity = _rigid.FindKernel("UpdateBodyVelocity");
            _rigidKernels = new[] { _forwardStepRigid, _stepJoint, _solveRigid, _updateDualsJoint, _updateBodyVelocity };

            // Kinematic bodies get zero inverse mass and inertia, as Newton's body_inv_mass_effective.
            var invMass = new float[n];
            var inertiaRows = new float3x3[n];
            var invInertiaRows = new float3x3[n];
            for (var b = 0; b < n; b++)
            {
                var kinematic = b < _model.BodyFlags.Length && (_model.BodyFlags[b] & Model.BodyKinematic) != 0;
                invMass[b] = kinematic ? 0f : _model.BodyInvMass[b];
                inertiaRows[b] = math.transpose(_model.BodyInertia[b]);
                invInertiaRows[b] = kinematic ? float3x3.zero : math.transpose(_model.BodyInvInertia[b]);
            }
            var (adjJoints, adjOffsets) = Adjacency.BodyJoints(_model);
            _hasJointRows = adjJoints.Length > 0;
            // Newton's rebaseline mask: the first step takes the state's pose as the previous pose.
            _rebaseline = Upload(new[] { 1 });

            var bodyQRest = Upload(_model.BodyQ);
            var bodyInertiaQ = Buffer<Transform>(n);
            var bodyCom = Upload(_model.BodyCom);
            var bodyMass = Upload(_model.BodyMass);
            var bodyInvMass = Upload(invMass);
            var bodyInertia = Upload(inertiaRows);
            var bodyInvInertia = Upload(invInertiaRows);
            var colorBodies = Upload(_model.BodyColorBodies);
            var adj = Upload(adjJoints);
            var adjOff = Upload(adjOffsets);
            var rest = RestAngles();
            var joints = Upload(JointArray());
            var dofs = Upload(DofArray(rest));
            // Per joint: lambda_lin, lambda_ang, C0_lin, C0_ang, rho_lin, rho_ang, all zero as in Newton.
            var jointState = Upload(new float[14 * Math.Max(_model.JointCount, 1)], JointStateStride);
            // Per DOF: support, drive lambda, limit lambda, previous hinge angle (starts at the rest angle).
            var dofInit = new float4[Math.Max(_model.JointDofCount, 1)];
            for (var i = 0; i < rest.Length; i++) dofInit[i].w = rest[i];
            var dofState = Upload(dofInit);

            Control = new Control(_model);
            _buffers.Add(Control.JointTargetQ);
            _buffers.Add(Control.JointTargetQd);

            foreach (var k in _rigidKernels)
            {
                _rigid.SetBuffer(k, RigidId.BodyQPrev, _bodyQPrev);
                _rigid.SetBuffer(k, RigidId.BodyInertiaQ, bodyInertiaQ);
                _rigid.SetBuffer(k, RigidId.JointState, jointState);
                _rigid.SetBuffer(k, RigidId.DofState, dofState);
                _rigid.SetBuffer(k, RigidId.Rebaseline, _rebaseline);
                _rigid.SetBuffer(k, RigidId.BodyQRest, bodyQRest);
                _rigid.SetBuffer(k, RigidId.BodyCom, bodyCom);
                _rigid.SetBuffer(k, RigidId.BodyMass, bodyMass);
                _rigid.SetBuffer(k, RigidId.BodyInvMass, bodyInvMass);
                _rigid.SetBuffer(k, RigidId.BodyInertia, bodyInertia);
                _rigid.SetBuffer(k, RigidId.BodyInvInertia, bodyInvInertia);
                _rigid.SetBuffer(k, RigidId.ColorBodies, colorBodies);
                _rigid.SetBuffer(k, RigidId.AdjJoints, adj);
                _rigid.SetBuffer(k, RigidId.AdjJointsOffsets, adjOff);
                _rigid.SetBuffer(k, RigidId.Joints, joints);
                _rigid.SetBuffer(k, RigidId.Dofs, dofs);
            }
        }

        const int JointStateStride = 56;

        GraphicsBuffer Upload(float[] data, int stride)
        {
            var b = new GraphicsBuffer(GraphicsBuffer.Target.Structured, data.Length * sizeof(float) / stride, stride);
            _buffers.Add(b);
            b.SetData(data);
            return b;
        }

        JointData[] JointArray()
        {
            var m = _model;
            var a = new JointData[m.JointCount];
            for (var j = 0; j < a.Length; j++)
                a[j] = new JointData
                {
                    Type = m.JointType[j], Parent = m.JointParent[j], Child = m.JointChild[j],
                    Enabled = m.JointEnabled[j] ? 1 : 0, QStart = m.JointQStart[j], QdStart = m.JointQdStart[j],
                    LinCount = m.JointDofDim[j].x, AngCount = m.JointDofDim[j].y, Xp = m.JointXP[j], Xc = m.JointXC[j]
                };
            return a;
        }

        DofData[] DofArray(float[] rest)
        {
            var m = _model;
            var a = new DofData[m.JointDofCount];
            for (var d = 0; d < a.Length; d++)
                a[d] = new DofData
                {
                    Axis = m.JointAxis[d], TargetKe = m.JointTargetKe[d], TargetKd = m.JointTargetKd[d],
                    LimitLower = m.JointLimitLower[d], LimitUpper = m.JointLimitUpper[d],
                    LimitKe = m.JointLimitKe[d], LimitKd = m.JointLimitKd[d], RestAngle = rest[d]
                };
            return a;
        }

        // Newton's _init_joint_rest_angle: JointQ of revolute DOFs and of one-axis angular D6 DOFs; kappa is measured
        // from the rest pose, drive targets and limits in absolute coordinates.
        float[] RestAngles()
        {
            var m = _model;
            var rest = new float[m.JointDofCount];
            for (var j = 0; j < m.JointCount; j++)
            {
                if (m.JointType[j] == Model.JointRevolute)
                    rest[m.JointQdStart[j]] = m.JointQ[m.JointQStart[j]];
                else if (m.JointType[j] == Model.JointD6 && m.JointDofDim[j].y == 1)
                {
                    var lin = m.JointDofDim[j].x;
                    rest[m.JointQdStart[j] + lin] = m.JointQ[m.JointQStart[j] + lin];
                }
            }
            return rest;
        }

        partial void InitializeRigidBodies(CommandBuffer cmd, State state, float dt)
        {
            var n = _model.BodyCount;
            if (n == 0) return;
            cmd.SetComputeIntParam(_rigid, RigidId.BodyCount, n);
            cmd.SetComputeIntParam(_rigid, RigidId.JointCount, _model.JointCount);
            cmd.SetComputeFloatParam(_rigid, RigidId.Dt, dt);
            cmd.SetComputeVectorParam(_rigid, RigidId.Gravity, (Vector3)_model.Gravity);
            cmd.SetComputeFloatParam(_rigid, RigidId.LinearKe, RigidJointLinearKe);
            cmd.SetComputeFloatParam(_rigid, RigidId.AngularKe, RigidJointAngularKe);
            cmd.SetComputeFloatParam(_rigid, RigidId.LinearKd, math.max(0f, RigidJointLinearKd));
            cmd.SetComputeFloatParam(_rigid, RigidId.AngularKd, math.max(0f, RigidJointAngularKd));
            cmd.SetComputeFloatParam(_rigid, RigidId.Alpha, RigidJointAlpha);
            // Compliant ALM retains multipliers by gamma alone (Newton's _rigid_lambda_retention).
            cmd.SetComputeFloatParam(_rigid, RigidId.Retention, RigidAvbdGamma);
            foreach (var k in _rigidKernels)
            {
                cmd.SetComputeBufferParam(_rigid, k, RigidId.BodyQ, state.BodyQ);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.BodyQd, state.BodyQd);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.TargetQ, Control.JointTargetQ);
                cmd.SetComputeBufferParam(_rigid, k, RigidId.TargetQd, Control.JointTargetQd);
            }
            cmd.DispatchCompute(_rigid, _forwardStepRigid, Groups(n), 1, 1);
            if (_hasJointRows) cmd.DispatchCompute(_rigid, _stepJoint, Groups(_model.JointCount), 1, 1);
            cmd.SetBufferData(_rebaseline, RebaselineDone);
        }

        partial void SolveRigidBodyIteration(CommandBuffer cmd, State state, float dt)
        {
            if (_model.BodyCount == 0) return;
            for (var c = 0; c < _model.BodyColorCount; c++)
            {
                var start = _model.BodyColorOffsets[c];
                var size = _model.BodyColorOffsets[c + 1] - start;
                if (size == 0) continue;
                cmd.SetComputeIntParam(_rigid, RigidId.ColorStart, start);
                cmd.SetComputeIntParam(_rigid, RigidId.ColorSize, size);
                cmd.DispatchCompute(_rigid, _solveRigid, Groups(size), 1, 1);
            }
            if (_hasJointRows) cmd.DispatchCompute(_rigid, _updateDualsJoint, Groups(_model.JointCount), 1, 1);
        }

        partial void FinalizeRigidBodies(CommandBuffer cmd, State state, float dt)
        {
            if (_model.BodyCount == 0) return;
            cmd.DispatchCompute(_rigid, _updateBodyVelocity, Groups(_model.BodyCount), 1, 1);
        }
    }
}
