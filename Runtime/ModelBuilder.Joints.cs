using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Vbd
{
    // Joints and articulations: newton.ModelBuilder's add_joint and its typed forms for the joint types SolverVbd
    // solves (prismatic, revolute, ball, fixed, free, D6), add_articulation, add_body, and the joint part of finalize.
    // Left out because SolverVbd does not read them: armature, passive damping, friction, effort and velocity limits,
    // actuator modes, labels, mimic joints.
    public sealed partial class ModelBuilder
    {
        // newton.ModelBuilder.JointDofConfig: one axis, its limits and drive, with Newton's defaults.
        public sealed class JointDofConfig
        {
            public readonly float3 Axis;
            public readonly float LimitLower, LimitUpper, LimitKe, LimitKd;
            public readonly float TargetPos, TargetVel, TargetKe, TargetKd;

            // The axis is normalized; a target outside the limits moves to their midpoint, as Newton's.
            public JointDofConfig(float3? axis = null, float limitLower = -Model.MaxVal, float limitUpper = Model.MaxVal,
                float limitKe = 1e4f, float limitKd = 1e1f, float targetPos = 0f, float targetVel = 0f,
                float targetKe = 0f, float targetKd = 0f)
            {
                Axis = math.normalize(axis ?? new float3(1f, 0f, 0f));
                LimitLower = limitLower;
                LimitUpper = limitUpper;
                LimitKe = limitKe;
                LimitKd = limitKd;
                TargetPos = targetPos > limitUpper || targetPos < limitLower ? 0.5f * (limitLower + limitUpper) : targetPos;
                TargetVel = targetVel;
                TargetKe = targetKe;
                TargetKd = targetKd;
            }

            public static JointDofConfig CreateUnlimited(float3 axis) => new(axis, limitKe: 0f, limitKd: 0f);
        }

        public JointDofConfig DefaultJointCfg = new();

        public readonly List<int> JointType = new();
        public readonly List<int> JointParent = new();
        public readonly List<int> JointChild = new();
        public readonly List<Transform> JointXP = new();
        public readonly List<Transform> JointXC = new();
        public readonly List<int2> JointDofDim = new();
        public readonly List<bool> JointEnabled = new();
        public readonly List<int> JointArticulation = new();
        public readonly List<int> JointQStart = new();
        public readonly List<int> JointQdStart = new();
        // Per coordinate.
        public readonly List<float> JointQ = new();
        public readonly List<float> JointTargetQ = new();
        // Per DOF.
        public readonly List<float> JointQd = new();
        public readonly List<float> JointF = new();
        public readonly List<float3> JointAxis = new();
        public readonly List<float> JointTargetQd = new();
        public readonly List<float> JointTargetKe = new();
        public readonly List<float> JointTargetKd = new();
        public readonly List<float> JointLimitLower = new();
        public readonly List<float> JointLimitUpper = new();
        public readonly List<float> JointLimitKe = new();
        public readonly List<float> JointLimitKd = new();

        public readonly List<int> ArticulationStart = new();
        public readonly List<int> ArticulationEnd = new();

        // Per child body: (parent, joint); per parent body: (child, joint). Newton's joint_parents / joint_children.
        readonly Dictionary<int, List<int2>> _jointParents = new();
        readonly Dictionary<int, List<int2>> _jointChildren = new();
        // Newton's joint_collision_filter_parent, per joint.
        readonly List<bool> _jointCollisionFilterParent = new();

        public int JointCount => JointType.Count;
        public int JointDofCount => JointAxis.Count;
        public int JointCoordCount => JointQ.Count;
        public int ArticulationCount => ArticulationStart.Count;

        // add_body: a free-floating body, i.e. AddLink, a FREE joint to the world and an articulation of that joint.
        public int AddBody(Transform? xform = null, float3? com = null, float3x3? inertia = null, float mass = 0f,
            bool lockInertia = false, bool isKinematic = false)
        {
            var body = AddLink(xform, com, inertia, mass, lockInertia, isKinematic);
            AddArticulation(new[] { AddJointFree(body) });
            return body;
        }

        // add_joint: parent -1 is the world; axes are in the parent's joint frame, linear ones first.
        // collisionFilterParent: no shape-shape contacts between parent and child; null is Newton's default, true but
        // for non-fixed joints to the world.
        public int AddJoint(int jointType, int parent, int child, IReadOnlyList<JointDofConfig> linearAxes = null,
            IReadOnlyList<JointDofConfig> angularAxes = null, Transform? parentXform = null,
            Transform? childXform = null, bool enabled = true, bool? collisionFilterParent = null)
        {
            linearAxes ??= Array.Empty<JointDofConfig>();
            angularAxes ??= Array.Empty<JointDofConfig>();
            if (jointType is not (Model.JointPrismatic or Model.JointRevolute or Model.JointBall or Model.JointFixed
                or Model.JointFree or Model.JointD6))
                throw new ArgumentException($"Joint type {jointType} is not supported by SolverVbd");
            if (parent != -1 && (parent < 0 || parent >= BodyCount))
                throw new ArgumentException($"Parent body index {parent} is out of range");
            if (child < 0 || child >= BodyCount) throw new ArgumentException($"Child body index {child} is out of range");
            WarnParallelJoint(jointType, parent, child);

            var joint = JointCount;
            JointType.Add(jointType);
            JointParent.Add(parent);
            JointChild.Add(child);
            Append(_jointParents, child, new int2(parent, joint));
            Append(_jointChildren, parent, new int2(child, joint));
            JointXP.Add(parentXform ?? Identity);
            JointXC.Add(childXform ?? Identity);
            JointDofDim.Add(new int2(linearAxes.Count, angularAxes.Count));
            JointEnabled.Add(enabled);
            JointArticulation.Add(-1);
            var filterParent = collisionFilterParent ?? (parent != -1 || jointType == Model.JointFixed);
            _jointCollisionFilterParent.Add(filterParent);
            if (filterParent)
                foreach (var childShape in BodyShapes(child))
                    if ((ShapeFlagsList[childShape] & ShapeFlags.CollideShapes) != 0) FilterAgainstBody(childShape, parent);
            foreach (var dof in linearAxes) AddAxis(dof);
            foreach (var dof in angularAxes) AddAxis(dof);

            var (dofCount, coordCount) = DofCount(jointType, linearAxes.Count + angularAxes.Count);
            JointQStart.Add(JointCoordCount);
            JointQdStart.Add(JointF.Count);
            var targetStart = JointTargetQ.Count;
            for (var i = 0; i < coordCount; i++)
            {
                JointQ.Add(0f);
                JointTargetQ.Add(0f);
            }
            for (var i = 0; i < dofCount; i++)
            {
                JointQd.Add(0f);
                JointF.Add(0f);
            }
            // A valid quaternion for the angular coordinates of BALL and FREE; the targets compose from per-axis angles.
            if (jointType is Model.JointBall or Model.JointFree)
            {
                JointQ[^1] = 1f;
                var quatStart = targetStart;
                if (jointType == Model.JointFree)
                {
                    for (var i = 0; i < linearAxes.Count; i++) JointTargetQ[targetStart + i] = linearAxes[i].TargetPos;
                    quatStart += 3;
                }
                var q = QuatFromAxisTargets(angularAxes[0].TargetPos, angularAxes[1].TargetPos, angularAxes[2].TargetPos);
                for (var k = 0; k < 4; k++) JointTargetQ[quatStart + k] = q[k];
            }
            else if (jointType != Model.JointFixed)
            {
                for (var i = 0; i < linearAxes.Count; i++) JointTargetQ[targetStart + i] = linearAxes[i].TargetPos;
                for (var i = 0; i < angularAxes.Count; i++)
                    JointTargetQ[targetStart + linearAxes.Count + i] = angularAxes[i].TargetPos;
            }
            return joint;
        }

        void AddAxis(JointDofConfig dof)
        {
            JointAxis.Add(dof.Axis);
            JointTargetQd.Add(dof.TargetVel);
            JointTargetKe.Add(dof.TargetKe);
            JointTargetKd.Add(dof.TargetKd);
            JointLimitKe.Add(dof.LimitKe);
            JointLimitKd.Add(dof.LimitKd);
            JointLimitLower.Add(math.isfinite(dof.LimitLower) ? dof.LimitLower : -Model.MaxVal);
            JointLimitUpper.Add(math.isfinite(dof.LimitUpper) ? dof.LimitUpper : Model.MaxVal);
        }

        // JointType.dof_count: (velocity DOFs, position coordinates).
        static (int dofs, int coords) DofCount(int jointType, int axes) => jointType switch
        {
            Model.JointBall => (3, 4),
            Model.JointFree => (6, 7),
            Model.JointFixed => (0, 0),
            _ => (axes, axes)
        };

        static void Append(Dictionary<int, List<int2>> map, int key, int2 value)
        {
            if (!map.TryGetValue(key, out var list)) map[key] = list = new List<int2>();
            list.Add(value);
        }

        void WarnParallelJoint(int jointType, int parent, int child)
        {
            var parallel = false;
            var parallelFree = false;
            foreach (var map in new[] { _jointParents, _jointChildren })
            {
                if (!map.TryGetValue(child, out var list)) continue;
                foreach (var e in list)
                {
                    if (e.x != parent) continue;
                    parallel = true;
                    parallelFree |= jointType == Model.JointFree || JointType[e.y] == Model.JointFree;
                }
            }
            if (parallelFree)
                Debug.LogWarning($"A FREE joint parallel to another joint between bodies {parent} and {child} is " +
                                 "inconsistent; use AddLink with the joint instead of AddBody.");
            else if (parallel)
                Debug.LogWarning($"Parallel joints between bodies {parent} and {child} have undefined semantics.");
        }

        // _quat_from_axis_targets: extrinsic ZYX composition of per-axis angles, (x, y, z, w).
        static float4 QuatFromAxisTargets(double tx, double ty, double tz)
        {
            double cx = math.cos(tx * 0.5), sx = math.sin(tx * 0.5);
            double cy = math.cos(ty * 0.5), sy = math.sin(ty * 0.5);
            double cz = math.cos(tz * 0.5), sz = math.sin(tz * 0.5);
            return (float4)new double4(sx * cy * cz + cx * sy * sz, cx * sy * cz - sx * cy * sz,
                cx * cy * sz + sx * sy * cz, cx * cy * cz - sx * sy * sz);
        }

        public int AddJointRevolute(int parent, int child, Transform? parentXform = null, Transform? childXform = null,
            float3? axis = null, float? targetPos = null, float? targetVel = null, float? targetKe = null,
            float? targetKd = null, float? limitLower = null, float? limitUpper = null, float? limitKe = null,
            float? limitKd = null, bool enabled = true) =>
            AddJointRevolute(parent, child, Dof(axis ?? DefaultJointCfg.Axis, targetPos, targetVel, targetKe, targetKd,
                limitLower, limitUpper, limitKe, limitKd), parentXform, childXform, enabled);

        public int AddJointRevolute(int parent, int child, JointDofConfig axis, Transform? parentXform = null,
            Transform? childXform = null, bool enabled = true) =>
            AddJoint(Model.JointRevolute, parent, child, null, new[] { axis }, parentXform, childXform, enabled);

        public int AddJointPrismatic(int parent, int child, Transform? parentXform = null, Transform? childXform = null,
            float3? axis = null, float? targetPos = null, float? targetVel = null, float? targetKe = null,
            float? targetKd = null, float? limitLower = null, float? limitUpper = null, float? limitKe = null,
            float? limitKd = null, bool enabled = true) =>
            AddJointPrismatic(parent, child, Dof(axis ?? new float3(1f, 0f, 0f), targetPos, targetVel, targetKe,
                targetKd, limitLower, limitUpper, limitKe, limitKd), parentXform, childXform, enabled);

        public int AddJointPrismatic(int parent, int child, JointDofConfig axis, Transform? parentXform = null,
            Transform? childXform = null, bool enabled = true) =>
            AddJoint(Model.JointPrismatic, parent, child, new[] { axis }, null, parentXform, childXform, enabled);

        // Axes x, y, z of the parent's joint frame with default limits; Newton's VBD does not drive ball joints.
        public int AddJointBall(int parent, int child, Transform? parentXform = null, Transform? childXform = null,
            bool enabled = true) =>
            AddJoint(Model.JointBall, parent, child, null,
                new[] { new JointDofConfig(new float3(1f, 0f, 0f)), new JointDofConfig(new float3(0f, 1f, 0f)),
                    new JointDofConfig(new float3(0f, 0f, 1f)) }, parentXform, childXform, enabled);

        public int AddJointFixed(int parent, int child, Transform? parentXform = null, Transform? childXform = null,
            bool enabled = true) =>
            AddJoint(Model.JointFixed, parent, child, null, null, parentXform, childXform, enabled);

        // add_joint_free: the coordinates start at the child's current pose relative to the parent's joint frame.
        public int AddJointFree(int child, Transform? parentXform = null, Transform? childXform = null, int parent = -1,
            bool enabled = true)
        {
            var x = JointDofConfig.CreateUnlimited(new float3(1f, 0f, 0f));
            var y = JointDofConfig.CreateUnlimited(new float3(0f, 1f, 0f));
            var z = JointDofConfig.CreateUnlimited(new float3(0f, 0f, 1f));
            var joint = AddJoint(Model.JointFree, parent, child, new[] { x, y, z }, new[] { x, y, z }, parentXform,
                childXform, enabled);
            InitJointQFromBodyPoses(joint, parent, child);
            return joint;
        }

        public int AddJointD6(int parent, int child, IReadOnlyList<JointDofConfig> linearAxes = null,
            IReadOnlyList<JointDofConfig> angularAxes = null, Transform? parentXform = null,
            Transform? childXform = null, bool enabled = true) =>
            AddJoint(Model.JointD6, parent, child, linearAxes, angularAxes, parentXform, childXform, enabled);

        JointDofConfig Dof(float3 axis, float? targetPos, float? targetVel, float? targetKe, float? targetKd,
            float? limitLower, float? limitUpper, float? limitKe, float? limitKd)
        {
            var d = DefaultJointCfg;
            return new JointDofConfig(axis, limitLower ?? d.LimitLower, limitUpper ?? d.LimitUpper, limitKe ?? d.LimitKe,
                limitKd ?? d.LimitKd, targetPos ?? d.TargetPos, targetVel ?? d.TargetVel, targetKe ?? d.TargetKe,
                targetKd ?? d.TargetKd);
        }

        // _init_joint_q_from_body_poses: inverse(parent body * parent anchor) * child body * child anchor.
        void InitJointQFromBodyPoses(int joint, int parent, int child)
        {
            var parentPose = parent == -1 ? Identity : BodyQ[parent];
            var anchor = Mul(parentPose, JointXP[joint]);
            var q = Mul(Mul(Inverse(anchor), BodyQ[child]), JointXC[joint]);
            var s = JointQStart[joint];
            for (var k = 0; k < 3; k++) JointQ[s + k] = q.P[k];
            for (var k = 0; k < 4; k++) JointQ[s + 3 + k] = q.Q.value[k];
        }

        // Warp's transform product and inverse, term by term as on Newton's host.
        static float3 Rotate(float4 q, float3 v) =>
            v * (2f * q.w * q.w - 1f) + math.cross(q.xyz, v) * q.w * 2f + q.xyz * math.dot(q.xyz, v) * 2f;

        static Transform Mul(Transform a, Transform b)
        {
            float4 p = a.Q.value, r = b.Q.value;
            var q = new float4(p.w * r.xyz + r.w * p.xyz + math.cross(p.xyz, r.xyz), p.w * r.w - math.dot(p.xyz, r.xyz));
            return new Transform(a.P + Rotate(p, b.P), new quaternion(q));
        }

        static Transform Inverse(Transform a)
        {
            var q = new float4(-a.Q.value.xyz, a.Q.value.w);
            return new Transform(-Rotate(q, a.P), new quaternion(q));
        }

        // add_articulation: joints contiguous and ascending, each in no other articulation, a tree (one parent per
        // child), kinematic bodies only on joints to the world.
        public void AddArticulation(IReadOnlyList<int> joints)
        {
            if (joints == null || joints.Count == 0) throw new ArgumentException("Cannot create an articulation with no joints");
            for (var i = 1; i < joints.Count; i++)
            {
                if (joints[i] <= joints[i - 1])
                    throw new ArgumentException($"Joints must be provided in monotonically increasing order: {string.Join(", ", joints)}");
                if (joints[i] != joints[i - 1] + 1)
                    throw new ArgumentException($"Joints must be contiguous: gap between {joints[i - 1]} and {joints[i]}");
            }
            var childToParent = new Dictionary<int, int>();
            foreach (var j in joints)
            {
                if (j < 0 || j >= JointCount) throw new ArgumentException($"Joint index {j} is out of range");
                if (JointArticulation[j] >= 0)
                    throw new ArgumentException($"Joint {j} already belongs to articulation {JointArticulation[j]}");
                var child = JointChild[j];
                var parent = JointParent[j];
                if (childToParent.TryGetValue(child, out var other) && other != parent)
                    throw new ArgumentException($"Body {child} has multiple parents in this articulation: {other} and {parent}");
                childToParent[child] = parent;
                if (parent != -1 && (BodyFlags[child] & Model.BodyKinematic) != 0)
                    throw new ArgumentException($"Body {child} is kinematic but attached to parent body {parent}; " +
                                                "only root bodies (jointed to the world) can be kinematic");
            }
            var index = ArticulationCount;
            ArticulationStart.Add(joints[0]);
            ArticulationEnd.Add(joints[^1] + 1);
            foreach (var j in joints) JointArticulation[j] = index;
        }

        // finalize's joint checks (_validate_joints, the joint part of _validate_structure): no joint between two
        // articulations, no joint outside an articulation unless rooted at the world or closing a loop, no self-joint.
        void ValidateJoints()
        {
            for (var j = 0; j < JointCount; j++)
                if (JointParent[j] == JointChild[j])
                    throw new InvalidOperationException($"Self-referential joint {j} on body {JointParent[j]}");
            var childArticulations = new Dictionary<int, HashSet<int>>();
            var articulatedBodies = new HashSet<int>();
            for (var j = 0; j < JointCount; j++)
            {
                if (JointArticulation[j] < 0) continue;
                if (!childArticulations.TryGetValue(JointChild[j], out var set))
                    childArticulations[JointChild[j]] = set = new HashSet<int>();
                set.Add(JointArticulation[j]);
                articulatedBodies.Add(JointParent[j]);
                articulatedBodies.Add(JointChild[j]);
            }
            for (var j = 0; j < JointCount; j++)
            {
                var art = JointArticulation[j];
                if (art >= 0)
                {
                    if (childArticulations.TryGetValue(JointParent[j], out var parentArts) && !parentArts.Contains(art))
                        throw new InvalidOperationException($"Joint {j} of articulation {art} has parent body " +
                                                            $"{JointParent[j]} from another articulation");
                }
                else if (JointParent[j] != -1 && !articulatedBodies.Contains(JointChild[j]))
                    throw new InvalidOperationException($"Joint {j} does not belong to any articulation; " +
                                                        "call AddArticulation for all joints");
            }
        }

        void FinalizeJoints(Model model)
        {
            model.JointType = JointType.ToArray();
            model.JointEnabled = JointEnabled.ToArray();
            model.JointParent = JointParent.ToArray();
            model.JointChild = JointChild.ToArray();
            model.JointXP = JointXP.ToArray();
            model.JointXC = JointXC.ToArray();
            model.JointQStart = JointQStart.ToArray();
            model.JointQdStart = JointQdStart.ToArray();
            model.JointDofDim = JointDofDim.ToArray();
            model.JointQ = JointQ.ToArray();
            model.JointQd = JointQd.ToArray();
            model.JointAxis = JointAxis.ToArray();
            model.JointTargetKe = JointTargetKe.ToArray();
            model.JointTargetKd = JointTargetKd.ToArray();
            model.JointLimitLower = JointLimitLower.ToArray();
            model.JointLimitUpper = JointLimitUpper.ToArray();
            model.JointLimitKe = JointLimitKe.ToArray();
            model.JointLimitKd = JointLimitKd.ToArray();
            model.JointTargetQ = JointTargetQ.ToArray();
            model.JointTargetQd = JointTargetQd.ToArray();
            model.JointF = JointF.ToArray();
        }
    }
}
