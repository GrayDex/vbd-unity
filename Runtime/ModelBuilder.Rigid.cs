using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace Vbd
{
    // Rigid bodies and shapes. Joints are added here when the rigid solver defines them.
    public sealed partial class ModelBuilder
    {
        // newton.ShapeFlags
        public static class ShapeFlags
        {
            public const int Visible = 1 << 0;
            public const int CollideShapes = 1 << 1;
            public const int CollideParticles = 1 << 2;
            public const int Site = 1 << 3;
        }

        // newton.ModelBuilder.ShapeConfig: the fields the Model keeps, with Newton's defaults.
        public sealed class ShapeConfig
        {
            public float Density = 1000f;
            public float Ke = 2.5e3f;
            public float Kd = 100f;
            public float Mu = 1f;
            // Outward offset of the collision surface; also the wall thickness of hollow shapes.
            public float Margin;
            // Null takes ModelBuilder.RigidGap.
            public float? Gap;
            public bool IsSolid = true;
            public bool HasShapeCollision = true;
            public bool HasParticleCollision = true;
            public bool IsVisible = true;
            public bool IsSite;

            public int Flags =>
                (IsVisible ? ShapeFlags.Visible : 0) | (HasShapeCollision ? ShapeFlags.CollideShapes : 0)
                | (HasParticleCollision ? ShapeFlags.CollideParticles : 0) | (IsSite ? ShapeFlags.Site : 0);

            public ShapeConfig Copy() => (ShapeConfig)MemberwiseClone();

            public void MarkAsSite()
            {
                IsSite = true;
                HasShapeCollision = false;
                HasParticleCollision = false;
                Density = 0f;
            }
        }

        public ShapeConfig DefaultShapeCfg = new();
        public float RigidGap = 0.1f;
        // Newton's inertia validation settings (compiler settings in its builder).
        public bool BalanceInertia = true;
        public float BoundMass;
        public float BoundInertia;

        public readonly List<Transform> BodyQ = new();
        public readonly List<float3> BodyCom = new();
        public readonly List<float> BodyMass = new();
        public readonly List<float3x3> BodyInertia = new();
        public readonly List<float> BodyInvMass = new();
        public readonly List<float3x3> BodyInvInertia = new();
        public readonly List<bool> BodyLockInertia = new();

        public readonly List<int> ShapeBody = new();
        public readonly List<Transform> ShapeTransform = new();
        public readonly List<int> ShapeType = new();
        public readonly List<float3> ShapeScale = new();
        public readonly List<int> ShapeFlagsList = new();
        public readonly List<float> ShapeMargin = new();
        public readonly List<float> ShapeGap = new();
        public readonly List<float> ShapeMaterialKe = new();
        public readonly List<float> ShapeMaterialKd = new();
        public readonly List<float> ShapeMaterialMu = new();

        public int BodyCount => BodyQ.Count;
        public int ShapeCount => ShapeType.Count;

        static Transform Identity => new(float3.zero, quaternion.identity);

        // add_link + add_body without the free joint: pose, center of mass in the body frame, inertia about it.
        // lockInertia stops shapes added later from changing the mass properties.
        public int AddBody(Transform? xform = null, float3? com = null, float3x3? inertia = null, float mass = 0f,
            bool lockInertia = false)
        {
            var i = inertia ?? default;
            BodyInertia.Add(i);
            BodyMass.Add(mass);
            BodyCom.Add(com ?? float3.zero);
            BodyLockInertia.Add(lockInertia);
            BodyInvMass.Add(mass > 0f ? 1f / mass : 0f);
            BodyInvInertia.Add(Inertia.AnyNonZero(i) ? math.inverse(i) : i);
            BodyQ.Add(xform ?? Identity);
            return BodyQ.Count - 1;
        }

        // _update_body_mass: merges mass m with inertia about its own center, placed at p with rotation q in the body.
        void UpdateBodyMass(int i, float m, float3x3 inertia, float3 p, quaternion q)
        {
            if (i == -1) return;
            var newMass = BodyMass[i] + m;
            if (newMass == 0f) return;
            var newCom = (BodyCom[i] * BodyMass[i] + p * m) / newMass;
            var newInertia = Inertia.Transform(BodyMass[i], BodyInertia[i], newCom - BodyCom[i], quaternion.identity)
                             + Inertia.Transform(m, inertia, newCom - p, q);
            BodyMass[i] = newMass;
            BodyInertia[i] = newInertia;
            BodyCom[i] = newCom;
            BodyInvMass[i] = newMass > 0f ? 1f / newMass : 0f;
            BodyInvInertia[i] = Inertia.AnyNonZero(newInertia) ? math.inverse(newInertia) : newInertia;
        }

        // add_shape; scale is as Newton: sphere radius in x, capsule radius in x and half height in y, box half extents.
        public int AddShape(int body, int type, Transform? xform, ShapeConfig cfg, float3 scale, bool isStatic = false)
        {
            if (body < -1 || body >= BodyCount) throw new ArgumentException($"Invalid body index {body}");
            cfg ??= DefaultShapeCfg;
            if (!math.isfinite(cfg.Density) || cfg.Density < 0f) throw new ArgumentException($"density must be finite and >= 0 (got {cfg.Density})");
            if (cfg.IsSite && (cfg.HasShapeCollision || cfg.HasParticleCollision || cfg.Density != 0f))
                throw new ArgumentException("A site cannot collide or have density");
            var pose = xform ?? Identity;
            scale = math.abs(scale);

            ShapeBody.Add(body);
            ShapeTransform.Add(pose);
            ShapeFlagsList.Add(cfg.Flags);
            ShapeType.Add(type);
            ShapeScale.Add(scale);
            ShapeMargin.Add(cfg.Margin);
            ShapeGap.Add(cfg.Gap ?? RigidGap);
            ShapeMaterialKe.Add(cfg.Ke);
            ShapeMaterialKd.Add(cfg.Kd);
            ShapeMaterialMu.Add(cfg.Mu);

            if (!isStatic && cfg.Density > 0f && body >= 0 && !BodyLockInertia[body])
            {
                var props = Inertia.Shape(type, scale, cfg.Density, cfg.IsSolid, cfg.Margin);
                UpdateBodyMass(body, props.Mass, props.Inertia, math.rotate(pose.Q, props.Com) + pose.P, pose.Q);
            }
            return ShapeCount - 1;
        }

        public int AddShapeSphere(int body, Transform? xform = null, float radius = 1f, ShapeConfig cfg = null) =>
            AddShape(body, GeoType.Sphere, xform, cfg, new float3(radius, 0f, 0f));

        // Capsule axis is the local z axis.
        public int AddShapeCapsule(int body, Transform? xform = null, float radius = 1f, float halfHeight = 0.5f,
            ShapeConfig cfg = null) =>
            AddShape(body, GeoType.Capsule, xform, cfg, new float3(radius, halfHeight, 0f));

        public int AddShapeBox(int body, Transform? xform = null, float hx = 0.5f, float hy = 0.5f, float hz = 0.5f,
            ShapeConfig cfg = null) =>
            AddShape(body, GeoType.Box, xform, cfg, new float3(hx, hy, hz));

        // add_shape_plane: without xform the pose comes from the plane equation (a, b, c, d); the normal is the local z axis.
        public int AddShapePlane(float4? plane = null, Transform? xform = null, float width = 10f, float length = 10f,
            int body = -1, ShapeConfig cfg = null)
        {
            if (xform == null)
            {
                var eq = plane ?? new float4(0f, 0f, 1f, 0f);
                var norm = math.length(eq.xyz);
                var normal = eq.xyz / norm;
                xform = new Transform(-eq.w / norm * normal, QuatBetweenVectors(new float3(0f, 0f, 1f), normal));
            }
            return AddShape(body, GeoType.Plane, xform, cfg, new float3(width, length, 0f), true);
        }

        // add_ground_plane: the plane z = height, infinite.
        public int AddGroundPlane(float height = 0f, ShapeConfig cfg = null) =>
            AddShapePlane(new float4(0f, 0f, 1f, -height), null, 0f, 0f, -1, cfg);

        // wp.quat_between_vectors; opposite vectors turn half a circle about x, which Warp leaves undefined.
        static quaternion QuatBetweenVectors(float3 a, float3 b)
        {
            a = math.normalize(a);
            b = math.normalize(b);
            var d = math.dot(a, b);
            if (d < -0.999999f) return new quaternion(1f, 0f, 0f, 0f);
            var c = math.cross(a, b);
            return math.normalizesafe(new quaternion(c.x, c.y, c.z, 1f + d));
        }

        void FinalizeRigid(Model model)
        {
            var n = BodyCount;
            model.BodyQ = BodyQ.ToArray();
            model.BodyQd = new SpatialVelocity[n];
            model.BodyCom = BodyCom.ToArray();
            model.BodyMass = new float[n];
            model.BodyInvMass = new float[n];
            model.BodyInertia = new float3x3[n];
            model.BodyInvInertia = new float3x3[n];
            for (var i = 0; i < n; i++)
            {
                var mass = BodyMass[i];
                var inertia = BodyInertia[i];
                Inertia.ValidateAndCorrect(ref mass, ref inertia, out var invMass, out var invInertia, BalanceInertia,
                    BoundMass, BoundInertia);
                model.BodyMass[i] = mass;
                model.BodyInertia[i] = inertia;
                model.BodyInvMass[i] = invMass;
                model.BodyInvInertia[i] = invInertia;
            }
            model.ShapeBody = ShapeBody.ToArray();
            model.ShapeTransform = ShapeTransform.ToArray();
            model.ShapeType = ShapeType.ToArray();
            model.ShapeScale = ShapeScale.ToArray();
            model.ShapeFlags = ShapeFlagsList.ToArray();
            model.ShapeMargin = ShapeMargin.ToArray();
            model.ShapeGap = ShapeGap.ToArray();
            model.ShapeMaterialKe = ShapeMaterialKe.ToArray();
            model.ShapeMaterialKd = ShapeMaterialKd.ToArray();
            model.ShapeMaterialMu = ShapeMaterialMu.ToArray();
        }
    }
}
