using System;
using Unity.Mathematics;

namespace Vbd
{
    // Port of newton/_src/geometry/inertia.py for the shapes the package has: sphere, box, capsule (+Z axis).
    public static class Inertia
    {
        public readonly struct MassProperties
        {
            public readonly float Mass;
            public readonly float3 Com;
            public readonly float3x3 Inertia;

            public MassProperties(double mass, float3 com, float3x3 inertia)
            {
                Mass = (float)mass;
                Com = com;
                Inertia = inertia;
            }
        }

        static float3x3 Diagonal(double a, double b, double c) => new((float)a, 0f, 0f, 0f, (float)b, 0f, 0f, 0f, (float)c);

        public static MassProperties Sphere(double density, double radius)
        {
            var m = density * 4.0 / 3.0 * math.PI_DBL * radius * radius * radius;
            var ia = 2.0 / 5.0 * m * radius * radius;
            return new MassProperties(m, default, Diagonal(ia, ia, ia));
        }

        public static float3x3 BoxFromMass(double mass, double hx, double hy, double hz) => Diagonal(
            1.0 / 3.0 * mass * (hy * hy + hz * hz),
            1.0 / 3.0 * mass * (hx * hx + hz * hz),
            1.0 / 3.0 * mass * (hx * hx + hy * hy));

        public static MassProperties Box(double density, double hx, double hy, double hz)
        {
            var m = density * 8.0 * hx * hy * hz;
            return new MassProperties(m, default, BoxFromMass(m, hx, hy, hz));
        }

        // Capsule along z; adapted from ODE as in Newton.
        public static MassProperties Capsule(double density, double radius, double halfHeight)
        {
            var h = 2.0 * halfHeight;
            var ms = density * (4.0 / 3.0) * math.PI_DBL * radius * radius * radius;
            var mc = density * math.PI_DBL * radius * radius * h;
            var ia = mc * (0.25 * radius * radius + 1.0 / 12.0 * h * h)
                     + ms * (0.4 * radius * radius + 0.375 * radius * h + 0.25 * h * h);
            var ib = (mc * 0.5 + ms * 0.4) * radius * radius;
            return new MassProperties(ms + mc, default, Diagonal(ia, ia, ib));
        }

        static MassProperties Subtract(MassProperties solid, MassProperties hollow) =>
            new(solid.Mass - hollow.Mass, solid.Com, solid.Inertia - hollow.Inertia);

        // compute_inertia_shape; thickness is the shape margin, used for hollow shapes only.
        public static MassProperties Shape(int type, float3 scale, float density, bool isSolid, float thickness)
        {
            if (density == 0f || type == GeoType.Plane) return default;
            switch (type)
            {
                case GeoType.Sphere:
                {
                    var solid = Sphere(density, scale.x);
                    if (isSolid) return solid;
                    if (!HollowThickness("sphere", thickness, scale.x)) return new MassProperties(0, solid.Com, default);
                    return Subtract(solid, Sphere(density, scale.x - thickness));
                }
                case GeoType.Box:
                {
                    var solid = Box(density, scale.x, scale.y, scale.z);
                    if (isSolid) return solid;
                    if (!HollowThickness("box", thickness, scale.x, scale.y, scale.z))
                        return new MassProperties(0, solid.Com, default);
                    return Subtract(solid, Box(density, scale.x - thickness, scale.y - thickness, scale.z - thickness));
                }
                case GeoType.Capsule:
                {
                    var solid = Capsule(density, scale.x, scale.y);
                    if (isSolid) return solid;
                    if (!HollowThickness("capsule", thickness, scale.x, scale.y)) return new MassProperties(0, solid.Com, default);
                    return Subtract(solid, Capsule(density, scale.x - thickness, scale.y - thickness));
                }
                default:
                    throw new NotSupportedException($"Unsupported shape type {type}");
            }
        }

        // _validate_hollow_thickness; false for zero thickness (zero mass and inertia).
        static bool HollowThickness(string shape, float thickness, params float[] limits)
        {
            if (!math.isfinite(thickness) || thickness < 0f)
                throw new ArgumentException($"thickness must be finite and >= 0 for a hollow {shape}; got {thickness}");
            if (thickness == 0f) return false;
            foreach (var limit in limits)
                if (thickness >= limit)
                    throw new ArgumentException($"thickness ({thickness}) must be smaller than {limit} for a hollow {shape}");
            return true;
        }

        // transform_inertia: rotate by quat, then parallel-axis shift by offset.
        public static float3x3 Transform(float mass, float3x3 inertia, float3 offset, quaternion quat)
        {
            var r = new float3x3(quat);
            var outer = new float3x3(offset * offset.x, offset * offset.y, offset * offset.z);
            var identity = float3x3.identity;
            return math.mul(math.mul(r, inertia), math.transpose(r)) + mass * (math.dot(offset, offset) * identity - outer);
        }

        public static bool AnyNonZero(float3x3 m) => math.any(m.c0 != 0f) || math.any(m.c1 != 0f) || math.any(m.c2 != 0f);

        static bool IsFinite(float3x3 m) => math.all(math.isfinite(m.c0)) && math.all(math.isfinite(m.c1)) && math.all(math.isfinite(m.c2));

        // validate_and_correct_inertia_kernel (balance_inertia on, no mass/inertia bounds unless given).
        public static void ValidateAndCorrect(ref float mass, ref float3x3 inertia, out float invMass, out float3x3 invInertia,
            bool balanceInertia = true, float boundMass = 0f, float boundInertia = 0f)
        {
            var original = inertia;
            var corrected = false;
            if (!math.isfinite(mass) || !IsFinite(inertia))
            {
                mass = 0f;
                inertia = default;
                corrected = true;
            }
            if (mass < 0f)
            {
                mass = 0f;
                corrected = true;
            }
            if (boundMass > 0f && mass < boundMass && mass > 0f)
            {
                mass = boundMass;
                corrected = true;
            }

            if (mass == 0f)
            {
                var sum = math.csum(inertia.c0 * inertia.c0 + inertia.c1 * inertia.c1 + inertia.c2 * inertia.c2);
                corrected |= sum > 0f;
                inertia = default;
            }
            else
            {
                // Matrices are by column, so m[c][r] = row r, column c of the mathematical matrix.
                var s01 = (inertia[1][0] + inertia[0][1]) * 0.5f;
                var s02 = (inertia[2][0] + inertia[0][2]) * 0.5f;
                var s12 = (inertia[2][1] + inertia[1][2]) * 0.5f;
                var sym = new float3x3(inertia[0][0], s01, s02, s01, inertia[1][1], s12, s02, s12, inertia[2][2]);
                var asymmetric = false;
                void Check(float a, float b, float s)
                {
                    var tol = 1e-8f + 1e-5f * math.abs(s);
                    asymmetric |= math.abs(a - s) > tol || math.abs(b - s) > tol;
                }
                Check(inertia[1][0], inertia[0][1], s01);
                Check(inertia[2][0], inertia[0][2], s02);
                Check(inertia[2][1], inertia[1][2], s12);
                corrected |= asymmetric;
                inertia = sym;

                SymmetricEigenvalues(inertia, out var i1, out var i2, out var i3);
                var threshold = math.max(1e-6f * i3, 1e-10f);
                if (i1 < threshold)
                {
                    var adjustment = threshold - i1 + 1e-6f;
                    i1 += adjustment;
                    i2 += adjustment;
                    i3 += adjustment;
                    inertia += adjustment * float3x3.identity;
                    corrected = true;
                }
                if (boundInertia > 0f && i1 < boundInertia)
                {
                    var adjustment = boundInertia - i1;
                    i1 += adjustment;
                    i2 += adjustment;
                    i3 += adjustment;
                    inertia += adjustment * float3x3.identity;
                    corrected = true;
                }
                var triTol = math.max(1.1920929e-7f * i3, 1e-10f);
                if (balanceInertia && i1 + i2 < i3 - triTol)
                {
                    inertia += (i3 - i1 - i2 + 1e-6f) * float3x3.identity;
                    corrected = true;
                }
            }

            if (!corrected) inertia = original;
            invMass = mass > 0f ? 1f / mass : 0f;
            invInertia = mass > 0f ? math.inverse(inertia) : default;
        }

        // Ascending eigenvalues of a symmetric 3x3 matrix (cyclic Jacobi in double precision).
        static void SymmetricEigenvalues(float3x3 m, out float e1, out float e2, out float e3)
        {
            var a = new double[3, 3];
            for (var r = 0; r < 3; r++)
            for (var c = 0; c < 3; c++)
                a[r, c] = m[c][r];
            for (var sweep = 0; sweep < 50; sweep++)
            {
                var off = Math.Abs(a[0, 1]) + Math.Abs(a[0, 2]) + Math.Abs(a[1, 2]);
                if (off < 1e-30) break;
                for (var p = 0; p < 2; p++)
                for (var q = p + 1; q < 3; q++)
                {
                    if (a[p, q] == 0.0) continue;
                    var theta = (a[q, q] - a[p, p]) / (2.0 * a[p, q]);
                    var t = Math.Sign(theta == 0.0 ? 1.0 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    var c = 1.0 / Math.Sqrt(t * t + 1.0);
                    var s = t * c;
                    for (var k = 0; k < 3; k++)
                    {
                        var akp = a[k, p];
                        var akq = a[k, q];
                        a[k, p] = c * akp - s * akq;
                        a[k, q] = s * akp + c * akq;
                    }
                    for (var k = 0; k < 3; k++)
                    {
                        var apk = a[p, k];
                        var aqk = a[q, k];
                        a[p, k] = c * apk - s * aqk;
                        a[q, k] = s * apk + c * aqk;
                    }
                }
            }
            var v = new[] { a[0, 0], a[1, 1], a[2, 2] };
            Array.Sort(v);
            e1 = (float)v[0];
            e2 = (float)v[1];
            e3 = (float)v[2];
        }
    }
}
