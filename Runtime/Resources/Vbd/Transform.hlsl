// Newton's wp.transform: origin and rotation quaternion (x, y, z, w).
#ifndef VBD_TRANSFORM
#define VBD_TRANSFORM

struct Xform { float3 p; float4 q; };

// Warp's quat_rotate term by term: the same result as Newton for unit quaternions and also for slightly non-unit ones.
float3 QuatRotate(float4 q, float3 v)
{
    return v * (2.0 * q.w * q.w - 1.0) + cross(q.xyz, v) * q.w * 2.0 + q.xyz * dot(q.xyz, v) * 2.0;
}

float4 QuatMul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

float4 QuatInverse(float4 q) { return float4(-q.xyz, q.w); }

Xform XformIdentity()
{
    Xform x;
    x.p = 0;
    x.q = float4(0, 0, 0, 1);
    return x;
}

float3 XformPoint(Xform x, float3 p) { return x.p + QuatRotate(x.q, p); }
float3 XformVector(Xform x, float3 v) { return QuatRotate(x.q, v); }

Xform XformMul(Xform a, Xform b)
{
    Xform x;
    x.p = a.p + QuatRotate(a.q, b.p);
    x.q = QuatMul(a.q, b.q);
    return x;
}

Xform XformInverse(Xform a)
{
    Xform x;
    x.q = QuatInverse(a.q);
    x.p = -QuatRotate(x.q, a.p);
    return x;
}

#endif
