// Newton's collision_primitive.py for the shape pairs it solves analytically: each returns the signed surface
// distance, the midpoint between the two surfaces and the normal from the first shape to the second.
#ifndef VBD_COLLISION_PRIMITIVE
#define VBD_COLLISION_PRIMITIVE

#include "Transform.hlsl"

#define MAXVAL 1.0e10
#define MINVAL 1.0e-15

// wp.quat_to_matrix: the rotated basis vectors as columns.
float3x3 QuatToMatrix(float4 q)
{
    return transpose(float3x3(QuatRotate(q, float3(1, 0, 0)), QuatRotate(q, float3(0, 1, 0)), QuatRotate(q, float3(0, 0, 1))));
}

float3 QuatRotateInv(float4 q, float3 v) { return QuatRotate(QuatInverse(q), v); }

// wp.normalize: zero for a zero vector.
float3 WarpNormalize(float3 v)
{
    float l = length(v);
    return l > 0.0 ? v / l : 0;
}

// normalize_with_norm: a zero vector comes back unchanged with norm 0.
float3 NormalizeWithNorm(float3 v, out float norm)
{
    norm = length(v);
    return norm == 0.0 ? v : v / norm;
}

float3 ClosestSegmentPoint(float3 a, float3 b, float3 pt)
{
    float3 ab = b - a;
    float t = dot(pt - a, ab) / (dot(ab, ab) + 1e-6);
    return a + clamp(t, 0.0, 1.0) * ab;
}

void CollidePlaneSphere(float3 planeNormal, float3 planePos, float3 spherePos, float sphereRadius, out float dist, out float3 pos)
{
    dist = dot(spherePos - planePos, planeNormal) - sphereRadius;
    pos = spherePos - planeNormal * (sphereRadius + 0.5 * dist);
}

void CollideSphereSphere(float3 pos1, float radius1, float3 pos2, float radius2, out float dist, out float3 pos, out float3 n)
{
    float3 d = pos2 - pos1;
    float l = length(d);
    n = l == 0.0 ? float3(1, 0, 0) : d / l;
    dist = l - (radius1 + radius2);
    pos = pos1 + n * (radius1 + 0.5 * dist);
}

void CollideSphereCapsule(float3 spherePos, float sphereRadius, float3 capsulePos, float3 capsuleAxis, float capsuleRadius,
                          float capsuleHalfLength, out float dist, out float3 pos, out float3 n)
{
    float3 segment = capsuleAxis * capsuleHalfLength;
    float3 pt = ClosestSegmentPoint(capsulePos - segment, capsulePos + segment, spherePos);
    CollideSphereSphere(spherePos, sphereRadius, pt, capsuleRadius, dist, pos, n);
}

// One contact for crossing axes, two (one per end of the first capsule) for parallel ones; unused slots get MAXVAL.
void CollideCapsuleCapsule(float3 pos1, float3 axis1In, float radius1, float halfLength1, float3 pos2, float3 axis2In,
                           float radius2, float halfLength2, out float2 dist, out float3 p0, out float3 p1, out float3 n)
{
    dist = float2(MAXVAL, MAXVAL);
    p0 = 0;
    p1 = 0;
    float3 axis1 = axis1In * halfLength1;
    float3 axis2 = axis2In * halfLength2;
    float3 dif = pos1 - pos2;
    float ma = dot(axis1, axis1);
    float mb = -dot(axis1, axis2);
    float mc = dot(axis2, axis2);
    float u = -dot(axis1, dif);
    float v = dot(axis2, dif);
    float det = ma * mc - mb * mb;
    float d;
    if (abs(det) >= MINVAL)
    {
        float invDet = 1.0 / det;
        float x1 = (mc * u - mb * v) * invDet;
        float x2 = (ma * v - mb * u) * invDet;
        if (x1 > 1.0)
        {
            x1 = 1.0;
            x2 = (v - mb) / mc;
        }
        else if (x1 < -1.0)
        {
            x1 = -1.0;
            x2 = (v + mb) / mc;
        }
        if (x2 > 1.0)
        {
            x2 = 1.0;
            x1 = clamp((u - mb) / ma, -1.0, 1.0);
        }
        else if (x2 < -1.0)
        {
            x2 = -1.0;
            x1 = clamp((u + mb) / ma, -1.0, 1.0);
        }
        CollideSphereSphere(pos1 + axis1 * x1, radius1, pos2 + axis2 * x2, radius2, d, p0, n);
        dist.x = d;
    }
    else
    {
        float x2 = clamp((v - mb) / mc, -1.0, 1.0);
        CollideSphereSphere(pos1 + axis1, radius1, pos2 + axis2 * x2, radius2, d, p0, n);
        dist.x = d;
        float3 unused;
        x2 = clamp((v + mb) / mc, -1.0, 1.0);
        CollideSphereSphere(pos1 - axis1, radius1, pos2 + axis2 * x2, radius2, d, p1, unused);
        dist.y = d;
    }
}

// _collide_plane_capsule_contacts: one contact per end sphere.
void CollidePlaneCapsule(float3 planeNormal, float3 planePos, float3 capsulePos, float3 capsuleAxis, float capsuleRadius,
                         float capsuleHalfLength, out float2 dist, out float3 p0, out float3 p1)
{
    float3 segment = capsuleAxis * capsuleHalfLength;
    CollidePlaneSphere(planeNormal, planePos, capsulePos + segment, capsuleRadius, dist.x, p0);
    CollidePlaneSphere(planeNormal, planePos, capsulePos - segment, capsuleRadius, dist.y, p1);
}

// Up to the four deepest box corners within `margin` of the plane; unused slots get MAXVAL.
void CollidePlaneBox(float3 planeNormal, float3 planePos, float3 boxPos, float3x3 boxRot, float3 boxSize, float margin,
                     out float4 dist, out float3 p0, out float3 p1, out float3 p2, out float3 p3)
{
    float centerDist = dot(boxPos - planePos, planeNormal);
    float d[4] = { MAXVAL, MAXVAL, MAXVAL, MAXVAL };
    float3 pos[4] = { float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0) };
    int ncontact = 0, worst = 0;
    for (int i = 0; i < 8; i++)
    {
        float3 corner = float3((i & 1) != 0 ? boxSize.x : -boxSize.x, (i & 2) != 0 ? boxSize.y : -boxSize.y,
                               (i & 4) != 0 ? boxSize.z : -boxSize.z);
        corner = mul(boxRot, corner);
        float cdist = centerDist + dot(planeNormal, corner);
        if (cdist > margin) continue;
        float3 cpos = corner + boxPos - 0.5 * planeNormal * cdist;
        if (ncontact < 4)
        {
            d[ncontact] = cdist;
            pos[ncontact] = cpos;
            if (ncontact == 0 || cdist > d[worst]) worst = ncontact;
            ncontact++;
        }
        else if (cdist < d[worst])
        {
            d[worst] = cdist;
            pos[worst] = cpos;
            worst = 0;
            if (d[1] > d[worst]) worst = 1;
            if (d[2] > d[worst]) worst = 2;
            if (d[3] > d[worst]) worst = 3;
        }
    }
    dist = float4(d[0], d[1], d[2], d[3]);
    p0 = pos[0]; p1 = pos[1]; p2 = pos[2]; p3 = pos[3];
}

void CollideSphereBox(float3 spherePos, float sphereRadius, float3 boxPos, float3x3 boxRot, float3 boxSize,
                      out float dist, out float3 position, out float3 n)
{
    float3 center = mul(transpose(boxRot), spherePos - boxPos);
    float3 clamped = max(-boxSize, min(boxSize, center));
    float d;
    float3 clampedDir = NormalizeWithNorm(clamped - center, d);
    float3 pos;
    if (d <= 1e-6)
    {
        float closest = 2.0 * (boxSize.x + boxSize.y + boxSize.z);
        int k = 0;
        for (int i = 0; i < 6; i++)
        {
            float faceDist = abs((i % 2 != 0 ? 1.0 : -1.0) * boxSize[i / 2] - center[i / 2]);
            if (closest > faceDist)
            {
                closest = faceDist;
                k = i;
            }
        }
        float sgn = k % 2 != 0 ? -1.0 : 1.0;
        float3 nearest = k / 2 == 0 ? float3(sgn, 0, 0) : k / 2 == 1 ? float3(0, sgn, 0) : float3(0, 0, sgn);
        pos = center + nearest * (sphereRadius - closest) / 2.0;
        n = mul(boxRot, nearest);
        dist = -closest - sphereRadius;
    }
    else
    {
        float3 deepest = center + clampedDir * sphereRadius;
        pos = 0.5 * (clamped + deepest);
        n = mul(boxRot, clampedDir);
        dist = d - sphereRadius;
    }
    position = boxPos + mul(boxRot, pos);
}

#endif
