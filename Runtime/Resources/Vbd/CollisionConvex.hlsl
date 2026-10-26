// Newton's fused MPR/GJK contact path for the convex pairs the analytic kernel leaves (capsule-box, box-box):
// mpr.py (portal, refine, raycast, separated), simplex_solver.py (GJK distance), penetration.py (box-box depth),
// multicontact.py (manifold) and collision_convex.py create_solve_convex_multi_contact. Included by
// CollideRigid.compute after ContactData and WriteContact; shape A sits at the origin, B is relative to it.
// Only the box and capsule supports exist; capsules are inflated cores of radius 1e-4 (the caller adds radius_eff).
#ifndef VBD_COLLISION_CONVEX
#define VBD_COLLISION_CONVEX

#define CVX_EPSILON 1.0e-8
#define FLOAT32_EPS 1.1920929e-7
#define GJK_CUTOFF_TOLERANCE (64.0 * FLOAT32_EPS)
#define GJK_REFINE_TOLERANCE (4.0 * FLOAT32_EPS)
#define MANIFOLD_EPS 0.00001
#define MAX_CLIP_VERTEX_MERGE_DISTANCE 0.0031622776601683794
#define SIN_TILT_ANGLE 0.03489949670250097
#define COS_TILT_ANGLE 0.9993908270190958
#define COS_DEEPEST_CONTACT_THRESHOLD_ANGLE 0.9999984769132877

struct Geom
{
    int type;
    float3 scale;
    float3 bounds;
};

struct Vert
{
    float3 B;
    float3 BtoA;
};

// ---- Support maps ----

float3 SupportBox(Geom g, float3 d)
{
    float ds = max(abs(d.x), max(abs(d.y), abs(d.z)));
    float th = 1.0e-10 * ds;
    return float3(d.x >= -th ? g.scale.x : -g.scale.x, d.y >= -th ? g.scale.y : -g.scale.y,
                  d.z >= -th ? g.scale.z : -g.scale.z);
}

float3 SupportMap(Geom g, float3 d)
{
    if (g.type == GEO_BOX) return SupportBox(g, d);
    float lsq = dot(d, d);
    float3 n = lsq > 1.0e-12 ? d * rsqrt(lsq) : float3(1, 0, 0);
    return n * g.scale.x + float3(0, 0, d.z >= 0.0 ? g.scale.y : -g.scale.y);
}

// create_shape_support_function: with center ties a box zeroes the axes that barely contribute.
float3 ShapeSupport(Geom g, float3 d, bool ties)
{
    float3 r = SupportMap(g, d);
    if (ties && g.type == GEO_BOX)
    {
        float3 c = abs(d) * g.scale;
        float th = 1.0e-6 * (c.x + c.y + c.z);
        if (c.x <= th) r.x = 0.0;
        if (c.y <= th) r.y = 0.0;
        if (c.z <= th) r.z = 0.0;
    }
    return r;
}

Vert MinkowskiSupport(Geom a, Geom b, float3 dir, float4 qB, float3 pB, float extend, bool ties)
{
    Vert v;
    float3 pa = ShapeSupport(a, dir, ties);
    v.B = QuatRotate(qB, ShapeSupport(b, QuatRotateInv(qB, -dir), ties)) + pB;
    if (extend != 0.0)
    {
        float3 d = WarpNormalize(dir) * extend * 0.5;
        pa = pa + d;
        v.B = v.B - d;
    }
    v.BtoA = pa - v.B;
    return v;
}

float LengthSq(float3 v) { return dot(v, v); }

// ---- GJK distance (simplex_solver.py) ----

// Simplex of four vertices; slot 2i holds B, slot 2i + 1 holds BtoA.
static float3 gSimplex[8];

float CoordinateScale(float3 x) { return abs(x.x) + abs(x.y) + abs(x.z); }

// A vector component write with a dynamic index would force the enclosing loops to unroll.
float4 SetComponent(float4 v, int i, float x)
{
    return float4(i == 0 ? x : v.x, i == 1 ? x : v.y, i == 2 ? x : v.z, i == 3 ? x : v.w);
}

void ClosestSegment(int i0, int i1, out float3 closest, out float4 bc, out uint mask)
{
    float3 a = gSimplex[2 * i0 + 1];
    float3 b = gSimplex[2 * i1 + 1];
    float3 edge = b - a;
    float vsq = LengthSq(edge);
    bool degenerate = vsq < CVX_EPSILON * CVX_EPSILON;
    float denom = degenerate ? CVX_EPSILON * CVX_EPSILON : vsq;
    float t = -dot(a, edge) / denom;
    float lambda0 = 1.0 - t;
    float lambda1 = t;
    mask = (1u << (uint)i0) | (1u << (uint)i1);
    bc = 0;
    if (lambda0 < 0.0 || degenerate)
    {
        mask = 1u << (uint)i1;
        lambda0 = 0.0;
        lambda1 = 1.0;
    }
    else if (lambda1 < 0.0)
    {
        mask = 1u << (uint)i0;
        lambda0 = 1.0;
        lambda1 = 0.0;
    }
    bc = SetComponent(SetComponent(bc, i0, lambda0), i1, lambda1);
    closest = lambda0 * a + lambda1 * b;
    if (lambda0 > 0.0 && lambda1 > 0.0)
    {
        float scaleSq = max(LengthSq(a), LengthSq(b));
        if (LengthSq(closest) <= max(CVX_EPSILON * CVX_EPSILON, 1.0e-4 * scaleSq))
            closest = cross(edge, cross(a, edge)) * (1.0 / vsq);
    }
}

void ClosestTriangle(int i0, int i1, int i2, out float3 closestPt, out float4 bc, out uint mask)
{
    float3 a = gSimplex[2 * i0 + 1];
    float3 b = gSimplex[2 * i1 + 1];
    float3 c = gSimplex[2 * i2 + 1];
    float3 u = a - b;
    float3 w = a - c;
    float3 normal = cross(u, w);
    float t = LengthSq(normal);
    bool degenerate = t <= CVX_EPSILON * LengthSq(u) * LengthSq(w);
    float denom = degenerate ? CVX_EPSILON : t;
    float it = 1.0 / denom;
    float3 c1 = cross(u, a);
    float3 c2 = cross(a, w);
    float lambda2 = dot(c1, normal) * it;
    float lambda1 = dot(c2, normal) * it;
    float lambda0 = 1.0 - lambda2 - lambda1;
    float bestDistance = 1.0e30;
    closestPt = 0;
    bc = 0;
    mask = 0;
    float3 closest;
    float4 bcTmp;
    uint m;
    if (lambda0 < 0.0 || degenerate)
    {
        ClosestSegment(i1, i2, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; bestDistance = dist; closestPt = closest; }
    }
    if (lambda1 < 0.0 || degenerate)
    {
        ClosestSegment(i0, i2, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; bestDistance = dist; closestPt = closest; }
    }
    if (lambda2 < 0.0 || degenerate)
    {
        ClosestSegment(i0, i1, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; closestPt = closest; }
    }
    if (mask != 0u) return;
    bc = SetComponent(SetComponent(SetComponent(bc, i0, lambda0), i1, lambda1), i2, lambda2);
    mask = (1u << (uint)i0) | (1u << (uint)i1) | (1u << (uint)i2);
    closestPt = normal * (dot(normal, a) * it);
}

float Determinant(float3 a, float3 b, float3 c, float3 d) { return dot(b - a, cross(c - a, d - a)); }

void ClosestTetrahedron(out float3 closestPt, out float4 bc, out uint mask)
{
    float3 v0 = gSimplex[1], v1 = gSimplex[3], v2 = gSimplex[5], v3 = gSimplex[7];
    float detT = Determinant(v0, v1, v2, v3);
    bool degenerate = abs(detT) < CVX_EPSILON;
    float inverseDetT = 1.0 / (degenerate ? CVX_EPSILON : detT);
    float3 zero = 0;
    float lambda0 = Determinant(zero, v1, v2, v3) * inverseDetT;
    float lambda1 = Determinant(v0, zero, v2, v3) * inverseDetT;
    float lambda2 = Determinant(v0, v1, zero, v3) * inverseDetT;
    float lambda3 = 1.0 - lambda0 - lambda1 - lambda2;
    float bestDistance = 1.0e30;
    closestPt = 0;
    bc = 0;
    mask = 0;
    float3 closest;
    float4 bcTmp;
    uint m;
    if (lambda0 < 0.0 || degenerate)
    {
        ClosestTriangle(1, 2, 3, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; bestDistance = dist; closestPt = closest; }
    }
    if (lambda1 < 0.0 || degenerate)
    {
        ClosestTriangle(0, 2, 3, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; bestDistance = dist; closestPt = closest; }
    }
    if (lambda2 < 0.0 || degenerate)
    {
        ClosestTriangle(0, 1, 3, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; bestDistance = dist; closestPt = closest; }
    }
    if (lambda3 < 0.0 || degenerate)
    {
        ClosestTriangle(0, 1, 2, closest, bcTmp, m);
        float dist = LengthSq(closest);
        if (dist < bestDistance) { bc = bcTmp; mask = m; closestPt = closest; }
    }
    if (mask != 0u) return;
    bc = float4(lambda0, lambda1, lambda2, lambda3);
    mask = 15u;
}

void SimplexClosest(float4 bc, uint usage, out float3 pointA, out float3 pointB)
{
    pointA = 0;
    pointB = 0;
    [loop]
    for (int i = 0; i < 4; i++)
    {
        if ((usage & (1u << (uint)i)) == 0u) continue;
        float3 vb = gSimplex[2 * i];
        float3 vba = gSimplex[2 * i + 1];
        pointA = pointA + bc[i] * (vb + vba);
        pointB = pointB + bc[i] * vb;
    }
}

// solve_closest_distance_core; returns whether the shapes are separated.
bool GjkCore(Geom a, Geom b, float4 qB, float3 pB, float extend, int maxIter, float collideEps, float maxDist,
             out float3 pointA, out float3 pointB, out float3 normal, out float distance)
{
    pointA = 0;
    pointB = 0;
    normal = 0;
    distance = 0;
    [loop]
    for (int k = 0; k < 8; k++) gSimplex[k] = 0;
    float4 barycentric = 0;
    uint usage = 0u;
    int iterCount = maxIter;
    float3 v = -pB;
    float distSq = LengthSq(v);
    float3 lastSearchDir = float3(1, 0, 0);
    float supportScale = 0.0;
    bool finished = false, resSep = false;
    [loop]
    while (iterCount > 0)
    {
        iterCount--;
        float duplicateEpsilon = max(CVX_EPSILON, GJK_REFINE_TOLERANCE * supportScale);
        if (distSq < collideEps * collideEps)
        {
            if (usage != 0u)
            {
                float3 nearDirection = lastSearchDir;
                if (distSq > 0.0) nearDirection = -v;
                if (LengthSq(nearDirection) > 0.0)
                {
                    float3 nearNormal = WarpNormalize(nearDirection);
                    Vert support = MinkowskiSupport(a, b, nearNormal, qB, pB, extend, false);
                    float supportPlane = dot(nearNormal, support.BtoA);
                    if (supportPlane <= 0.0 && distSq <= CVX_EPSILON * CVX_EPSILON)
                    {
                        SimplexClosest(barycentric, usage, pointA, pointB);
                        normal = nearNormal;
                        if (supportPlane < 0.0)
                        {
                            distance = -supportPlane;
                            finished = true; resSep = true; break;
                        }
                        distance = 0;
                        finished = true; resSep = false; break;
                    }
                }
            }
            if (usage != 0u && distSq <= CVX_EPSILON * CVX_EPSILON)
            {
                SimplexClosest(barycentric, usage, pointA, pointB);
                normal = 0;
                distance = 0;
                finished = true; resSep = false; break;
            }
            duplicateEpsilon = CVX_EPSILON;
        }
        float3 searchDir = -v;
        lastSearchDir = searchDir;
        Vert w = MinkowskiSupport(a, b, searchDir, qB, pB, extend, false);
        float3 wv = w.BtoA;
        supportScale = max(supportScale, CoordinateScale(w.B) + CoordinateScale(wv));
        if (maxDist > 0.0)
        {
            float cutoff = maxDist + GJK_CUTOFF_TOLERANCE * (supportScale + maxDist);
            if (usage != 0u && dot(v, wv) > cutoff * sqrt(distSq)) break;
        }
        float deltaDist = dot(v, v - wv);
        if (usage != 0u && (deltaDist <= 0.0 || deltaDist < GJK_REFINE_TOLERANCE * supportScale * sqrt(distSq))) break;
        bool isDuplicate = false;
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            if ((usage & (1u << (uint)i)) != 0u && LengthSq(gSimplex[2 * i + 1] - wv) < duplicateEpsilon * duplicateEpsilon)
            {
                isDuplicate = true;
                break;
            }
        }
        if (isDuplicate) break;

        int useCount = 0, freeSlot = 0;
        int indices[4] = { 0, 0, 0, 0 };
        [unroll]
        for (int j = 0; j < 4; j++)
        {
            if ((usage & (1u << (uint)j)) != 0u)
            {
                indices[useCount] = j;
                useCount++;
            }
            else
                freeSlot = j;
        }
        indices[useCount] = freeSlot;
        useCount++;
        gSimplex[2 * freeSlot] = w.B;
        gSimplex[2 * freeSlot + 1] = w.BtoA;
        float3 closest = 0;
        bool success = true;
        float4 bc;
        uint mask;
        if (useCount == 1)
        {
            closest = gSimplex[2 * indices[0] + 1];
            usage = 1u << (uint)indices[0];
            barycentric = SetComponent(float4(0, 0, 0, 0), indices[0], 1.0);
        }
        else if (useCount == 2)
        {
            ClosestSegment(indices[0], indices[1], closest, bc, mask);
            barycentric = bc;
            usage = mask;
        }
        else if (useCount == 3)
        {
            ClosestTriangle(indices[0], indices[1], indices[2], closest, bc, mask);
            barycentric = bc;
            usage = mask;
        }
        else
        {
            ClosestTetrahedron(closest, bc, mask);
            barycentric = bc;
            usage = mask;
            success = mask != 15u;
        }
        if (!success)
        {
            SimplexClosest(barycentric, usage, pointA, pointB);
            normal = 0;
            distance = 0;
            finished = true; resSep = false; break;
        }
        v = closest;
        distSq = LengthSq(v);
    }
    if (finished) return resSep;
    SimplexClosest(barycentric, usage, pointA, pointB);
    if (distSq < collideEps * collideEps)
    {
        float3 nearNormal = WarpNormalize(-v);
        Vert support = MinkowskiSupport(a, b, nearNormal, qB, pB, extend, false);
        if (dot(nearNormal, support.BtoA) >= 0.0)
        {
            normal = 0;
            distance = 0;
            return false;
        }
    }
    normal = WarpNormalize(-v);
    distance = dot(pointB - pointA, normal);
    return true;
}

// ---- Box-box depth (penetration.py _box_box_depth, mpr.py _enclosing_box_depth) ----

float3 AxisOf(int index)
{
    return float3(index == 0 ? 1.0 : 0.0, index == 1 ? 1.0 : 0.0, index == 2 ? 1.0 : 0.0);
}

void BoxBoxDepth(float3 halfA, float3 halfB, float4 rotation, float3 position, out float3 normal, out float depth)
{
    float3x3 axesB = QuatToMatrix(rotation);
    float3x3 axesBT = transpose(axesB);
    normal = float3(1, 0, 0);
    depth = 1.0e30;
    [loop]
    for (int index = 0; index < 15; index++)
    {
        float3 axis = 0;
        if (index < 3) axis = AxisOf(index);
        else if (index < 6) axis = mul(axesB, AxisOf(index - 3));
        else axis = cross(AxisOf((index - 6) / 3), mul(axesB, AxisOf((index - 6) % 3)));
        float len = length(axis);
        if (len > 0.0)
        {
            axis = axis / len;
            float radius = dot(abs(axis), halfA) + dot(abs(mul(axesBT, axis)), halfB);
            float offset = dot(axis, position);
            float overlap = radius - abs(offset);
            if (overlap < depth)
            {
                depth = overlap;
                normal = offset >= 0.0 ? axis : -axis;
            }
        }
    }
}

void EnclosingBoxDepth(float3 halfA, float3 halfB, float4 rotation, float3 position, out float3 normal, out float depth)
{
    float3x3 axesB = QuatToMatrix(rotation);
    normal = float3(1, 0, 0);
    depth = 1.0e30;
    [loop]
    for (int index = 0; index < 3; index++)
    {
        float3 axisA = AxisOf(index);
        float overlapA = halfA[index] + dot(abs(axesB[index]), halfB) - abs(position[index]);
        if (overlapA < depth)
        {
            depth = overlapA;
            normal = position[index] >= 0.0 ? axisA : -axisA;
        }
        float3 axisB = mul(axesB, axisA);
        float offsetB = dot(axisB, position);
        float overlapB = halfB[index] + dot(abs(axisB), halfA) - abs(offsetB);
        if (overlapB < depth)
        {
            depth = overlapB;
            normal = offsetB >= 0.0 ? axisB : -axisB;
        }
    }
}

// ---- MPR (mpr.py) ----

float3 PortalRetrySeed(Vert v0, Vert v1, Vert v2, Vert v3, float3 normal, float3 seed)
{
    float radius = 2.0 * length(seed);
    if (radius == 0.0)
    {
        radius = abs(dot(normal, v1.BtoA));
        [loop]
        for (int edge = 0; edge < 3; edge++)
        {
            float3 edgeA = v1.BtoA;
            float3 edgeB = v2.BtoA;
            if (edge == 1) { edgeA = v2.BtoA; edgeB = v3.BtoA; }
            else if (edge == 2) { edgeA = v3.BtoA; edgeB = v1.BtoA; }
            float3 face = cross(edgeA - v0.BtoA, edgeB - v0.BtoA);
            float faceLength = length(face);
            if (faceLength > 0.0) radius = min(radius, abs(dot(face, v0.BtoA)) / faceLength);
        }
    }
    return -normal * (0.5 * radius);
}

// solve_mpr_portal; returns `hit`.
bool MprPortal(Geom a, Geom b, float4 qB, float3 pB, float extend, float3 seed, int maxIter, float collideEps,
               float retryDepth, out float3 pointA, out float3 pointB, out float3 normal, out float penetration,
               out float3 nextSeed, out bool valid)
{
    const float NUMERIC_EPSILON = 1.0e-16;
    penetration = 0;
    pointA = 0;
    pointB = 0;
    normal = 0;
    nextSeed = 0;
    valid = true;
    Vert v0;
    v0.B = pB;
    v0.BtoA = -pB;
    if (LengthSq(seed) > 0.0) v0.BtoA = seed;
    normal = v0.BtoA;
    if (LengthSq(normal) < NUMERIC_EPSILON)
    {
        v0.BtoA = 0;
        float bestDot = -1.0e30;
        float3 bestDir = float3(1, 0, 0);
        [loop]
        for (int axisIdx = 0; axisIdx < 3; axisIdx++)
        {
            float3 probe = AxisOf(axisIdx);
            Vert sv = MinkowskiSupport(a, b, probe, qB, pB, extend, true);
            float d = dot(sv.BtoA, probe);
            if (d > bestDot) { bestDot = d; bestDir = probe; }
        }
        v0.BtoA = bestDir * 1.0e-5;
    }
    normal = -v0.BtoA;
    Vert v1 = MinkowskiSupport(a, b, normal, qB, pB, extend, true);
    pointA = v1.B + v1.BtoA;
    pointB = v1.B;
    if (dot(v1.BtoA, normal) <= 0.0) return false;
    normal = cross(v1.BtoA, v0.BtoA);
    if (LengthSq(normal) < NUMERIC_EPSILON * NUMERIC_EPSILON)
    {
        normal = WarpNormalize(v1.BtoA - v0.BtoA);
        penetration = dot(v1.BtoA, normal);
        return true;
    }
    Vert v2 = MinkowskiSupport(a, b, normal, qB, pB, extend, true);
    if (dot(v2.BtoA, normal) <= 0.0) return false;
    float3 temp1 = v1.BtoA - v0.BtoA;
    float3 temp2 = v2.BtoA - v0.BtoA;
    normal = cross(temp1, temp2);
    float dist = dot(normal, v0.BtoA);
    if (dist > 0.0)
    {
        Vert tmp = v1;
        v1 = v2;
        v2 = tmp;
        normal = -normal;
    }
    int phase1 = 0, phase2 = 0;
    bool hit = false;
    Vert v3;
    v3.B = 0;
    v3.BtoA = 0;
    [loop]
    while (true)
    {
        if (phase1 > maxIter) return false;
        phase1++;
        v3 = MinkowskiSupport(a, b, normal, qB, pB, extend, true);
        if (dot(v3.BtoA, normal) <= 0.0) return false;
        temp1 = cross(v1.BtoA, v3.BtoA);
        if (dot(temp1, v0.BtoA) < 0.0)
        {
            v2 = v3;
            normal = cross(v1.BtoA - v0.BtoA, v3.BtoA - v0.BtoA);
            continue;
        }
        temp1 = cross(v3.BtoA, v2.BtoA);
        if (dot(temp1, v0.BtoA) < 0.0)
        {
            v1 = v3;
            normal = cross(v3.BtoA - v0.BtoA, v2.BtoA - v0.BtoA);
            continue;
        }
        break;
    }
    [loop]
    while (true)
    {
        phase2++;
        temp1 = v2.BtoA - v1.BtoA;
        temp2 = v3.BtoA - v1.BtoA;
        normal = cross(temp1, temp2);
        float normalSq = LengthSq(normal);
        if (normalSq < NUMERIC_EPSILON * NUMERIC_EPSILON)
        {
            valid = false;
            return hit;
        }
        if (!hit) hit = dot(normal, v1.BtoA) >= 0.0;
        Vert v4 = MinkowskiSupport(a, b, normal, qB, pB, extend, true);
        float3 temp3 = v4.BtoA - v3.BtoA;
        float delta = dot(temp3, normal);
        penetration = dot(v4.BtoA, normal);
        if (delta * delta <= collideEps * collideEps * normalSq || penetration <= 0.0 || phase2 > maxIter)
        {
            if (hit)
            {
                float invNormal = 1.0 / sqrt(normalSq);
                penetration *= invNormal;
                normal = normal * invNormal;
                temp3 = cross(v1.BtoA, temp1);
                float gamma = dot(temp3, normal) * invNormal;
                temp3 = cross(temp2, v1.BtoA);
                float beta = dot(temp3, normal) * invNormal;
                float alpha = 1.0 - gamma - beta;
                float3 a1 = v1.B + v1.BtoA, a2 = v2.B + v2.BtoA, a3 = v3.B + v3.BtoA;
                pointA = alpha * a1 + beta * a2 + gamma * a3;
                pointB = alpha * v1.B + beta * v2.B + gamma * v3.B;
                if (alpha < 0.0 || beta < 0.0 || gamma < 0.0)
                {
                    float3 clamped = float3(max(alpha, 0.0), max(beta, 0.0), max(gamma, 0.0));
                    clamped /= clamped.x + clamped.y + clamped.z;
                    float3 offsetA = pointA - (clamped.x * a1 + clamped.y * a2 + clamped.z * a3);
                    float3 offsetB = pointB - (clamped.x * v1.B + clamped.y * v2.B + clamped.z * v3.B);
                    if (max(LengthSq(offsetA), LengthSq(offsetB)) > collideEps * collideEps)
                    {
                        nextSeed = PortalRetrySeed(v0, v1, v2, v3, normal, seed);
                        valid = false;
                        return hit;
                    }
                }
                if (delta * delta > collideEps * collideEps * normalSq)
                {
                    valid = false;
                    return hit;
                }
                if (penetration > retryDepth + collideEps) nextSeed = PortalRetrySeed(v0, v1, v2, v3, normal, seed);
            }
            return hit;
        }
        temp1 = cross(v4.BtoA, v0.BtoA);
        float d1 = dot(temp1, v1.BtoA);
        if (d1 >= 0.0)
        {
            if (dot(temp1, v2.BtoA) >= 0.0) v1 = v4;
            else v3 = v4;
        }
        else
        {
            if (dot(temp1, v3.BtoA) >= 0.0) v2 = v4;
            else v1 = v4;
        }
    }
    return hit;
}

// solve_mpr_raycast: returns whether the physical contact is resolved.
bool MprRaycast(Geom a, Geom b, float4 qB, float3 pB, float extend, float collideEps, inout float3 pointA,
                inout float3 pointB, inout float3 normal, inout float penetration)
{
    float3 direction = WarpNormalize(normal);
    if (LengthSq(direction) == 0.0) direction = float3(1, 0, 0);
    Vert support = MinkowskiSupport(a, b, direction, qB, pB, 0.0, true);
    pointA = support.B + support.BtoA;
    pointB = support.B;
    Vert corner = MinkowskiSupport(a, b, direction, qB, pB, 0.0, false);
    float tolerance = max(1.0, max(length(corner.B + corner.BtoA), length(corner.B)));
    normal = direction;
    penetration = dot(support.BtoA, direction) + extend;
    bool resolved = false;
    pointA += normal * (0.5 * extend);
    pointB -= normal * (0.5 * extend);
    float accuracy = collideEps * tolerance;
    [loop]
    for (int optimization = 0; optimization < 4; optimization++)
    {
        float depthBefore = 1.0e30;
        if (resolved) depthBefore = penetration;
        support = MinkowskiSupport(a, b, direction, qB, pB, 0.0, true);
        float translation = dot(support.BtoA, direction) + 1.0e-5 * tolerance;
        float3 gradient = direction;
        bool accepted = false;
        [loop]
        for (int rayStep = 0; rayStep < 30; rayStep++)
        {
            float3 pa, pb, separatingNormal;
            float gap;
            bool separated = GjkCore(a, b, qB, pB + translation * direction, 0.0, 30, 1.0e-7 * tolerance, 0.0, pa, pb,
                                     separatingNormal, gap);
            if (!separated || !isfinite(gap) || gap < 0.0) break;
            pb = pb - translation * direction;
            float3 physicalDelta = pa - pb;
            float physicalLength = length(physicalDelta);
            float physicalDepth = physicalLength;
            if (dot(physicalDelta, separatingNormal) < 0.0) physicalDepth = -physicalDepth;
            float depth = physicalDepth + extend;
            float3 candidate = separatingNormal;
            if (physicalLength > 1.0e-7 * tolerance) candidate = physicalDelta / physicalDepth;
            Vert certificate = MinkowskiSupport(a, b, candidate, qB, pB, extend, false);
            float supportGap = dot(certificate.BtoA, candidate) - depth;
            float rayDepth = dot(physicalDelta, direction);
            float3 tangentError = physicalDelta - rayDepth * direction;
            Vert rayCertificate = MinkowskiSupport(a, b, direction, qB, pB, 0.0, false);
            float rayGap = dot(rayCertificate.BtoA, direction) - rayDepth;
            float roundoff = 2.0e-7 * tolerance;
            if (length(tangentError) <= roundoff && abs(rayGap) <= collideEps + roundoff)
            {
                candidate = direction;
                physicalDepth = rayDepth;
                physicalLength = abs(rayDepth);
                depth = rayDepth + extend;
                supportGap = rayGap;
                float3 center = 0.5 * (pa + pb);
                pa = center + 0.5 * rayDepth * direction;
                pb = center - 0.5 * rayDepth * direction;
            }
            if (isfinite(depth) && depth >= 0.0 && physicalLength <= abs(penetration - extend) + accuracy &&
                abs(LengthSq(separatingNormal) - 1.0) <= 1.0e-4 && abs(supportGap) <= accuracy)
            {
                normal = candidate;
                float3 correction = normal * (0.5 * extend);
                pointA = pa + correction;
                pointB = pb - correction;
                penetration = depth;
                resolved = true;
                accepted = true;
            }
            if (gap > 2.0e-6 * tolerance) gradient = separatingNormal;
            if (accepted) break;
            if (gap < 5.0e-7 * tolerance) break;
            float slope = dot(separatingNormal, direction);
            if (slope <= 1.0e-8) break;
            translation -= max(gap - 2.0e-7 * tolerance, 0.0) / slope;
        }
        if (dot(direction, gradient) > 1.0 - 1.0e-7) break;
        if (resolved && depthBefore - penetration < accuracy) break;
        direction = gradient;
    }
    if (!resolved)
    {
        float3 pa, pb, physicalNormal, physicalSeed;
        float physicalDepth;
        bool physicalValid;
        bool physicalHit = MprPortal(a, b, qB, pB, 0.0, 0, 30, 1.0e-5, penetration - extend, pa, pb, physicalNormal,
                                     physicalDepth, physicalSeed, physicalValid);
        [loop]
        for (int retry = 0; retry < 8; retry++)
        {
            bool withinBound = physicalDepth <= penetration - extend + collideEps * tolerance;
            if (!physicalHit || (physicalValid && withinBound) || LengthSq(physicalSeed) == 0.0) break;
            if (physicalValid) physicalSeed = -normal * length(physicalSeed);
            physicalHit = MprPortal(a, b, qB, pB, 0.0, physicalSeed, 30, 1.0e-5, penetration - extend, pa, pb,
                                    physicalNormal, physicalDepth, physicalSeed, physicalValid);
        }
        if (physicalHit && physicalValid && physicalDepth <= penetration - extend + collideEps * tolerance)
        {
            normal = physicalNormal;
            pointA = pa + normal * (0.5 * extend);
            pointB = pb - normal * (0.5 * extend);
            penetration = physicalDepth + extend;
            resolved = true;
        }
    }
    return resolved;
}

bool BoxPair(Geom a, Geom b) { return a.type == GEO_BOX && b.type == GEO_BOX; }

// solve_mpr_refine on the fused path (no ready certificate); returns whether the contact is resolved.
bool MprRefine(Geom a, Geom b, float4 qB, float3 pB, float extend, float3 seed, inout float3 pointA, inout float3 pointB,
               inout float3 normal, inout float penetration, bool valid)
{
    const float COLLIDE_EPSILON = 1.0e-5;
    float bound = penetration + COLLIDE_EPSILON;
    float3 lastNormal = normal;
    bool retry = LengthSq(seed) > 0.0;
    float minimumDepth = -1.0;
    bool minimumVerified = false;
    float3 boundNormal = normal;
    float boundDepth = 1.0e30;
    if (BoxPair(a, b))
    {
        float3 minimumNormal;
        BoxBoxDepth(a.scale, b.scale, qB, pB, minimumNormal, minimumDepth);
        minimumVerified = true;
        if (minimumVerified && minimumDepth > COLLIDE_EPSILON)
        {
            if (!valid || abs(penetration - extend - minimumDepth) > COLLIDE_EPSILON)
            {
                seed = -0.5 * minimumDepth * minimumNormal;
                lastNormal = minimumNormal;
                bound = max(bound, minimumDepth + extend + COLLIDE_EPSILON);
                valid = false;
                retry = true;
            }
        }
    }
    else if (min(a.bounds.x, min(a.bounds.y, a.bounds.z)) > 0.0 && min(b.bounds.x, min(b.bounds.y, b.bounds.z)) > 0.0)
    {
        EnclosingBoxDepth(a.bounds, b.bounds, qB, pB, boundNormal, boundDepth);
        if (penetration > boundDepth + extend + COLLIDE_EPSILON)
        {
            normal = boundNormal;
            valid = false;
            retry = false;
        }
    }
    [loop]
    for (int attempt = 0; attempt < 3; attempt++)
    {
        if (valid || !retry) break;
        float3 nextSeed;
        bool hit = MprPortal(a, b, qB, pB, extend, seed, 30, COLLIDE_EPSILON, 1.0e30, pointA, pointB, normal, penetration,
                             nextSeed, valid);
        seed = nextSeed;
        retry = LengthSq(seed) > 0.0;
        bool wrongDepth = minimumVerified && minimumDepth > COLLIDE_EPSILON &&
                          abs(penetration - extend - minimumDepth) > COLLIDE_EPSILON;
        if (hit && penetration > boundDepth + extend + COLLIDE_EPSILON)
        {
            normal = boundNormal;
            valid = false;
            break;
        }
        if (!hit || penetration > bound || wrongDepth)
        {
            normal = lastNormal;
            valid = false;
            break;
        }
        lastNormal = normal;
    }
    if (!valid) valid = MprRaycast(a, b, qB, pB, extend, COLLIDE_EPSILON, pointA, pointB, normal, penetration);
    return valid;
}

// solve_mpr_separated
bool MprSeparated(Geom a, Geom b, float4 qB, float3 pB, float extend, out float3 pointA, out float3 pointB,
                  out float3 normal, out float depth)
{
    float scale = max(1.0, length(pB));
    float gap;
    bool separated = GjkCore(a, b, qB, pB, 0.0, 30, 1.0e-7 * scale, 0.0, pointA, pointB, normal, gap);
    depth = extend - gap;
    if (!separated || abs(LengthSq(normal) - 1.0) > 1.0e-4) return false;
    Vert certificate = MinkowskiSupport(a, b, normal, qB, pB, 0.0, false);
    float tolerance = 1.0e-5 * max(1.0, max(length(pointA), length(pointB)));
    if (abs(dot(certificate.BtoA, normal) + gap) > tolerance) return false;
    float3 correction = normal * (0.5 * extend);
    pointA = pointA + correction;
    pointB = pointB - correction;
    return true;
}

// solve_mpr_core; returns `collision`.
bool MprCore(Geom a, Geom b, float4 qB, float3 pB, float extend, out float3 pointA, out float3 pointB, out float3 normal,
             out float penetration)
{
    const float COLLIDE_EPSILON = 1.0e-5;
    float3 seed;
    bool valid;
    bool collision = MprPortal(a, b, qB, pB, extend, 0, 30, COLLIDE_EPSILON, 1.0e30, pointA, pointB, normal, penetration,
                               seed, valid);
    bool needsRefine = false;
    if (collision)
    {
        if (!valid || BoxPair(a, b)) needsRefine = true;
        else if (min(a.bounds.x, min(a.bounds.y, a.bounds.z)) > 0.0 && min(b.bounds.x, min(b.bounds.y, b.bounds.z)) > 0.0)
        {
            float3 boundNormal;
            float boundDepth;
            EnclosingBoxDepth(a.bounds, b.bounds, qB, pB, boundNormal, boundDepth);
            needsRefine = penetration > boundDepth + extend + COLLIDE_EPSILON;
        }
    }
    if (needsRefine)
    {
        if (!valid && penetration <= extend)
        {
            float3 sa, sb, sn;
            float sd;
            if (MprSeparated(a, b, qB, pB, extend, sa, sb, sn, sd))
            {
                pointA = sa;
                pointB = sb;
                normal = sn;
                penetration = sd;
                return true;
            }
        }
        collision = MprRefine(a, b, qB, pB, extend, seed, pointA, pointB, normal, penetration, valid);
    }
    return collision;
}

// ---- Contact manifold (multicontact.py) ----

// B's polygon lives in gPoly[0..4] and may grow to ten points; A's is gPoly[5..9], as Newton aliases them.
static float2 gPoly[10];
static float3 gPointsA[5];
static float3 gPointsB[5];

struct PlaneTracker
{
    float3 referencePoint;
    float3 previousPoint;
    float3 normal;
    float largestAreaSq;
    bool nonplanar;
};

struct Projector
{
    float planeD;
    float3 normal;
};

void UpdatePlaneTracker(inout PlaneTracker t, float3 current, int id, bool checkPlanarity)
{
    if (id == 0)
    {
        t.referencePoint = current;
        t.largestAreaSq = 0.0;
        t.normal = 0;
        t.nonplanar = false;
        t.previousPoint = 0;
    }
    else if (id == 1)
        t.previousPoint = current;
    else
    {
        if (checkPlanarity && t.largestAreaSq > 0.0)
        {
            float offset = dot(t.normal, current - t.referencePoint);
            if (offset * offset > MANIFOLD_EPS * MANIFOLD_EPS * t.largestAreaSq) t.nonplanar = true;
        }
        float3 c = cross(t.previousPoint - t.referencePoint, current - t.referencePoint);
        float areaSq = dot(c, c);
        if (areaSq > t.largestAreaSq)
        {
            t.largestAreaSq = areaSq;
            t.normal = c;
        }
        t.previousPoint = current;
    }
}

bool FeatureSupportsAnchor(PlaneTracker t, float3 anchor)
{
    float offset = dot(t.normal, anchor - t.referencePoint);
    return !t.nonplanar && offset * offset <= MANIFOLD_EPS * MANIFOLD_EPS * t.largestAreaSq;
}

float3 SegmentProjectorNormal(float3 segmentDir, float3 referenceNormal)
{
    float3 right = cross(segmentDir, referenceNormal);
    float3 normal = cross(right, segmentDir);
    float len = length(normal);
    return len > 1.0e-12 ? normal * (1.0 / len) : referenceNormal;
}

void CreateProjectors(PlaneTracker ta, float3 anchorA, PlaneTracker tb, float3 anchorB, float3 contactNormal,
                      out Projector pa, out Projector pb)
{
    pa.planeD = 0;
    pa.normal = 0;
    pb.planeD = 0;
    pb.normal = 0;
    if (ta.largestAreaSq == 0.0 && tb.largestAreaSq == 0.0)
    {
        float3 dirA = ta.previousPoint - ta.referencePoint;
        float3 dirB = tb.previousPoint - tb.referencePoint;
        float3 onPlaneA = 0.5 * (ta.referencePoint + ta.previousPoint);
        pa.normal = SegmentProjectorNormal(dirA, contactNormal);
        pa.planeD = -dot(onPlaneA, pa.normal);
        float3 onPlaneB = 0.5 * (tb.referencePoint + tb.previousPoint);
        pb.normal = SegmentProjectorNormal(dirB, contactNormal);
        pb.planeD = -dot(onPlaneB, pb.normal);
        return;
    }
    if (ta.largestAreaSq > 0.0)
    {
        float inv = 1.0 / sqrt(max(1.0e-12, ta.largestAreaSq));
        pa.normal = ta.normal * inv;
        pa.planeD = -dot(anchorA, pa.normal);
    }
    if (tb.largestAreaSq > 0.0)
    {
        float inv = 1.0 / sqrt(max(1.0e-12, tb.largestAreaSq));
        pb.normal = tb.normal * inv;
        pb.planeD = -dot(anchorB, pb.normal);
    }
    if (ta.largestAreaSq == 0.0)
    {
        float3 dir = ta.previousPoint - ta.referencePoint;
        float3 onPlane = 0.5 * (ta.referencePoint + ta.previousPoint);
        pa.normal = SegmentProjectorNormal(dir, pb.normal);
        pa.planeD = -dot(onPlane, pa.normal);
    }
    if (tb.largestAreaSq == 0.0)
    {
        float3 dir = tb.previousPoint - tb.referencePoint;
        float3 onPlane = 0.5 * (tb.referencePoint + tb.previousPoint);
        pb.normal = SegmentProjectorNormal(dir, pa.normal);
        pb.planeD = -dot(onPlane, pb.normal);
    }
}

float3 ProjectOnto(Projector p, float3 input, float3 contactNormal)
{
    float denom = dot(contactNormal, p.normal);
    if (abs(denom) < 1.0e-12) return input;
    float t = -(dot(input, p.normal) + p.planeD) / denom;
    return input + contactNormal * t;
}

float SignedArea(float2 a, float2 b, float2 q) { return (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x); }

float2 IntersectionPoint(float2 s, float2 e, float2 a, float2 b)
{
    float signedA = SignedArea(s, e, a);
    float signedB = SignedArea(s, e, b);
    float interp = abs(signedA) / abs(signedA - signedB);
    return (1.0 - interp) * a + interp * b;
}

void InsertPoly(int arrCount, int index, float2 element)
{
    int i = arrCount;
    while (i > index)
    {
        gPoly[i] = gPoly[i - 1];
        i--;
    }
    gPoly[index] = element;
}

// trim_in_place on the loop at gPoly[0..].
int TrimInPlace(float2 s, float2 e, int loopCount)
{
    if (loopCount < 3) return loopCount;
    float2 intersectionA = 0, intersectionB = 0;
    int changeA = -1, changeB = -1;
    bool keep = false;
    bool prevOutside = SignedArea(s, e, gPoly[0]) <= 0.0;
    [loop]
    for (int i = 0; i < loopCount; i++)
    {
        int next = i + 1 == loopCount ? 0 : i + 1;
        bool outside = SignedArea(s, e, gPoly[next]) <= 0.0;
        if (outside != prevOutside)
        {
            float2 inter = IntersectionPoint(s, e, gPoly[i], gPoly[next]);
            if (changeA < 0)
            {
                changeA = i;
                keep = !prevOutside;
                intersectionA = inter;
            }
            else
            {
                changeB = i;
                intersectionB = inter;
            }
        }
        prevOutside = outside;
    }
    int newLoopCount;
    if (changeA >= 0 && changeB >= 0)
    {
        int loopIndexer = -1;
        newLoopCount = loopCount;
        int i2 = 0;
        [loop]
        while (i2 < loopCount)
        {
            if (keep)
            {
                loopIndexer++;
                gPoly[loopIndexer] = gPoly[i2];
            }
            if (i2 == changeA || i2 == changeB)
            {
                float2 pt = i2 == changeA ? intersectionA : intersectionB;
                if (loopIndexer == i2 && !keep)
                {
                    loopIndexer++;
                    InsertPoly(newLoopCount, loopIndexer, pt);
                    newLoopCount++;
                    i2++;
                    changeB++;
                    loopCount++;
                }
                else
                {
                    loopIndexer++;
                    gPoly[loopIndexer] = pt;
                }
                keep = !keep;
            }
            i2++;
        }
        newLoopCount = loopIndexer + 1;
    }
    else if (prevOutside)
        newLoopCount = 0;
    else
        newLoopCount = loopCount;
    return newLoopCount;
}

// trim_all_in_place: the trim polygon sits at gPoly[trimBase..], the loop at gPoly[0..].
int TrimAllInPlace(int trimBase, int trimCount, int loopCount)
{
    if (trimCount <= 1) return min(1, loopCount);
    if (trimCount == 2 || loopCount == 2)
    {
        float2 start = gPoly[0];
        float2 end = gPoly[1];
        if (trimCount == 2)
        {
            start = gPoly[trimBase];
            end = gPoly[trimBase + 1];
        }
        float2 direction = end - start;
        float lengthSq = dot(direction, direction);
        if (lengthSq <= MANIFOLD_EPS * MANIFOLD_EPS) return 0;
        float lo = 0.0, hi = 1.0;
        if (trimCount == 2 && loopCount == 2)
        {
            float2 otherStart = gPoly[0];
            float2 otherEnd = gPoly[1];
            float2 otherDirection = otherEnd - otherStart;
            float crossV = direction.x * otherDirection.y - direction.y * otherDirection.x;
            if (abs(crossV) > 1.0e-6 * sqrt(lengthSq * dot(otherDirection, otherDirection))) return 0;
            float t0 = dot(otherStart - start, direction) / lengthSq;
            float t1 = dot(otherEnd - start, direction) / lengthSq;
            lo = max(lo, min(t0, t1));
            hi = min(hi, max(t0, t1));
        }
        else
        {
            int polygonBase = trimBase;
            int polygonCount = trimCount;
            if (trimCount == 2)
            {
                polygonBase = 0;
                polygonCount = loopCount;
            }
            [loop]
            for (int i = 0; i < polygonCount; i++)
            {
                float2 edgeStart = gPoly[polygonBase + i];
                float2 edgeEnd = gPoly[polygonBase + (i + 1 == polygonCount ? 0 : i + 1)];
                float a = SignedArea(edgeStart, edgeEnd, start);
                float b = SignedArea(edgeStart, edgeEnd, end);
                if (a < 0.0 && b < 0.0) return 0;
                if (a < 0.0) lo = max(lo, a / (a - b));
                else if (b < 0.0) hi = min(hi, a / (a - b));
            }
        }
        if (hi <= lo) return 0;
        gPoly[0] = start + lo * direction;
        gPoly[1] = start + hi * direction;
        return 2;
    }
    int currentLoopCount = loopCount;
    float2 trim0 = gPoly[trimBase];
    [loop]
    for (int i = 0; i < trimCount; i++)
    {
        float2 s = gPoly[trimBase + i];
        float2 e = i == trimCount - 1 ? trim0 : gPoly[trimBase + i + 1];
        currentLoopCount = TrimInPlace(s, e, currentLoopCount);
    }
    return currentLoopCount;
}

// approx_max_quadrilateral_area_with_calipers on gPoly[0..n).
int4 MaxQuadrilateral(int n)
{
    int p1 = 0, p3 = 1;
    float2 d0 = gPoly[p1] - gPoly[p3];
    float maxDistSq = d0.x * d0.x + d0.y * d0.y;
    const float tieEpsilonRel = 1.0e-3;
    int j = 1;
    [loop]
    for (int i = 0; i < n; i++)
    {
        int i1 = i + 1 == n ? 0 : i + 1;
        float2 hullI = gPoly[i];
        float2 hullI1 = gPoly[i1];
        [loop]
        while (true)
        {
            int j1 = j + 1 == n ? 0 : j + 1;
            float areaJ1 = SignedArea(hullI, hullI1, gPoly[j1]);
            float areaJ = SignedArea(hullI, hullI1, gPoly[j]);
            if (areaJ1 > areaJ) j = j1;
            else break;
        }
        float2 hj = gPoly[j];
        float2 d1 = hullI - hj;
        float distSq1 = d1.x * d1.x + d1.y * d1.y;
        if (distSq1 > maxDistSq * (1.0 + tieEpsilonRel))
        {
            maxDistSq = distSq1;
            p1 = i;
            p3 = j;
        }
        float2 d2 = hullI1 - hj;
        float distSq2 = d2.x * d2.x + d2.y * d2.y;
        if (distSq2 > maxDistSq * (1.0 + tieEpsilonRel))
        {
            maxDistSq = distSq2;
            p1 = i1;
            p3 = j;
        }
    }
    int p2 = 0, p4 = 0;
    float maxArea1 = 0.0, maxArea2 = 0.0;
    float2 hp1 = gPoly[p1], hp3 = gPoly[p3];
    [loop]
    for (int k = 0; k < n; k++)
    {
        float area = SignedArea(hp1, hp3, gPoly[k]);
        if (area > maxArea1 * (1.0 + tieEpsilonRel))
        {
            maxArea1 = area;
            p2 = k;
        }
        else if (-area > maxArea2 * (1.0 + tieEpsilonRel))
        {
            maxArea2 = -area;
            p4 = k;
        }
    }
    return int4(p1, p2, p3, p4);
}

float PolygonWidth(int base, int count)
{
    if (count < 3) return 0.0;
    float2 lower = gPoly[base], upper = gPoly[base];
    float twiceArea = 0.0;
    [loop]
    for (int i = 0; i < count; i++)
    {
        float2 p = gPoly[base + i];
        float2 q = gPoly[base + (i + 1 == count ? 0 : i + 1)];
        lower = min(lower, p);
        upper = max(upper, p);
        twiceArea += p.x * q.y - q.x * p.y;
    }
    float extent = max(upper.x - lower.x, upper.y - lower.y);
    if (extent <= 0.0) return 0.0;
    return 0.5 * abs(twiceArea) / extent;
}

int RemoveZeroLengthEdges(int count, float eps)
{
    if (count < 2) return 0;
    int write = 0;
    [loop]
    for (int read = 1; read < count; read++)
    {
        float2 diff = gPoly[read] - gPoly[write];
        if (dot(diff, diff) > eps * eps)
        {
            write++;
            gPoly[write] = gPoly[read];
        }
    }
    int newCount;
    if (write > 0)
    {
        float2 diff = gPoly[write] - gPoly[0];
        newCount = dot(diff, diff) < eps * eps ? write : write + 1;
    }
    else
        newCount = write + 1;
    if (newCount < 2) newCount = 0;
    return newCount;
}

// add_avoid_duplicates_vec2 on the polygon at gPoly[base..]; returns whether the point was added.
bool AddAvoidDuplicates(int base, inout int count, float2 vec, float eps)
{
    if (count > 0)
    {
        float2 d = gPoly[base] - vec;
        if (dot(d, d) < eps * eps) return false;
    }
    if (count > 1)
    {
        float2 d = gPoly[base + count - 1] - vec;
        if (dot(d, d) < eps * eps) return false;
    }
    gPoly[base + count] = vec;
    count++;
    return true;
}

float3 SupportContactFeature(Geom g, float3 direction, float3 normal)
{
    float3 surfacePoint = SupportMap(g, direction);
    if (g.type == GEO_CAPSULE && abs(normal.z) <= 1.0e-4)
    {
        float3 radial = WarpNormalize(float3(normal.x, normal.y, 0.0)) * g.scale.x;
        surfacePoint.x = radial.x;
        surfacePoint.y = radial.y;
        surfacePoint.z = direction.z >= 0.0 ? g.scale.y : -g.scale.y;
    }
    return surfacePoint;
}

bool ContactPlaneIsSupporting(Geom g, float3 direction, float planeD)
{
    float3 surfacePoint = SupportMap(g, direction);
    float offset = dot(direction, surfacePoint) + planeD;
    return offset * offset <= MANIFOLD_EPS * MANIFOLD_EPS * LengthSq(direction);
}

bool FeaturePlaneIsSupporting(Geom g, Projector projector, float3 normal, bool onB, float4 relQ, float3 relP)
{
    float3 direction = projector.normal;
    float planeD = projector.planeD;
    float alignment = dot(direction, normal);
    if ((onB && alignment > 0.0) || (!onB && alignment < 0.0))
    {
        direction = -direction;
        planeD = -planeD;
    }
    if (onB)
    {
        planeD += dot(direction, relP);
        direction = QuatRotateInv(relQ, direction);
    }
    return ContactPlaneIsSupporting(g, direction, planeD);
}

bool ExcessNormalDeviation(float3 a, float3 b) { return abs(dot(a, b)) < COS_TILT_ANGLE; }

bool PointInContactPolygon(float2 surfacePoint, int base, int count)
{
    float side = 0.0;
    bool inside = count >= 3;
    [loop]
    for (int k = 0; k < count; k++)
    {
        float2 start = gPoly[base + k];
        float2 edge = gPoly[base + (k + 1 == count ? 0 : k + 1)] - start;
        float crossV = edge.x * (surfacePoint.y - start.y) - edge.y * (surfacePoint.x - start.x);
        if (crossV * crossV > MANIFOLD_EPS * MANIFOLD_EPS * dot(edge, edge))
        {
            if (side == 0.0) side = sign(crossV);
            else if (crossV * side < 0.0) inside = false;
        }
    }
    return inside;
}

// The writer side of one manifold point: post_process_minkowski_only for spheres and capsules, then write_contact.
void WriteManifoldContact(ContactData tmpl, int typeA, int typeB, float3 centerWorld, float3 normalWorld, float distance)
{
    ContactData c = tmpl;
    c.center = centerWorld;
    c.normal = normalWorld;
    c.distance = distance;
    if (typeA == GEO_SPHERE || typeA == GEO_CAPSULE)
    {
        c.center = c.center + normalWorld * (c.radiusA * 0.5);
        c.distance = c.distance - c.radiusA;
    }
    if (typeB == GEO_SPHERE || typeB == GEO_CAPSULE)
    {
        c.center = c.center - normalWorld * (c.radiusB * 0.5);
        c.distance = c.distance - c.radiusB;
    }
    WriteContact(c, -1);
}

// write_vertex_face_contacts; the vertex feature is shape A's when vertexOnA, its polygon at gPoly[vertexBase..].
int WriteVertexFaceContacts(bool vertexOnA, int vertexCount, int vertexBase, float3 vertexAnchor, Projector faceProjector,
                            int faceBase, int faceCount, float3 normalLocal, float4 orientationA, float3 positionAWorld,
                            float3 normalWorld, ContactData tmpl, int typeA, int typeB)
{
    float distances[5];
    int candidates = 0;
    [loop]
    for (int i = 0; i < 5; i++)
    {
        distances[i] = 1.0e30;
        if (i < vertexCount)
        {
            float3 vertex = vertexOnA ? gPointsA[i] : gPointsB[i];
            float3 dv = vertex - vertexAnchor;
            if (dot(dv, dv) > MANIFOLD_EPS * MANIFOLD_EPS && PointInContactPolygon(gPoly[vertexBase + i], faceBase, faceCount))
            {
                float3 onFace = ProjectOnto(faceProjector, vertex, normalLocal);
                float signedDistance = dot(onFace - vertex, normalLocal);
                if (!vertexOnA) signedDistance = -signedDistance;
                distances[i] = signedDistance;
                candidates++;
            }
        }
    }
    if (candidates > 4)
    {
        int shallowest = 0;
        [loop]
        for (int k = 1; k < 5; k++)
            if (distances[k] > distances[shallowest]) shallowest = k;
        distances[shallowest] = 1.0e30;
    }
    int written = 0;
    [loop]
    for (int m = 0; m < 5; m++)
    {
        if (distances[m] < 1.0e30)
        {
            float3 vertex = vertexOnA ? gPointsA[m] : gPointsB[m];
            float3 onFace = ProjectOnto(faceProjector, vertex, normalLocal);
            float3 pointLocal = 0.5 * (vertex + onFace);
            float3 pointWorld = QuatRotate(orientationA, pointLocal) + positionAWorld;
            WriteManifoldContact(tmpl, typeA, typeB, pointWorld, normalWorld, distances[m]);
            written++;
        }
    }
    return written;
}

// build_manifold; all work in A's frame. Returns the number of contacts written.
int BuildManifold(Geom a, Geom b, float4 orientationA, float3 positionAWorld, float4 relQ, float3 relP, float3 pA,
                  float3 pB, float3 normal, ContactData tmpl)
{
    const float PENT_COS[5] = { 1.0, 0.30901699437494745, -0.8090169943749473, -0.8090169943749476, 0.30901699437494723 };
    const float PENT_SIN[5] = { 0.0, 0.9510565162951535, 0.5877852522924732, -0.5877852522924734, -0.9510565162951536 };
    int aCount = 0, bCount = 0;
    float3 tangentA, tangentB;
    OrthonormalBasis(normal, tangentA, tangentB);
    PlaneTracker trackerA, trackerB;
    trackerA = (PlaneTracker)0;
    trackerB = (PlaneTracker)0;
    bool planarA = a.type == GEO_BOX;
    bool planarB = b.type == GEO_BOX;
    float3 center = 0.5 * (pA + pB);
    [loop]
    for (int k = 0; k < 10; k++) gPoly[k] = 0;
    float3 localNormalB = QuatRotateInv(relQ, -normal);
    float3 localTaB = QuatRotateInv(relQ, -tangentA);
    float3 localTbB = QuatRotateInv(relQ, -tangentB);
    [loop]
    for (int e = 0; e < 5; e++)
    {
        float c = PENT_COS[e], s = PENT_SIN[e];
        float cSin = c * SIN_TILT_ANGLE, sSin = s * SIN_TILT_ANGLE;
        float3 dirA = normal * COS_TILT_ANGLE + cSin * tangentA + sSin * tangentB;
        float3 ptA3 = SupportContactFeature(a, dirA, normal);
        float3 projectedA = ptA3 - center;
        float2 ptA2 = float2(dot(tangentA, projectedA), dot(tangentB, projectedA));
        if (AddAvoidDuplicates(5, aCount, ptA2, MANIFOLD_EPS))
        {
            gPointsA[aCount - 1] = ptA3;
            UpdatePlaneTracker(trackerA, ptA3, aCount - 1, planarA);
        }
        float3 localDirB = localNormalB * COS_TILT_ANGLE + cSin * localTaB + sSin * localTbB;
        float3 ptBLocal = SupportContactFeature(b, localDirB, localNormalB);
        float3 ptB3 = QuatRotate(relQ, ptBLocal) + relP;
        float3 projectedB = ptB3 - center;
        float2 ptB2 = float2(dot(tangentA, projectedB), dot(tangentB, projectedB));
        if (AddAvoidDuplicates(0, bCount, ptB2, MANIFOLD_EPS))
        {
            gPointsB[bCount - 1] = ptB3;
            UpdatePlaneTracker(trackerB, ptB3, bCount - 1, planarB);
        }
    }
    float3 normalWorld = QuatRotate(orientationA, normal);
    bool featureAFlat = !planarA || FeatureSupportsAnchor(trackerA, pA);
    bool featureBFlat = !planarB || FeatureSupportsAnchor(trackerB, pB);
    int countOut = 0;
    float normalDot = 0.0;
    if (aCount < 2 || bCount < 2 || !featureAFlat || !featureBFlat)
    {
        if (planarA && planarB && featureAFlat != featureBFlat)
        {
            bool faceOnB = featureBFlat;
            PlaneTracker faceTracker = trackerA;
            if (faceOnB) faceTracker = trackerB;
            float3 faceAnchor = faceOnB ? pB : pA;
            if (faceTracker.largestAreaSq > 0.0)
            {
                Projector faceProjector;
                faceProjector.normal = faceTracker.normal * (1.0 / sqrt(faceTracker.largestAreaSq));
                faceProjector.planeD = -dot(faceAnchor, faceProjector.normal);
                bool faceSupported = !ExcessNormalDeviation(normal, faceProjector.normal);
                if (faceSupported)
                {
                    Geom faceGeom = a;
                    if (faceOnB) faceGeom = b;
                    faceSupported = FeaturePlaneIsSupporting(faceGeom, faceProjector, normal, faceOnB, relQ, relP);
                }
                if (faceSupported)
                {
                    // The vertex feature is the other shape's: A's when the face is on B.
                    countOut = WriteVertexFaceContacts(faceOnB, faceOnB ? aCount : bCount, faceOnB ? 5 : 0,
                                                       faceOnB ? pA : pB, faceProjector, faceOnB ? 0 : 5,
                                                       faceOnB ? bCount : aCount, normal, orientationA, positionAWorld,
                                                       normalWorld, tmpl, a.type, b.type);
                }
            }
        }
    }
    else
    {
        Projector projectorA, projectorB;
        CreateProjectors(trackerA, pA, trackerB, pB, normal, projectorA, projectorB);
        bool validPlanes = !ExcessNormalDeviation(normal, projectorA.normal) && !ExcessNormalDeviation(normal, projectorB.normal);
        if (validPlanes && planarA && (trackerA.largestAreaSq == 0.0 || LengthSq(cross(normal, projectorA.normal)) > 0.0))
            validPlanes = FeaturePlaneIsSupporting(a, projectorA, normal, false, relQ, relP);
        if (validPlanes && planarB && (trackerB.largestAreaSq == 0.0 || LengthSq(cross(normal, projectorB.normal)) > 0.0))
            validPlanes = FeaturePlaneIsSupporting(b, projectorB, normal, true, relQ, relP);
        if (validPlanes)
        {
            // extract_4_point_contact_manifolds
            normalDot = abs(dot(projectorA.normal, projectorB.normal));
            float mergeDistance = clamp(0.5 * min(PolygonWidth(5, aCount), PolygonWidth(0, bCount)), MANIFOLD_EPS,
                                        MAX_CLIP_VERTEX_MERGE_DISTANCE);
            int loopCount = TrimAllInPlace(5, aCount, bCount);
            loopCount = RemoveZeroLengthEdges(loopCount, mergeDistance);
            if (loopCount > 1)
            {
                int4 result = int4(0, 1, 2, 3);
                if (loopCount > 4)
                {
                    result = MaxQuadrilateral(loopCount);
                    loopCount = 4;
                }
                [loop]
                for (int i = 0; i < loopCount; i++)
                {
                    int ia = result[i];
                    float3 pLocal = gPoly[ia].x * tangentA + gPoly[ia].y * tangentB + center;
                    float3 pa = ProjectOnto(projectorA, pLocal, normal);
                    float3 pb = ProjectOnto(projectorB, pLocal, normal);
                    float3 pointLocal = 0.5 * (pa + pb);
                    float signedDistance = dot(pb - pa, normal);
                    float3 pointWorld = QuatRotate(orientationA, pointLocal) + positionAWorld;
                    WriteManifoldContact(tmpl, a.type, b.type, pointWorld, normalWorld, signedDistance);
                }
                countOut = min(loopCount, 4);
            }
            else
            {
                normalDot = 0.0;
                countOut = 0;
            }
        }
    }
    if (normalDot < COS_DEEPEST_CONTACT_THRESHOLD_ANGLE || countOut == 0)
    {
        float3 deepestCenter = QuatRotate(orientationA, 0.5 * (pA + pB)) + positionAWorld;
        float deepestDistance = dot(pB - pA, normal);
        WriteManifoldContact(tmpl, a.type, b.type, deepestCenter, normalWorld, deepestDistance);
        countOut++;
    }
    return countOut;
}

// ---- Fused solve (collision_convex.py solve_convex_multi_contact) ----

float GjkSeparationCutoff(float contactThreshold, float3 positionA, float3 positionB)
{
    if (contactThreshold <= 0.0) return 0.0;
    return contactThreshold + GJK_CUTOFF_TOLERANCE * (CoordinateScale(positionA) + CoordinateScale(positionB));
}

void SolveConvexMultiContact(Geom a, Geom b, float4 orientationA, float4 orientationB, float3 positionA, float3 positionB,
                             float contactThreshold, bool skipMultiContact, ContactData tmpl)
{
    float4 relQ = QuatMul(QuatInverse(orientationA), orientationB);
    float3 relP = QuatRotateInv(orientationA, positionB - positionA);
    float marginSum = tmpl.marginA + tmpl.marginB;
    const float eps = 1.0e-4;
    float enlarge = 0.0;
    if (marginSum <= 0.0) enlarge = eps;
    else if (marginSum < eps) enlarge = 2.0 * eps;

    float3 pointA, pointB, normal;
    float penetration, signedDistance;
    bool collision = MprCore(a, b, relQ, relP, enlarge, pointA, pointB, normal, penetration);
    if (collision)
    {
        signedDistance = -penetration + enlarge;
        float halfEnlarge = enlarge * 0.5;
        pointA = pointA - normal * halfEnlarge;
        pointB = pointB + normal * halfEnlarge;
    }
    else
        GjkCore(a, b, relQ, relP, 0.0, 30, 1.0e-4, GjkSeparationCutoff(contactThreshold, positionA, positionB), pointA,
                pointB, normal, signedDistance);

    if (skipMultiContact || signedDistance > contactThreshold)
    {
        float3 surfacePoint = QuatRotate(orientationA, 0.5 * (pointA + pointB)) + positionA;
        WriteManifoldContact(tmpl, a.type, b.type, surfacePoint, QuatRotate(orientationA, normal), signedDistance);
        return;
    }
    BuildManifold(a, b, orientationA, positionA, relQ, relP, pointA, pointB, normal, tmpl);
}

#endif
