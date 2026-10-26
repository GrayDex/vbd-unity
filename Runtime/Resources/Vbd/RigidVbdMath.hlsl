// Warp's quaternion, transform and matrix helpers that rigid VBD relies on, with Warp's exact formulas
// (quaternions are (x, y, z, w); HLSL float3x3 is indexed [row][column] like Warp's mat33).
#ifndef RIGID_VBD_MATH
#define RIGID_VBD_MATH

#define PI_F 3.14159274
#define HALF_PI_F 1.57079637

struct Transform { float3 p; float4 q; };
struct Mat33 { float3 r0; float3 r1; float3 r2; };

float3x3 Rows(Mat33 m) { return float3x3(m.r0, m.r1, m.r2); }
float3x3 FromCols(float3 a, float3 b, float3 c) { return transpose(float3x3(a, b, c)); }
float3x3 Outer(float3 a, float3 b) { return float3x3(a.x * b, a.y * b, a.z * b); }
float3x3 Skew(float3 v) { return float3x3(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0); }
float3x3 Identity3() { return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1); }
float Trace(float3x3 m) { return m[0][0] + m[1][1] + m[2][2]; }

// atan2 to a few ulp (Cephes atanf), instead of the compiler's approximation: Newton's float32 CUDA atan2f is
// near exact, and rotation angles enter every rotation-vector residual and angular velocity.
float Atan2(float y, float x)
{
    float ax = abs(x), ay = abs(y);
    float hi = max(ax, ay), lo = min(ax, ay);
    float t = hi > 0.0 ? lo / hi : 0.0;
    float base = 0.0;
    if (t > 0.41421356)
    {
        base = 0.25 * PI_F;
        t = (t - 1.0) / (t + 1.0);
    }
    float z = t * t;
    float r = base + ((((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1) * z * t + t);
    if (ay > ax) r = HALF_PI_F - r;
    if (x < 0.0) r = PI_F - r;
    return y < 0.0 ? -r : r;
}

// Warp's normalize: zero stays zero instead of NaN.
float3 Normalize3(float3 a)
{
    float l = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return l > 0.0 ? a / l : float3(0, 0, 0);
}

float4 QNormalize(float4 q)
{
    float l = sqrt(dot(q, q));
    if (l > 0.0) return q * (1.0 / l);
    return float4(0, 0, 0, 1);
}

float4 QInverse(float4 q) { return float4(-q.xyz, q.w); }

float4 QMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
        a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
        a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 QRotate(float4 q, float3 x)
{
    float c = 2.0 * q.w * q.w - 1.0;
    float d = 2.0 * (q.x * x.x + q.y * x.y + q.z * x.z);
    return float3(
        x.x * c + q.x * d + (q.y * x.z - q.z * x.y) * q.w * 2.0,
        x.y * c + q.y * d + (q.z * x.x - q.x * x.z) * q.w * 2.0,
        x.z * c + q.z * d + (q.x * x.y - q.y * x.x) * q.w * 2.0);
}

float3 QRotateInv(float4 q, float3 x)
{
    float c = 2.0 * q.w * q.w - 1.0;
    float d = 2.0 * (q.x * x.x + q.y * x.y + q.z * x.z);
    return float3(
        x.x * c + q.x * d - (q.y * x.z - q.z * x.y) * q.w * 2.0,
        x.y * c + q.y * d - (q.z * x.x - q.x * x.z) * q.w * 2.0,
        x.z * c + q.z * d - (q.x * x.y - q.y * x.x) * q.w * 2.0);
}

float4 QFromAxisAngle(float3 axis, float angle)
{
    float half_ = angle * 0.5;
    return float4(axis * sin(half_), cos(half_));
}

void QToAxisAngle(float4 q, out float3 axis, out float angle)
{
    float3 v = q.xyz;
    axis = q.w < 0.0 ? -Normalize3(v) : Normalize3(v);
    angle = 2.0 * Atan2(sqrt(dot(v, v)), abs(q.w));
}

float3x3 QToMatrix(float4 q)
{
    return FromCols(QRotate(q, float3(1, 0, 0)), QRotate(q, float3(0, 1, 0)), QRotate(q, float3(0, 0, 1)));
}

float4 QFromMatrix(float3x3 m)
{
    float tr = m[0][0] + m[1][1] + m[2][2];
    float x = 0, y = 0, z = 0, w = 0, h = 0;
    if (tr >= 0.0)
    {
        h = sqrt(tr + 1.0);
        w = 0.5 * h;
        h = 0.5 / h;
        x = (m[2][1] - m[1][2]) * h;
        y = (m[0][2] - m[2][0]) * h;
        z = (m[1][0] - m[0][1]) * h;
    }
    else
    {
        int maxDiag = 0;
        if (m[1][1] > m[0][0]) maxDiag = 1;
        if (m[2][2] > m[maxDiag][maxDiag]) maxDiag = 2;
        if (maxDiag == 0)
        {
            h = sqrt((m[0][0] - (m[1][1] + m[2][2])) + 1.0);
            x = 0.5 * h;
            h = 0.5 / h;
            y = (m[0][1] + m[1][0]) * h;
            z = (m[2][0] + m[0][2]) * h;
            w = (m[2][1] - m[1][2]) * h;
        }
        else if (maxDiag == 1)
        {
            h = sqrt((m[1][1] - (m[2][2] + m[0][0])) + 1.0);
            y = 0.5 * h;
            h = 0.5 / h;
            z = (m[1][2] + m[2][1]) * h;
            x = (m[0][1] + m[1][0]) * h;
            w = (m[0][2] - m[2][0]) * h;
        }
        else
        {
            h = sqrt((m[2][2] - (m[0][0] + m[1][1])) + 1.0);
            z = 0.5 * h;
            h = 0.5 / h;
            x = (m[2][0] + m[0][2]) * h;
            y = (m[1][2] + m[2][1]) * h;
            w = (m[1][0] - m[0][1]) * h;
        }
    }
    return QNormalize(float4(x, y, z, w));
}

Transform MakeTransform(float3 p, float4 q)
{
    Transform t;
    t.p = p;
    t.q = q;
    return t;
}

Transform TransformIdentity() { return MakeTransform(float3(0, 0, 0), float4(0, 0, 0, 1)); }
Transform TMul(Transform a, Transform b) { return MakeTransform(QRotate(a.q, b.p) + a.p, QMul(a.q, b.q)); }
float3 TPoint(Transform t, float3 x) { return t.p + QRotate(t.q, x); }

// C roundf (half away from zero), as Warp's round; HLSL round() may round half to even.
float RoundHalfAway(float x) { return sign(x) * floor(abs(x) + 0.5); }
// Warp's sign: 1 at zero.
float WarpSign(float x) { return x < 0.0 ? -1.0 : 1.0; }

// Newton's quat_velocity: world angular velocity from successive orientations.
float3 QuatVelocity(float4 qNow, float4 qPrev, float dt)
{
    float4 q1 = QNormalize(qNow);
    float4 q0 = QNormalize(qPrev);
    if (dot(q1, q0) < 0.0) q0 = -q0;
    float4 dq = QNormalize(QMul(q1, QInverse(q0)));
    float3 axis;
    float angle;
    QToAxisAngle(dq, axis, angle);
    return axis * (angle / dt);
}

// Newton's _wrap_angle_pm_pi and quat_decompose (Warp's quat_to_euler with axes 2, 1, 0).
float WrapAnglePmPi(float theta)
{
    float twoPi = 2.0 * PI_F;
    float wrapped = fmod(theta + PI_F, twoPi);
    if (wrapped < 0.0) wrapped += twoPi;
    return wrapped - PI_F;
}

float3 QuatDecompose(float4 q)
{
    float a = q.w - q.y, b = q.z - q.x, c = q.y + q.w, d = -q.x - q.z;
    // acos(2(a²+b²)/(a²+b²+c²+d²) − 1) written through atan2, which is well conditioned everywhere.
    float t2 = 2.0 * Atan2(sqrt(c * c + d * d), sqrt(a * a + b * b));
    float tp = Atan2(b, a);
    float tm = Atan2(d, c);
    float t1 = 0.0, t3 = 0.0;
    if (abs(t2) < 1e-6) t3 = 2.0 * tp - t1;
    else if (abs(t2 - PI_F) < 1e-6) t3 = 2.0 * tm + t1;
    else
    {
        t1 = tp - tm;
        t3 = tp + tm;
    }
    t2 -= HALF_PI_F;
    t3 = -t3;
    return float3(WrapAnglePmPi(t3), WrapAnglePmPi(t2), WrapAnglePmPi(t1));
}

// Newton's ldlt6_solve: the 6×6 SPD block system [[h_ll, h_alᵀ], [h_al, h_aa]] by LDLᵀ.
void Ldlt6Solve(float3x3 hll, float3x3 haa, float3x3 hal, float3 rl, float3 ra, out float3 xl, out float3 xa)
{
    float A11 = hll[0][0], A21 = hll[1][0], A22 = hll[1][1], A31 = hll[2][0], A32 = hll[2][1], A33 = hll[2][2];
    float A41 = hal[0][0], A42 = hal[0][1], A43 = hal[0][2], A44 = haa[0][0];
    float A51 = hal[1][0], A52 = hal[1][1], A53 = hal[1][2], A54 = haa[1][0], A55 = haa[1][1];
    float A61 = hal[2][0], A62 = hal[2][1], A63 = hal[2][2], A64 = haa[2][0], A65 = haa[2][1], A66 = haa[2][2];

    float L21 = A21 / A11, L31 = A31 / A11, L41 = A41 / A11, L51 = A51 / A11, L61 = A61 / A11;
    float D2 = A22 - L21 * L21 * A11;
    float L32 = (A32 - L21 * L31 * A11) / D2;
    float L42 = (A42 - L21 * L41 * A11) / D2;
    float L52 = (A52 - L21 * L51 * A11) / D2;
    float L62 = (A62 - L21 * L61 * A11) / D2;
    float D3 = A33 - (L31 * L31 * A11 + L32 * L32 * D2);
    float L43 = (A43 - L31 * L41 * A11 - L32 * L42 * D2) / D3;
    float L53 = (A53 - L31 * L51 * A11 - L32 * L52 * D2) / D3;
    float L63 = (A63 - L31 * L61 * A11 - L32 * L62 * D2) / D3;
    float D4 = A44 - (L41 * L41 * A11 + L42 * L42 * D2 + L43 * L43 * D3);
    float L54 = (A54 - L41 * L51 * A11 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    float L64 = (A64 - L41 * L61 * A11 - L42 * L62 * D2 - L43 * L63 * D3) / D4;
    float D5 = A55 - (L51 * L51 * A11 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    float L65 = (A65 - L51 * L61 * A11 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;
    float D6 = A66 - (L61 * L61 * A11 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

    float y1 = rl.x;
    float y2 = rl.y - L21 * y1;
    float y3 = rl.z - L31 * y1 - L32 * y2;
    float y4 = ra.x - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = ra.y - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = ra.z - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    float z1 = y1 / A11, z2 = y2 / D2, z3 = y3 / D3, z4 = y4 / D4, z5 = y5 / D5, z6 = y6 / D6;

    float x6 = z6;
    float x5 = z5 - L65 * x6;
    float x4 = z4 - L54 * x5 - L64 * x6;
    float x3 = z3 - L43 * x4 - L53 * x5 - L63 * x6;
    float x2 = z2 - L32 * x3 - L42 * x4 - L52 * x5 - L62 * x6;
    float x1 = z1 - L21 * x2 - L31 * x3 - L41 * x4 - L51 * x5 - L61 * x6;
    xl = float3(x1, x2, x3);
    xa = float3(x4, x5, x6);
}

#endif
