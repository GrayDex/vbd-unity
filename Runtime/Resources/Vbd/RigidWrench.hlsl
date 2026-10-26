// Force, torque and the 6x6 Hessian blocks a body gathers in its solve (world frame, about the center of mass).
#ifndef RIGID_WRENCH
#define RIGID_WRENCH

#include "RigidVbdMath.hlsl"

struct Wrench { float3 f; float3 t; float3x3 hll; float3x3 hal; float3x3 haa; };

Wrench ZeroWrench()
{
    Wrench w;
    w.f = 0; w.t = 0; w.hll = 0; w.hal = 0; w.haa = 0;
    return w;
}

Wrench AddWrench(Wrench a, Wrench b)
{
    a.f += b.f; a.t += b.t; a.hll += b.hll; a.hal += b.hal; a.haa += b.haa;
    return a;
}

// A force and linear Hessian applied at offset r from the center of mass (_point_force_to_body_torque_and_hessian).
Wrench PointForce(float3 r, float3 force, float3x3 hLinear)
{
    float3x3 rx = Skew(r);
    Wrench w;
    w.f = force;
    w.t = cross(r, force);
    w.hll = hLinear;
    w.hal = mul(rx, hLinear);
    w.haa = mul(mul(transpose(rx), hLinear), rx);
    return w;
}

#endif
