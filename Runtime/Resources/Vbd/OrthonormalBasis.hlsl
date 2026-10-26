// Newton's orthonormal_basis (newton/_src/math): two unit tangents of a unit normal.
#ifndef VBD_ORTHONORMAL_BASIS
#define VBD_ORTHONORMAL_BASIS

void OrthonormalBasis(float3 n, out float3 b1, out float3 b2)
{
    if (n.z < 0.0)
    {
        float a = 1.0 / (1.0 - n.z);
        float b = n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
    }
    else
    {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
}

#endif
