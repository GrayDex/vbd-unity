// Newton's _compute_body_particle_contact_force: quadratic penalty (no log barrier), damping while approaching,
// regularized isotropic Coulomb friction. Force on the particle; the body takes the opposite.
// u is the particle's displacement over the step relative to the body surface; mat is stiffness, damping, friction.
#ifndef VBD_CONTACT_FORCE
#define VBD_CONTACT_FORCE

void BodyParticleContactForce(float distance, float collisionRadius, float3 n, float3 u, float3 mat,
                              float frictionEpsilon, float dt, out float3 force, out float3x3 hessian)
{
    force = 0;
    hessian = 0;
    float penetration = collisionRadius - distance;
    if (penetration <= 0.0) return;
    float fn = mat.x * penetration;
    float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
    force = n * fn;
    hessian = mat.x * nn;
    if (dot(n, u) < 0.0)
    {
        float3x3 damping = (mat.y / dt) * nn;
        hessian += damping;
        force -= mul(damping, u);
    }
    float3 ut = u - n * dot(n, u);
    float uNorm = length(ut);
    if (uNorm > 0.0)
    {
        float eps = frictionEpsilon * dt;
        float scale = uNorm > eps ? 1.0 / uNorm : eps > 0.0 ? (-uNorm / eps + 2.0) / eps : 0.0;
        scale *= mat.z * fn;
        force -= scale * ut;
        hessian += scale * (float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - nn);
    }
}

#endif
