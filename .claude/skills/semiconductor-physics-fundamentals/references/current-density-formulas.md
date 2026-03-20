# Current Density Formula Details

## Dark Saturation Current Density (J0)

**Formula:**
$$J_0 = q \left[ \frac{n_i^2 D_n}{N_A L_n} + \frac{n_i^2 D_p}{N_D L_p} \right]$$

**Variables:**
- $q$: Elementary charge (1.602 × 10⁻¹⁹ C)
- $n_i$: Intrinsic carrier density
- $D_n, D_p$: Diffusion coefficients for electrons and holes
- $N_A, N_D$: Acceptor and donor densities
- $L_n, L_p$: Diffusion lengths

**Diffusion Lengths:**
$$L_n = \sqrt{\tau_n D_n}$$
$$L_p = \sqrt{\tau_p D_p}$$

Where $\tau_n, \tau_p$ are minority carrier lifetimes.

## Short Circuit Current Density (JSC)

**Theoretical Maximum:**
$$J_{SC,max} = q \int_{E_g}^{\infty} \phi_0(E_\gamma) \cdot \eta(E_\gamma) \, dE_\gamma$$

**Variables:**
- $\phi_0$: Incident photon flux density (photons per unit area per unit time per energy)
- $\eta$: External quantum efficiency
- $E_\gamma$: Photon energy
- $E_g$: Material bandgap

**Perfect Absorber Approximation:**
For $\eta(E_\gamma) = 1$ when $E_\gamma \geq E_g$ and $\eta(E_\gamma) = 0$ otherwise:
$$J_{SC,max} = q \int_{E_g}^{\infty} \phi_0(E_\gamma) \, dE_\gamma$$

**Uniform Generation Rate Conversion:**
$$g_0 = \frac{j_{SC,max}}{d_{DR}}$$

Where:
- $j_{SC,max} = J_{SC,max} / q$ (carrier flux density)
- $d_{DR}$: Depletion region thickness

## Depletion Width Calculations

**n-type region width:**
$$w_n = \sqrt{\frac{2\epsilon \psi_{bi}}{q N_D} \cdot \frac{N_A}{N_A + N_D}}$$

**p-type region width:**
$$w_p = \sqrt{\frac{2\epsilon \psi_{bi}}{q N_A} \cdot \frac{N_D}{N_A + N_D}}$$

**Variables:**
- $\epsilon$: Permittivity
- $\psi_{bi}$: Built-in potential
- $q$: Elementary charge
- $N_A, N_D$: Acceptor and donor densities