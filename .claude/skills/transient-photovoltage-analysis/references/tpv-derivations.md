# TPV Analytical Model Derivations

## Voltage-Carrier Density Relationship

Starting from the Boltzmann approximation for carrier statistics:

$$n = n_i e^{(E_F - E_i)/kT}$$

For small perturbations around steady-state:

$$\frac{\Delta n}{n_{OC}} = \frac{\Delta E_F}{kT}$$

Since open-circuit voltage is related to Fermi level splitting:

$$\Delta V_{OC} = \frac{\Delta E_F}{q} = \frac{kT}{q} \cdot \frac{\Delta n}{n_{OC}}$$

## Kinetic Model for Carrier Density

**Rate Equation:**
$$\frac{d(\Delta n)}{dt} = \Delta g - k_{TPV} \cdot \Delta n$$

**Solution During Pulse ($t < t_{pulse}$):**
With initial condition $\Delta n(0) = 0$:

$$\Delta n(t) = \frac{\Delta g}{k_{TPV}} (1 - e^{-k_{TPV} t})$$

**Solution After Pulse ($t \geq t_{pulse}$):**
With initial condition $\Delta n(t_{pulse})$ from above:

$$\Delta n(t) = \Delta n(t_{pulse}) e^{-k_{TPV}(t - t_{pulse})}$$

## Generation Rate Calculation

**AM1.5G Integration:**
$$g_0 = \int_{E_g}^{\infty} \phi_{AM1.5G}(E) \, dE$$

For $E_g = 1.6$ eV:
$$g_0 \approx 1.89 \times 10^{21} \text{ cm}^{-3} \text{ s}^{-1}$$

**Pulse Generation Rate:**
$$\Delta g = 0.2 \times g_0$$

(20% of bias light intensity)