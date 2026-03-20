# Tunneling - Detailed Derivation

## Rectangular Barrier Setup
```
Region 1 (x < -a/2):  V = 0
Region 2 (-a/2 < x < +a/2):  V = eV_0
Region 3 (x > +a/2):  V = 0
```

## Wave Functions
- ψ_1 = A_1 e^(ik_0x) + B_1 e^(-ik_0x)
- ψ_2 = A_2 e^(k_1x) + B_2 e^(-k_1x)
- ψ_3 = A_3 e^(ik_0x)

## Continuity Conditions
At x = -a/2:
- ψ_1 = ψ_2
- dψ_1/dx = dψ_2/dx

At x = +a/2:
- ψ_2 = ψ_3
- dψ_2/dx = dψ_3/dx

## Transmission Probability Derivation
After solving boundary conditions:
```
T_e = |A_3/A_1|² = 1 / [1 + ((k_0² + k_1²)² / (4k_0²k_1²)) × sinh²(k_1a)]
```

## For Thick Barriers (k_1a >> 1)
```
sinh²(k_1a) ≈ (1/4)exp(2k_1a)
T_e ≈ 16(k_0k_1/(k_0² + k_1²))²exp(-2k_1a)
```

## For High Barriers (eV_0 >> E)
```
k_1 ≈ sqrt(2meV_0)/ħ
k_0 ≈ sqrt(2mE)/ħ
T_e ≈ 16(E/eV_0)(1 - E/eV_0)exp(-2k_1a)
```