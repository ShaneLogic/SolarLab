# Mixed immobile-ion steady fixture

`multiple_ionic_d0_steady.json.gz` contains the exact, independently accepted
`MultipleIonDomainsV1/ImmobileSteady/MixedSteady.json` research evidence.
Its decompressed SHA-256 is
`52fbe8cad99b739698459daaf428cba17acf5a8a70ea19de8a4fd9c4852651aa`.
Compression changes no source identities, solver controls, fields or outcomes.

The 13-node device has two separated shared-site ion domains. The right domain's
negative species is immobile, while its positive species and both left-domain
species are mobile. The steady state passed its original numerical gates at
2 mV in the dark. Final scientific qualification remains false.

The transient tests verify the original digest and materialize these bytes in
local temporary storage. They validate the saved device/profile binding before
running the four physical time steps; they never regenerate the steady initial.
This fixture replaces a machine-specific cloud path that failed on CI run
40788604559. The original failure and archived evidence are preserved.
