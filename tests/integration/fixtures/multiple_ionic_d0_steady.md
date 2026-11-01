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
local temporary storage. They use the historical coordinates as an unverified
seed for one current-source steady solve per module, under the original gates.
The resulting new evidence has its own source identity and digest, which the
production loader checks before four physical time steps. No archived identity,
accepted status or field is rewritten.

This fixture replaces a machine-specific cloud path that failed on CI run
40788604559. Run 40789506628 then exposed the checkout-path component of the
configuration source identity; the explicit seeded solve handles that difference
without weakening production admission. Both original failures and the archived
accepted evidence are preserved.
