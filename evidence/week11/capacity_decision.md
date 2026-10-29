# Capacity on the Week 11 VM: finding and decision (2026-10-28)

## Finding

Source: `capacity_recheck.txt` and `capacity_followup.txt` (5 repeats per rate, c1 and c2).

| | capacity-v3 (old VM) | this VM |
|---|---|---|
| 140 flows/s, CPU p95 | 91.3-92.9 % | 100.7-100.9 % |
| 140 flows/s, setup p95 | 46 ms | 244-284 ms |
| 160 / 200 flows/s, QoS success | 1 / 1 | 0 / 0 |
| 200 flows/s, processed rate | 2033-2034 pps | 1593-1607 pps |
| 80 flows/s, CPU p95 / setup p95 | 68-70 % / 28-30 ms | 81.1 % / 36.7-37.9 ms |
| 100 flows/s, CPU p95 / setup p95 | 78.7 % / 34-36 ms | 94.3-95.1 % / 48.0-49.1 ms |
| 120 flows/s, CPU p95 | 85.3 % | 100.2-100.7 % |

- The code of the capacity-v3 measurement (commit a188beb) gives the same result at
  140 flows/s as the frozen code (tag w11-collect-freeze), so the difference comes from
  the VM, not from the code.
- capacity-v3 health rules (QoS 100 %, process CPU p95 < 95 %, setup p95 < 2x the
  low-load baseline): 80 flows/s passes; 100 flows/s is borderline (c1 CPU p95 95.06 %,
  c2 94.28 %); 120 flows/s and above fail. The low-load baseline (30 flows/s) was not
  re-measured on this VM, so the setup rule is only checked against the old baseline.

## Effect on migration-outcome-raw-v1

- Every raw snapshot keeps `processed_packet_in_rate` (pps) and `safe_capacity_pps`;
  `utilization` = pps / 1433 (capacity-v3). The raw data does not change and U can be
  recomputed with another capacity.
- "No destination overload" (destination U after migration 0.474-0.902) holds in
  capacity-v3 units only.
- Indicative only (not a capacity derivation: no lower confidence bound, no full
  sweep): with the processed mean at 100 flows/s (about 1033 pps) as capacity, U
  scales by 1.387 and destination U after migration has median 1.07, max 1.25, with
  160 of 300 MIGRATE attempts at U >= 1; at 80 flows/s (about 833 pps) the factor is 1.72.

## Decision (Vy and My, to confirm with the advisor)

1. Week 11 closes with migration-outcome-raw-v1 as collected.
   `data/benchmarks/controller_capacity.json` is not edited: capacity-v3 stays the record
   of the old VM.
2. The Data Card states the U scale (capacity-v3) and this finding.
3. First task of Week 12 (Vy): capacity-v4 on this VM with the capacity-v3 protocol
   (fine sweep including the 30 flows/s baseline, 5 repeats, both controllers), saved
   as a new version. The processed outcome dataset of Week 12 computes U from pps with
   capacity-v4; the raw data stays as it is.
4. To ask the advisor: whether the Week 10 forecaster (trained on U in capacity-v3
   units) is rescaled or retrained, and whether the attempts above the capacity-v4
   limit are used as the overload region instead of a separate stress dataset.

## Outcome (2026-10-29, Week 12)

- capacity-v4 (c_safe 933 pps: c1 932.9, c2 933.1) was measured on this VM with the
  capacity-v3 rule (80 runs, 80-200 flows/s, 5 repeats, both controllers; the 30-70
  flows/s points are kept from capacity-v3). The rate where the capacity-v3 rule still
  holds is 90 flows/s (about 933 pps).
- CPU in the raw v1 snapshots confirms the saturation: only 61 of 340 attempts have both
  controllers below CPU p95 95% in every phase; 220 are saturated in both phases. Because
  U is censored in the saturated zone, rescaling v1 with capacity-v4 does not recover the
  true load. The matrix was therefore re-collected as migration-outcome-raw-v2
  (capacity-v4; 320 of 340 healthy). v1 is not edited.
- Measured with capacity-v4, 10 of 300 COMMITTED v2 attempts end with
  dst_u_after_mean >= 1.0 (hot_switch high/peak, destination 0.75). The 160-of-300
  figure above is an extrapolation from saturated v1 data and is superseded by this.
