# Canary Index Execution Report — 20260816T145616-6b31766b

**Status**: failed
**Git commit**: e05926798a15d64d3b07e24921d7430bbd84a7ce
**Manifest ID**: canary-42-test
**Manifest checksum**: 2607ea9c67df21a8cfb45425f8c8e6687950986115638342739e92ff438aa953
**Contract fingerprint**: a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa
**Index**: msmarco-xi / pilot_v1

## Timing
- Start: 2026-08-16T14:56:16.964738+00:00
- End: 2026-08-16T14:56:16.982302+00:00
- Duration: 0.02s

## Throughput
- Records: 300
- Tokens: 1,500
- Records/second: unknown
- Tokens/minute: unknown

## Batches
- Batch size: 96
- Concurrency: 4
- Total batches: 4
- Completed: 0
- Skipped (resumed): 0
- Failed: 0
- Total attempts: 0
- Retries: 0
- Throttle waits (s): 0.0

## Validation
- Resume used: True
- Remote index validation: not run
- Freshness reconciliation: not run

## Failure
- Category: CorruptCheckpoint
- Message: Checkpoint file /private/var/folders/j9/15l2rcp96cd0_3kkfb_rzrww0000gn/T/pytest-of-suvra/pytest-26/test_corrupt_checkpoint_causes0/checkpoints/canary_canary-42-test_BAD.json exists but cannot be read or parsed: Expecting value: line 1 column 1 (char 0). Delete or repair the checkpoint file before retrying.
- Safe next action: Delete or repair /private/var/folders/j9/15l2rcp96cd0_3kkfb_rzrww0000gn/T/pytest-of-suvra/pytest-26/test_corrupt_checkpoint_causes0/checkpoints/canary_canary-42-test_BAD.json before retrying. Starting fresh without --resume is also safe.
