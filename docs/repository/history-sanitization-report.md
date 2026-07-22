# Git history artifact sanitization report

Date: 2026-07-22

Verdict at the rewritten source checkpoint: history sanitation passed.

> **Existing clones must not push their old history.** Prefer a fresh clone.
> Pre-rewrite commits and tags are deliberately incompatible with the active
> remote refs.

## Outcome

The controlled rewrite removed machine-generated build output, cache metadata,
raw benchmark-run output, profiler captures, and an agent-local context file
from every active branch and tag. Source history was preserved.

| Checkpoint | Commit | Tree |
|---|---|---|
| Old `main` | `f4fac8f60cd308ec665b9e071cddc6147ea8a31d` | `90e04d986d1b9c2621d3a77d69fa85624663c0b3` |
| Rewritten equivalent | `926df63a44c8fc6515ac3dd4035442b7b20bd1cb` | `90e04d986d1b9c2621d3a77d69fa85624663c0b3` |

The trees are byte-identical. The documentation commit containing this report
intentionally advances `main` after that source-tree identity gate.

The rewrite mapped all 292 commits: 225 received new IDs and 67 were already
free of the approved objects and kept their IDs. No commit was dropped. For
every mapped pair, the non-purge path, file mode, and blob map is identical;
parent topology, author, committer, dates, and message are identical.

Seven signed commit objects necessarily lost their invalidated `gpgsig`
headers: `0be66c39028b96049ae54488822aad9fa7f0980c`,
`412cbc83edd93761327357b2a39b40a244be3da9`,
`5865fdc2ddce2622d62418595e8ee99553d39944`,
`993d1d3544bc69838f54747ea867ba4a54725dd0`,
`ada133d508825e24cba1f052fb1e2e5bfed9e627`,
`ee6875d2228019ccafa4c1c10aae5282d507fda2`, and
`f4fac8f60cd308ec665b9e071cddc6147ea8a31d`. This is expected when the
signed payload changes; no source or other commit metadata changed.

## Recovery material

Recovery material is intentionally outside the repository and must never be
pushed to `onlyxItachi/MatcoreDSL`.

| Artifact | Location | SHA-256 |
|---|---|---|
| Pre-rewrite network mirror | `/home/hamza-usta/archives/MatcoreDSL-pre-purge-20260722T111636Z.git` | Mirror inventory file hash `63a3593504e7226213f7c7fde06672f18ea85c4b5c3d231c0ee47433554a3f44` |
| Complete pre-rewrite bundle | `/home/hamza-usta/archives/MatcoreDSL-pre-purge-20260722T111636Z.bundle` | `7475e0550971c5cbdacf31adc5e48371c86e7939aef4da10dd91c4302fb7eb35` |
| Sanitized 31-head bundle | `/home/hamza-usta/archives/MatcoreDSL-post-purge-local-heads-20260722T111636Z.bundle` | `69ea1f85149a9c6f7b6eb73a33b4749fcbab15548b62ee67edb19fd86800154c` |
| Complete external audit | `/home/hamza-usta/archives/MatcoreDSL-history-purge-20260722T111636Z/` | Manifest checksum file hash `e125388f4a9c3fa6a19e115eccc3afa114727e2933118782827ce8a90a38b393` |

The old checkout is preserved at
`/home/hamza-usta/MatcoreDSL-pre-rewrite-checkout-20260722T111636Z`.
Its linked worktrees were repaired after the move and its `origin` push URL is
set to `disabled://pre-rewrite-history-do-not-push`.

Both pre-rewrite recovery forms passed `git fsck --full --strict` or
`git bundle verify`; the sanitized bundle also passed `git bundle verify`.

## Filter operation

The disposable mirror was populated with the 10 remote heads and 21 local-only
heads, while GitHub-managed `refs/pull/*` and the local stash were excluded.
The reviewed command was first run with `--dry-run`, then applied unchanged:

```sh
PYTHONPATH=/home/hamza-usta/.local/lib/python3.12/site-packages \
git filter-repo \
  --force \
  --invert-paths \
  --paths-from-file purge-paths.txt \
  --strip-blobs-with-ids purge-blob-ids.txt \
  --prune-empty never \
  --prune-degenerate never \
  --preserve-commit-hashes
```

The manifests contain 56 exact paths and 62 unique blob IDs. No broad
extension wildcard was used. No purge blob was shared by a preserved path.

## Exact purge manifest

Each `blob:size` value is a Git blob ID and its uncompressed byte count.

| Historical path | Blob IDs and sizes |
|---|---|
| `.matcore_cache_reviewmeta/c1aae08ae61f577fddcc0afaaa70e250/metadata.json` | `a53b603d4bb0919d449871aab48d3a67c64d7160:226` |
| `benchmark_reports/complete_20260423_182039/absurd_activation.stderr.log` | `5ba4c9923bccc2cecf24752b1985e447ce695a33:5352` |
| `benchmark_reports/complete_20260423_182039/absurd_activation.stdout.log` | `82c7a70b5cf60906156210187e465942ad48fc83:2376` |
| `benchmark_reports/complete_20260423_182039/attention.stderr.log` | `4d8e99b963cdb73244689df823e9583b9832d509:840` |
| `benchmark_reports/complete_20260423_182039/attention.stdout.log` | `df5c748fb25f204abd58cb9883de72a46c11141b:424` |
| `benchmark_reports/complete_20260423_182039/core_suite.stderr.log` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391:0` |
| `benchmark_reports/complete_20260423_182039/core_suite.stdout.log` | `9e1ae85167e3d95804a5ce81792555343b376e3d:874` |
| `benchmark_reports/complete_20260423_182039/device_resident.stderr.log` | `d075565eff3132649bbb833298c900b8d2896b3d:208` |
| `benchmark_reports/complete_20260423_182039/device_resident.stdout.log` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391:0` |
| `benchmark_reports/complete_20260423_182039/full_comparison.stderr.log` | `4c91ecbbabf792c7d4472781c29ecd3cf2d6df84:14824` |
| `benchmark_reports/complete_20260423_182039/full_comparison.stdout.log` | `e5a70516a5c00c67e7081150add09c09b5369b89:1765` |
| `benchmark_reports/complete_20260423_182039/fusion_devtensor.stderr.log` | `36b915cf83a7fd707dee768086a7f7bdf8bc8e92:9720` |
| `benchmark_reports/complete_20260423_182039/fusion_devtensor.stdout.log` | `dd8cf30e8922c9f654b593270039fdb131c7b339:2885` |
| `benchmark_reports/complete_20260423_182039/fusion_fair.stderr.log` | `47dd64b7c67c91cbbce9fc3e4065467d6820ef93:5532` |
| `benchmark_reports/complete_20260423_182039/fusion_fair.stdout.log` | `bd0df60878107f20050b76ce2bed940f3237c0ca:3691` |
| `benchmark_reports/complete_20260423_182039/fusion_suite.stderr.log` | `b0ccd7fc38d1eadc6585e98a181d9d52b47ea05f:5112` |
| `benchmark_reports/complete_20260423_182039/fusion_suite.stdout.log` | `749703a3e64b7da7d5635415372ba3b5e2b91e00:1927` |
| `benchmark_reports/complete_20260423_182039/graph_vs_torch.stderr.log` | `751f4aea274dfe6b38b97bcac1c677e65993d536:7492` |
| `benchmark_reports/complete_20260423_182039/graph_vs_torch.stdout.log` | `114497e7ba4b575439e3622c2d86d93310eccc88:1013` |
| `benchmark_reports/complete_20260423_182039/pow2_extended.stderr.log` | `401e2cf9885dad2c1ba061c98930dec83c44d4a1:15447` |
| `benchmark_reports/complete_20260423_182039/pow2_extended.stdout.log` | `41977536413f6dddfa6fc64c14a2c3d13f60c3b6:1262` |
| `benchmark_reports/complete_20260423_182039/rsqrt_sin_softmax.stderr.log` | `fc1acc1ae248c6c059129ff6853fa559deaaa4c9:5782` |
| `benchmark_reports/complete_20260423_182039/rsqrt_sin_softmax.stdout.log` | `3c45eb140b3d136e59f3a56ff499b92686351f2d:3466` |
| `benchmark_reports/complete_20260423_182039/summary.json` | `ea8794e40c2ac59cf4818cee86ce5ed242f12823:8537` |
| `benchmark_reports/complete_20260423_182039/summary.md` | `457b43f2c95c1e7aa8b96f93d4b5ccdf7f30ca22:3151` |
| `benchmark_reports/complete_20260423_182039/three_way.stderr.log` | `06e319d7ad674b58d4e6fda27e8f7d450726eba4:6053` |
| `benchmark_reports/complete_20260423_182039/three_way.stdout.log` | `562dee63680480a11cae6d3bf8fc39ebe8ae15c5:2533` |
| `benchmark_reports/family_c_attention_profile_20260426.md` | `5d0a3cd779d8daca88603bb88976a1667afe2103:6590` |
| `benchmark_reports/family_c_restore_20260429.md` | `82f3e9ee021c6a07ba865ed0e85e47871c11b37e:3802` |
| `build-review/.ninja_deps` | `018e3d0ffa3e741343859768342d5fa560e1c187:386824`, `3abc6b55d275e6b1448f9161df224dfafb2c2b3f:77936`, `6969388a8da3f2af9f2bbcc03ef5947d8f666e6e:180560` |
| `build-review/.ninja_log` | `0fb9c6bbde6f3fe793e3e84d6cdf0463d9c52d45:4174`, `35d16cbeb032166b49153135db2067d9f076e2c7:10371`, `7924214759048f2bcd126b4e619520d4366eccf5:582` |
| `build-review/CMakeCache.txt` | `54d27b1c107445df8a3d331d5b81b000c73a9c6d:24647`, `d0c5b7feb289a29ee11eb8f760c0cd4f64193d97:24642` |
| `build-review/CMakeFiles/4.2.3/CMakeCCompiler.cmake` | `895adc92dc146ec43e8ecffbfc292a142ea53748:2999` |
| `build-review/CMakeFiles/4.2.3/CMakeCXXCompiler.cmake` | `c34d763422b2a6dce2223834009066f41484cf47:6704` |
| `build-review/CMakeFiles/4.2.3/CMakeDetermineCompilerABI_C.bin` | `86ab74932fd9445a5544384e8724772ea0ada06f:16128` |
| `build-review/CMakeFiles/4.2.3/CMakeDetermineCompilerABI_CXX.bin` | `b7ea36652eb19cf12172dac5b3ab94bce4c64e81:16152` |
| `build-review/CMakeFiles/4.2.3/CMakeSystem.cmake` | `8490f96ce3e5584789c49043a061409290f59a6b:402` |
| `build-review/CMakeFiles/4.2.3/CompilerIdC/CMakeCCompilerId.c` | `ab3c3593124d26e674ba6afdb1efc3116955f93c:28668` |
| `build-review/CMakeFiles/4.2.3/CompilerIdC/a.out` | `f026ea824250331097975c3449a229666c6a2846:16192` |
| `build-review/CMakeFiles/4.2.3/CompilerIdCXX/CMakeCXXCompilerId.cpp` | `b35f567c2b995ac6939f38bb02d9c14ef707dde9:29480` |
| `build-review/CMakeFiles/4.2.3/CompilerIdCXX/a.out` | `d4fc4d57291d2b244fbac2aab349941ea5dc4cf6:16208` |
| `build-review/CMakeFiles/CMakeConfigureLog.yaml` | `b643e3173fec3e30fc6f4f71f978208b9a398986:371015` |
| `build-review/CMakeFiles/InstallScripts.json` | `b03072820739d06926c18f015c9e077d1a29af65:117` |
| `build-review/CMakeFiles/TargetDirectories.txt` | `fc7ea6ca854cb7e89f0630330e6ddfec0809aa4e:851` |
| `build-review/CMakeFiles/cmake.check_cache` | `3dccd731726d7faa8b29d8d7dba3b981a53ca497:85` |
| `build-review/CMakeFiles/rules.ninja` | `5cbec1b769252982ea5cdc0cfe829394c1c208f1:5263`, `68e927f8d84debd939eb3a1930dbc22cd1ce178c:5340` |
| `build-review/build.ninja` | `1f0348a6da621c56a4ab89ef48dd06541361c95e:66789`, `7ea63c1b4a12d20b5dcee153e06cf2e4841b311f:67315`, `ce5b82c0ce65969d6948f928edfd5d7e1bf46443:59863` |
| `build-review/cmake_install.cmake` | `cfc499572e79d5d1651387f07ccc1502fa2c3b60:2121`, `f32cea6beb3cf5adbe3caf47be890e38845b88c4:2114` |
| `custom-cache-root/c1aae08ae61f577fddcc0afaaa70e250/metadata.json` | `a53b603d4bb0919d449871aab48d3a67c64d7160:226` |
| `memory/2026-04-23-codex-context.md` | `55ef7e9216b8d78eb0f07734a86f886fb2dbb000:16184` |
| `profiling/nsys_fused_attention.nsys-rep` | `49aa737b09c2ba53dbfc88262b7dbb75a2925568:238539` |
| `profiling/nsys_fused_attention.sqlite` | `799d0b3b3167e1ef1a78347a85f9ec5f5316a67b:815104` |
| `profiling/results/ncu_output.log` | `63a5045b10f3c3fb2cc14cab500179424970420f:2349` |
| `profiling/results/ncu_splitk_output.log` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391:0` |
| `profiling/results/nsys_splitk.nsys-rep` | `2e08a9f35bea149b8b579a3c220ef51c7de65603:200974` |
| `profiling/results/nsys_splitk.sqlite` | `d3d392154c82d760d50260a18f7e1a4b832e30f9:712704` |

The 62 unique blobs total 3,530,280 uncompressed bytes. The largest were the
815,104-byte and 712,704-byte Nsight SQLite databases, the 386,824-byte Ninja
dependency database, the 371,015-byte CMake configure log, and the 238,539-byte
and 200,974-byte Nsight report files.

## Preserved material

The rewrite explicitly preserved:

- legacy `include/`, `src/`, `matcore/`, root CMake, Python/JIT/MLIR/CUDA, and
  profiling-script history;
- the complete `compiler/` native frontend, typed IR v1, CPU planner, runtime,
  packaging, examples, tests, fixtures, and goldens;
- `docs/adr/**`, `docs/mdslc/**`, and `docs/legacy/performance/**`;
- workflow and repository-hygiene source; and
- `benchmark_reports/complete_20260423_182039/notes.md`, blob
  `02583f059890f6631d8b17da69aa28cede4458ff`, as an ambiguous handwritten
  engineering note rather than machine output.

The two purged `benchmark_reports/family_c_*.md` paths have normalized semantic
equivalents under `docs/legacy/performance/`.

## Branch map

`remote` rows were atomically updated on GitHub. `local-only` rows are preserved
in the sanitized bundle and restored as local branches, but were not published.

| Ref | Old commit | Rewritten commit | Scope |
|---|---|---|---|
| `refs/heads/agent/repository-artifact-hygiene` | `d8b4a9f3eba2c4af273baa7111d2ab1ea864958b` | `61234fc387a9138022236f6f8bf15e44707a705d` | remote |
| `refs/heads/codex/phase6-wip-snapshot` | `e9b41b687aa989ed41a6f35b7348a9a0dc573c36` | `e9b41b687aa989ed41a6f35b7348a9a0dc573c36` | remote |
| `refs/heads/copilot/analyze-test-coverage` | `ee6875d2228019ccafa4c1c10aae5282d507fda2` | `3776d905014feadd11e2537a46dbe4166ac226e5` | remote |
| `refs/heads/copilot/perform-code-review-hpc-principles` | `f4780d59ff6dd778be550c2c293d817eabb7967a` | `31ae0fd35876d7672c0473a14b5d63fd5262f958` | remote |
| `refs/heads/feature/device-resident-tensors` | `351075e4d8af1880330b7c0474d701ca76776dfa` | `9549ffefb8fab1bd42e3d97b27134791247a096f` | remote |
| `refs/heads/main` | `f4fac8f60cd308ec665b9e071cddc6147ea8a31d` | `926df63a44c8fc6515ac3dd4035442b7b20bd1cb` | remote |
| `refs/heads/mdslc/audit-and-adr` | `d6a658103fb717eb9f0b8d6deea4f7c6a60acc0a` | `88d7314b16f21d367f8e6e36c3042e494815c12b` | local-only |
| `refs/heads/mdslc/bootstrap-v0` | `3e3fa5b2d1990e1c37870f8b2096fbda6128716b` | `6e4b4dd65cd64dee2e434f94739cba5c32e57ba1` | remote |
| `refs/heads/mdslc/codegen-wiring` | `ed4f27e7c4828afa3ebeaf6a10139e545b4f8552` | `5f1599ecd03b3afe2e34b3de8a5e179456eadaf3` | local-only |
| `refs/heads/mdslc/cpu-planner-runtime` | `2df0f7ae72720d9606f2a954708faa167c11d275` | `4575976c9d646e24de5dfa273b8c9f6cd727d093` | local-only |
| `refs/heads/mdslc/driver-build` | `c1e947a8e2540ae7bbdf50c6a0a5424d5b694563` | `fe31c7c87e5c05c164965c02f01a5cfdd0c576a1` | local-only |
| `refs/heads/mdslc/driver-pipeline-v0` | `89d72ba4796ec936a815b4a9482818fc4509a325` | `bd34f236f97fcf4fd3d90089d6dd96f19747f102` | local-only |
| `refs/heads/mdslc/final-review-v0` | `b8254c59eae9da7fb28f78e8ca7c2d376b5dae89` | `cc3367c7f42d9826b6c499a93d23388a84a9ee0b` | local-only |
| `refs/heads/mdslc/frontend-poc` | `8ddf2393ccfea5f52e2d8dcfaa9a9c83086cd059` | `3cc3ba7f6075bc608e9b586a049aac8ca5ba2f82` | local-only |
| `refs/heads/mdslc/frontend-rewrite-v0` | `2f160fd4d075206754d17a2d51ccb2e3dd899d97` | `1f22f83b03c9d23d2fac4c73380dd93cb0cff78a` | local-only |
| `refs/heads/mdslc/install-consumer-v0` | `db4e8ecc470ad7ab97c4b947c039e0f980270d83` | `6cace104017404e81ac1ead061b993c9fd28231d` | local-only |
| `refs/heads/mdslc/ir-v1-core` | `a13b874091f73a7b6927b53e387af61bb8054ffd` | `475ef40ddfaae5f58ee6ad354e8d7904174c33c7` | local-only |
| `refs/heads/mdslc/legacy-reuse` | `97362bfbda188574532f6621535b06b2ad2d2e83` | `1685e382c7736111fb08248b63bb92b69a736879` | local-only |
| `refs/heads/mdslc/mainline-integration-v2` | `1c87e793bcce25f4b49ec81ab0fd7c3a59fe74b5` | `e37dfdbf1934fa9b2ef113e5e8196fb145902cc6` | remote |
| `refs/heads/mdslc/matcore-ir-v1-cpu-planner` | `58d3e7c75eadf5a71416870ac4c99b4829db8566` | `f0889597adab6ffb6f635d6b2cae4eb1d2240113` | remote |
| `refs/heads/mdslc/milestone2-final-review` | `672819b3402d1d5f8a91f780484d2479595a10c7` | `0e390de396a2f693380c6df8452b9d246698b76a` | local-only |
| `refs/heads/mdslc/milestone2-validation` | `2c07cfe0b98ed50c3eef65d4694878a9afc7d255` | `53bbe29acb6835eb14e4201a4da9f8b64a17f77f` | local-only |
| `refs/heads/mdslc/native-driver-package-v1` | `2cd5b224ed33506b1bfdef4497fc9a294d86d214` | `d4d712834a9f13e79883c8c0934c877e4ffc532d` | local-only |
| `refs/heads/mdslc/native-final-review-v1` | `422dd6dc6c9c9fa7a6682f72da989c100adcee2a` | `c608b64966c20b465b0b047fd75cc85371218bbc` | local-only |
| `refs/heads/mdslc/native-frontend-v1` | `c6aefac7757aab702a3d5b7ad04a9a8562abf4c0` | `74e89a350abc9c6b34880329a4c8c5fdc40e365d` | local-only |
| `refs/heads/mdslc/native-libtooling-v1` | `993d1d3544bc69838f54747ea867ba4a54725dd0` | `626a2783baba5afcbde7822bc8dddd94017eacb9` | remote |
| `refs/heads/mdslc/native-validation-v1` | `ce7a1d7d899f9f1d5747f827d916089a25bb9074` | `68d203d2a9af1f4e6a67128ed4db79e1c35b8981` | local-only |
| `refs/heads/mdslc/runtime-cpu` | `feb2442cc97b05dafe5b814bbcad7a16acc25876` | `a1d4c78ba04a5ce71fe5bd91aee8b44f67f415a8` | local-only |
| `refs/heads/mdslc/validation` | `f0a409179c545b1a75f3d1c614e6ac3f804d4196` | `aa60c26bc5e930f89ca2c252f7d2f848732e97e0` | local-only |
| `refs/heads/mdslc/validation-v0-final` | `5374df32b8db0e7651f44da0475dcc9e9f0a68b8` | `76880829af624902d025d6987d066a2c8ac412dc` | local-only |
| `refs/heads/snapshot/fused-attention-v1` | `adb5e1c0e3e0679a7557fc09b756e3067dfac928` | `3bd03f4e83a912f08a9d057a760b58c6b4f4c4e0` | local-only |

## Tag and legacy-boundary map

| Tag | Old tag object / target | New tag object / target |
|---|---|---|
| `mdslc-native-cpu-proof-v1` | `4bb7be7be251e2f85cf662b1651351a5a5a7787c` / `c025df534d11d1bc08285a174f2cd357aecadb0e` | `c234ca05795a8cce33e23050a3c8765cdf4be12e` / `8e70d069e111cdf2d55ff0719c870f2bf5e4772b` |
| `mdslc-mainline-cpu-proof-v2` | `26fbcbd7ab48222e18cd6861e09819511d42cba0` / `5865fdc2ddce2622d62418595e8ee99553d39944` | `e06c605de8efb0136e7f4d9825dcffc72578ee0b` / `d9753d40bf23a45c77fdc43e09a1f62ad68d4d77` |
| `matcoredsl-legacy-final-v1` | absent | `02fcd877e1362e643ff8b3f4d3e6f702a3b46d41` / `9549ffefb8fab1bd42e3d97b27134791247a096f` |

The legacy boundary maps from
`351075e4d8af1880330b7c0474d701ca76776dfa` to
`9549ffefb8fab1bd42e3d97b27134791247a096f`. It remains the source-bearing
`Harden RegionV1 cache invalidation` commit, the final legacy-only checkpoint,
and the parent of the standalone MDSLC lineage.

Existing tag messages and tagger metadata are byte-identical except for their
mapped target object. No product release was created.

## Size evidence

For the complete 31-head plus original two-tag ref set:

| Metric | Before | After filtering | Delta |
|---|---:|---:|---:|
| Reachable blobs | 984 | 922 | -62 |
| Uncompressed blob bytes | 18,798,068 | 15,267,788 | -3,530,280 |
| Reachable objects | 2,414 | 2,340 | -74 |
| Uncompressed reachable bytes | 19,367,484 | 15,793,417 | -3,574,067 |

The new annotated legacy tag adds one tag object and no blob. The fresh network
mirror pack was 1,245,081 bytes; the sanitized 31-head pack is 1,187,826 bytes.
The complete pre-rewrite bundle is 1,629,326 bytes and the sanitized local-head
bundle is 1,190,823 bytes.

## Remote update

Immediately before mutation, every remote branch and tag matched the recorded
maintenance-window map. There were no open pull requests, no active workflow
runs, no Git process, and no active MatcoreDSL writer. PR #3 was closed and
unmerged. The backups, identical tree, tests, secret scan, and independent
review all passed.

The remote update was one atomic transaction. The existing refs used exact
old-object leases; the legacy tag was required to be absent:

```sh
git push --atomic \
  --force-with-lease=refs/heads/agent/repository-artifact-hygiene:d8b4a9f3eba2c4af273baa7111d2ab1ea864958b \
  --force-with-lease=refs/heads/copilot/analyze-test-coverage:ee6875d2228019ccafa4c1c10aae5282d507fda2 \
  --force-with-lease=refs/heads/copilot/perform-code-review-hpc-principles:f4780d59ff6dd778be550c2c293d817eabb7967a \
  --force-with-lease=refs/heads/feature/device-resident-tensors:351075e4d8af1880330b7c0474d701ca76776dfa \
  --force-with-lease=refs/heads/main:f4fac8f60cd308ec665b9e071cddc6147ea8a31d \
  --force-with-lease=refs/heads/mdslc/bootstrap-v0:3e3fa5b2d1990e1c37870f8b2096fbda6128716b \
  --force-with-lease=refs/heads/mdslc/mainline-integration-v2:1c87e793bcce25f4b49ec81ab0fd7c3a59fe74b5 \
  --force-with-lease=refs/heads/mdslc/matcore-ir-v1-cpu-planner:58d3e7c75eadf5a71416870ac4c99b4829db8566 \
  --force-with-lease=refs/heads/mdslc/native-libtooling-v1:993d1d3544bc69838f54747ea867ba4a54725dd0 \
  --force-with-lease=refs/tags/mdslc-mainline-cpu-proof-v2:26fbcbd7ab48222e18cd6861e09819511d42cba0 \
  --force-with-lease=refs/tags/mdslc-native-cpu-proof-v1:4bb7be7be251e2f85cf662b1651351a5a5a7787c \
  origin \
  refs/heads/agent/repository-artifact-hygiene:refs/heads/agent/repository-artifact-hygiene \
  refs/heads/copilot/analyze-test-coverage:refs/heads/copilot/analyze-test-coverage \
  refs/heads/copilot/perform-code-review-hpc-principles:refs/heads/copilot/perform-code-review-hpc-principles \
  refs/heads/feature/device-resident-tensors:refs/heads/feature/device-resident-tensors \
  refs/heads/main:refs/heads/main \
  refs/heads/mdslc/bootstrap-v0:refs/heads/mdslc/bootstrap-v0 \
  refs/heads/mdslc/mainline-integration-v2:refs/heads/mdslc/mainline-integration-v2 \
  refs/heads/mdslc/matcore-ir-v1-cpu-planner:refs/heads/mdslc/matcore-ir-v1-cpu-planner \
  refs/heads/mdslc/native-libtooling-v1:refs/heads/mdslc/native-libtooling-v1 \
  refs/tags/mdslc-mainline-cpu-proof-v2:refs/tags/mdslc-mainline-cpu-proof-v2 \
  refs/tags/mdslc-native-cpu-proof-v1:refs/tags/mdslc-native-cpu-proof-v1 \
  refs/tags/matcoredsl-legacy-final-v1:refs/tags/matcoredsl-legacy-final-v1
```

The same command passed with `--dry-run` first. The unchanged
`codex/phase6-wip-snapshot` branch required no update. `git push --mirror` was
not used, no remote branch was deleted, and no backup ref was published.

## Validation results

Pre-push validation on the disposable rewrite and a normal clone from it:

- `git fsck --full --strict`: passed;
- all 56 purge paths absent from every rewritten head/tag;
- all 62 purge blobs unreachable and physically absent;
- all 292 mapped commit trees and metadata exhaustively compared;
- all 1,089 pairwise ancestry relationships among mapped refs preserved;
- fresh standalone Release: 14/14 CTest tests passed in 62.25 seconds;
- fresh standalone Debug: 14/14 passed in 65.69 seconds;
- ASan/UBSan supported subset: 4/4 passed, plus an instrumented generated GEMM;
- native `.mdsl -> .o -> executable`: printed `MDSLC CPU GEMM PASS`;
- object inspection: ordinary ELF64 x86-64 relocatable with the expected C ABI
  runtime boundary;
- fresh install and external `find_package(MatcoreDSL REQUIRED)` consumer:
  passed, including no-op rebuild and absolute-path scan;
- legacy frontend contract: 14/14 passed;
- strict credential scan of 922 retained blobs, 62 removed blobs, and all
  commit/tag messages: no match; and
- independent pre-push review: no unresolved high or medium finding.

Fresh network-clone validation from GitHub repeated:

- clean `main` at `926df63a44c8fc6515ac3dd4035442b7b20bd1cb`;
- tree `90e04d986d1b9c2621d3a77d69fa85624663c0b3`;
- `git fsck --full --strict`, repository hygiene, and `git diff --check` passed;
- no purge path or blob reachable from fetched branches or tags;
- fresh Release: 14/14 CTest tests passed in 61.83 seconds;
- native GEMM object, ordinary final link, execution, `file`, `readelf`, `nm`,
  and `ldd` checks passed;
- fresh install and external consumer passed; and
- legacy frontend contract: 14/14 passed.

The root legacy CMake probe remains intentionally unclaimed: it requests MLIR
18.1.3 while this machine currently exposes MLIR 22.1.2. No system toolchain was
changed for this history migration.

Hosted runs on the rewritten checkpoint also passed:

- `ci`: run `29917486163`;
- `repository-hygiene` on `main`: run `29917486228`;
- `mdslc-native` on the rewritten native branch: run `29917486287`; and
- `repository-hygiene` on the rewritten hygiene branch: run `29917486596`.

## Clone migration

Existing clones must not merge, tag, or push their pre-rewrite refs.

```sh
git fetch --all --prune --tags
# Prefer a fresh clone.
# Do not merge or push old pre-rewrite branches.
```

The safe migration is to preserve any uncommitted work outside Git, archive the
old clone, and clone `git@github.com:onlyxItachi/MatcoreDSL.git` again. Recovery
bundles are for offline incident recovery only and must never be added as a
remote or pushed to GitHub.

## GitHub retention limitation

No user-controlled branch or tag points to pre-rewrite history. GitHub-managed
pull-request refs, caches, and backend object storage can retain unreachable
objects temporarily. In particular, closed PR #3 remains closed and unmerged;
it was not reopened or merged. This report does not claim that GitHub has
physically garbage-collected inaccessible objects.
