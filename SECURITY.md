# Security

## Reporting

Open a private security advisory on the repository rather than a public issue.

---

## Resolved: customer geography in git history

**Status: remediated on 2026-08-31** by a history rewrite and force-push.
One follow-up action remains and only the repository owner can perform it —
see [Remaining action](#remaining-action).

### What was exposed

Model artifacts (`models/**/model.pkl`, `mlruns/**`) were committed to this public
repository before the geographic features were removed from the model. They had
been untracked at `HEAD`, but **git keeps every version of every file ever
committed**, so 95 blobs (~3.1 MB) remained reachable from history and could be
recovered by anyone who cloned the repository.

The pickled artifacts contain a fitted `OneHotEncoder` whose `categories_` arrays
hold, as plaintext strings, the distinct values seen during training:

| Data | Detail |
|---|---|
| City names | 997 distinct California cities from the training set |
| ZIP codes | ~1,650 distinct values |
| `Lat Long` | ~1,650 coordinate pairs, as literal strings |

This is customer geography at household granularity. It is not aggregated and not
hashed — the encoder needs the literal values to transform new rows, so it stores
them verbatim.

An `mlflow.db` SQLite file and every experiment bundle were affected.

### What is already fixed

* **The current model contains none of it.** Geography is excluded in
  [`build_features.py`](src/churn_system/features/build_features.py) at the builder,
  so it is never fitted, never encoded, and never enters `feature_schema`.
* **The API no longer collects it.** The geographic fields are accepted for
  backward compatibility and discarded before inference — they never reach
  `build_features` and never reach the event store.
* **The event store redacts it.** `SENSITIVE_KEYS` strips geography before a
  prediction row is persisted.
* **`/explain` no longer discloses it**, because the features do not exist.
* **`.gitignore` covers `models/` and `mlruns/`**, so it cannot recur.

So the exposure is strictly historical. Nothing being written today adds to it.

### What was done

The history was rewritten with `git-filter-repo` and force-pushed on 2026-08-31.
Verified before pushing, on a fresh clone:

| Check | Result |
|---|---|
| Commits preserved | 188 → 188 |
| Author dates unchanged | identical (contribution graph intact) |
| Working tree at `HEAD` | byte-identical to before |
| Model/mlruns blobs in history | 95 → **0** |
| Distinctive customer city strings in any blob | **0** |
| `git fsck` | clean |
| Pack size | 3.4 MB → 2.6 MB |

`--prune-empty=never` was used deliberately: without it the two
`chore: untrack ...` commits become empty and are dropped, costing two
contributions.

A mirror backup of the pre-rewrite remote was taken first.

### Remaining action

**Only the repository owner can do this, and it is not optional if the data
matters.** GitHub retains unreachable objects and serves them by SHA until it
garbage-collects. Until that happens, anyone who recorded an old commit SHA can
still fetch the blob.

[Open a GitHub Support request](https://support.github.com/contact) asking them to
run a garbage collection on `Churn_ML_System`, and reference the rewrite.

Two other things a rewrite cannot reach:

* **Existing clones and forks.** Outside your control.
* **Anything that already scraped the repository.**

Treat the data as disclosed. The rewrite reduces further exposure; it is not an
undo.

Everyone with a working copy must **re-clone**. `git pull` on a rewritten history
merges the old and new histories and reintroduces the blobs.

### Reproducing the rewrite

The procedure is scripted in
[`scripts/remediate_pii_history.sh`](scripts/remediate_pii_history.sh). It stops
before pushing and prints the publish commands.

**Before you start, understand what this does and does not achieve.**

It *does* remove the blobs from the canonical repository. It does **not**:

* remove them from existing clones or forks — those are outside your control;
* remove them from GitHub's own storage immediately. Unreachable objects stay
  accessible by SHA until GitHub garbage-collects them. You must
  [contact GitHub Support](https://support.github.com/contact) and ask them to run
  a GC on the repository, quoting the commit SHAs;
* remove them from anything that already scraped the repository.

If the data is genuinely sensitive, treat it as disclosed and act accordingly. A
history rewrite reduces further exposure; it is not an undo.

**The rewrite has been prepared and verified.** It preserves every commit, every
author date (so the contribution graph is unchanged), and produces a working tree
at `HEAD` byte-identical to the current one. The whole procedure — backup, rewrite,
and verification — is scripted in
[`scripts/remediate_pii_history.sh`](scripts/remediate_pii_history.sh), which stops
before pushing and prints the publish commands for you to run deliberately:

```bash
pip install git-filter-repo
bash scripts/remediate_pii_history.sh <github-owner>
```

The manual equivalent:

```bash
# 1. Work on a fresh clone; never rewrite in place.
git clone https://github.com/<owner>/Churn_ML_System.git churn-rewrite
cd churn-rewrite

# 2. Keep a full backup you can restore from.
git clone --mirror https://github.com/<owner>/Churn_ML_System.git ../churn-backup.git

# 3. Strip the artifacts from every commit.
#    --prune-empty=never matters: without it, the two "chore: untrack ..." commits
#    become empty and are dropped, losing two contributions.
pip install git-filter-repo
git filter-repo --force --prune-empty=never \
    --path mlruns/ --path models/ \
    --path-glob '*.pkl' --path-glob '*.db' \
    --invert-paths

# 4. Verify before pushing anything.
git rev-list --count HEAD                       # expect the original count
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(rest)' \
  | awk '$1=="blob" && ($3 ~ /\.(pkl|db)$/ || $3 ~ /^(mlruns|models)\//)'
                                                # expect no output

# 5. filter-repo removes the remote on purpose. Re-add it, then force-push.
git remote add origin https://github.com/<owner>/Churn_ML_System.git
git push --force --all origin
git push --force --tags origin

# 6. Ask GitHub Support to garbage-collect the unreachable objects.
```

Everyone with a clone must re-clone afterwards; `git pull` on a rewritten history
produces a merge of the old and new histories and reintroduces the blobs.

### If you would rather not rewrite

The alternatives, in decreasing order of effectiveness:

1. **Delete and recreate the repository**, pushing only the cleaned history. This
   also removes forks. It resets stars, issues and the repository's creation date.
2. **Make the repository private.** Existing forks remain public.
3. **Accept the exposure** and document it — which is what this file does.

---

## Fail-closed secrets

Two environment variables have no defaults, and the application refuses to operate
without them. This is deliberate: a missing secret must not silently downgrade to
a weaker mode.

| Variable | Guards | Failure mode if defaulted |
|---|---|---|
| `CHURN_ARTIFACT_SIGNING_KEY` | HMAC-SHA256 over the model bundle, verified **before** `pickle.load` | A writable `models/` volume becomes remote code execution in the API process |
| `CHURN_SUBJECT_KEY_SALT` | Pseudonymous subject key linking a customer's prediction rows | An unsalted hash is trivially reversible against a known customer-id list |

Generate each with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Use the same signing key everywhere that trains, promotes or serves — a mismatch
makes every existing bundle fail verification. Rotating the subject salt orphans
existing rows, so an erasure request will no longer find predictions written under
the previous salt.

`docker compose` refuses to start without both, and `.env` is gitignored.

## Trust boundaries

* The API mounts `models/` **read-only**. The process that unpickles the artifact
  cannot write it.
* The scheduler mounts it read-write; it is the only component that promotes.
* Signature verification happens before deserialisation, not after, and fails
  closed. `CHURN_ALLOW_UNSIGNED_MODEL=1` exists for local experiments and must
  never be set in a deployment.
* `/admin/*` accepts a dedicated `CHURN_ADMIN_API_KEY` when set, so a leaked
  prediction key does not grant model-reload or erasure.
* Prometheus is bound to loopback in Compose: it serves unauthenticated PromQL
  over every retained series.
