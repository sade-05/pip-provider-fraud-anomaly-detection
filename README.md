# Medicare Provider Billing Anomalies in No-Fault PIP States

**Unsupervised anomaly detection on 38,816 Medicare providers across seven no-fault insurance states**

---

## The Problem

Personal Injury Protection (PIP) insurance operates on a no-fault basis — your own insurer
pays your medical bills after an automobile accident regardless of who caused the collision.
This automatic payment mechanism, while designed to speed recovery for accident victims,
creates a structural vulnerability: organized fraud rings exploit it by steering claimants
to complicit medical providers who bill for unnecessary, inflated, or fabricated treatment.

The challenge for detecting provider-level fraud is a fundamental one — **there are no labels.**
No dataset exists that systematically marks which clinics are PIP mills and which are
legitimate practices. Supervised machine learning, which learns from labeled examples,
cannot be applied. What can be applied is the structural property that makes Medicare billing
data uniquely suited to this problem: **Medicare's allowed payment rate is fixed per procedure
code, per geography.** Every physical medicine provider in New York receives the same
reimbursement for a therapeutic exercise session. The only variable any provider controls
is what they choose to submit.

When a provider submits $495 for a service Medicare reimburses at $28 — a billing inflation
ratio of 17.6x — while averaging 34 treatment sessions per patient for soft-tissue injuries
where the clinical norm is 8 to 14, the pattern cannot be explained by legitimate practice
variation. This project formalizes that intuition into a statistically rigorous,
reproducible anomaly detection pipeline.

---

## Data

**Source:** CMS Medicare Fee-for-Service Provider Utilization and Payment Data,
Physician and Other Practitioners by Provider and Service (2023 release).
Publicly available at [data.cms.gov](https://data.cms.gov). No registration required.

**Scope:** Filtered from 9.7 million raw records to 85,862 rows representing
38,816 unique providers across seven no-fault PIP states — New York, New Jersey,
Michigan, Florida, Massachusetts, Pennsylvania, and Hawaii — and seven soft-tissue
injury specialties including Physical Medicine and Rehabilitation, Chiropractic,
Neurology, Orthopedic Surgery, Internal Medicine, and Pain Management.

**Why these states:** These are the only U.S. states with mandatory or elective
no-fault Personal Injury Protection laws and sufficient provider populations to
support statistically reliable peer group comparisons. The remaining five no-fault
states were excluded due to insufficient provider density in the target specialty
and procedure code combinations.

**Why Medicare data for a PIP problem:** PIP insurers and Medicare share a
substantial overlap in the provider networks billing for soft-tissue injury treatment.
The procedure codes, provider types, and billing patterns that appear in PIP claims
are the same ones visible in Medicare Part B data. Critically, Medicare's fixed
reimbursement rates create the analytical foundation that makes peer comparison
possible — a structural property that private insurer data, which involves negotiated
rates, does not provide.

**A note on Hawaii:** Hawaii providers are retained in the dataset but excluded from
modeling. With 958 rows across all specialties, peer groups in Hawaii are too small
(fewer than 5 members in many cases) to produce statistically reliable z-scores.
All Hawaii results are flagged as low-confidence and reported separately.

---

## Methods

### Step 1 — Feature engineering

Three features are engineered from the raw CMS billing columns. Each captures
a distinct dimension of anomalous behavior.

**Billing inflation ratio** (charge-to-payment ratio): The provider's average
submitted charge divided by the Medicare payment amount for the same service.
Medicare payment is fixed per procedure code and geography — it is the same for
every provider billing that code in that state. Submitted charge is the only
variable the provider controls. A ratio above 3x is elevated; above 8x is
anomalous; above 15x is consistent with organized billing fraud in PIP contexts.

**Services per patient:** Total service lines billed divided by the distinct
beneficiary count. This captures medically implausible treatment frequency.
For therapeutic exercise (CPT 97110) and chiropractic manipulation (CPT 98941),
evidence-based treatment guidelines support 8 to 14 sessions for acute soft-tissue
injuries. A provider averaging 30 to 50 sessions per patient for these conditions
is not treating a more severely injured population — the clinical evidence does
not support that explanation.

**Peer group z-score:** How many standard deviations above the mean a provider's
billing inflation ratio falls, computed within a granular peer group defined by
specialty, state, procedure code, and place of service. A physical medicine provider
in New York billing CPT 97110 in an office setting is compared only to other
physical medicine providers in New York billing CPT 97110 in an office setting.
This controls for legitimate variation between specialties and geographies —
a neurosurgeon's billing profile is not a valid comparison for a chiropractor.
Peer groups with fewer than five members receive a z-score of zero and a
low-confidence flag.

### Step 2 — Isolation Forest

Isolation Forest (Liu, Ting, and Zhou, 2008) detects anomalies by measuring
how quickly each provider can be separated from the rest using randomly
constructed binary decision trees. The core insight is that anomalous data
points are both few in number and different in character from normal points —
they require fewer random splits to isolate because they answer questions
differently from the crowd.

Formally, the anomaly score for provider *x* is:

$$s(x, n) = 2^{-\frac{\bar{h}(x)}{c(n)}}$$

where $\bar{h}(x)$ is the average path length across 100 trees and $c(n)$ is
the expected path length of an unsuccessful Binary Search Tree search,
used as a normalizing constant:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

Scores approaching 1.0 indicate short path lengths and high anomaly probability.
Scores near 0.5 are indistinguishable from normal. The model is configured with
`contamination=0.05`, treating the most isolated 5% of providers as anomalous —
a conservative threshold that surfaces the clearest cases without over-flagging.

No fraud labels are required. The algorithm finds statistical isolation, not
deviation from a labeled definition of fraud. This is the methodologically
appropriate choice when ground truth is unavailable.

### Step 3 — Anomaly score stability

Because Isolation Forest uses randomness to construct its trees, results can
vary slightly across runs. Stability is validated by running the model under
two independent random seeds (42 and 99) and measuring overlap among the
top 20 flagged providers. High overlap — 16 or more of 20 providers appearing
under both seeds — confirms that findings reflect genuine statistical signal
rather than an artifact of one particular random initialization.

### Step 4 — K-means clustering

K-means clustering ($k=3$) characterizes the anomalous providers by fraud typology.
Where Isolation Forest identifies who is anomalous, clustering identifies what
type of anomaly they represent. Three clusters emerge consistently from the data:

- **Billing inflation** — elevated charge ratios with moderate service volumes,
  consistent with systematic upcoding or fee schedule manipulation
- **Treatment mill** — high services-per-patient with moderate charge ratios,
  consistent with medically unnecessary treatment protocols
- **High inflation and high volume** — extreme on both dimensions simultaneously,
  consistent with organized fraud ring activity

### Step 5 — OIG exclusion list cross-reference

The Office of Inspector General publishes a monthly list of providers excluded
from Medicare participation for fraud and abuse. Cross-referencing the top 50
flagged providers against this list provides external validation: if an
unsupervised statistical model independently surfaces providers that federal
regulators have separately sanctioned, that convergence constitutes meaningful
evidence that the anomaly signals are real.

---

## Key Findings

- **6.1% of providers** (2,377 of 38,816) submit charges exceeding 10 times
  the Medicare payment rate for the same procedure — a billing inflation ratio
  that cannot be explained by legitimate practice economics
- **Three distinct fraud typologies** emerge from cluster analysis, each
  requiring a different investigative and remediation approach
- **Anomaly scores are highly stable** across independent random seeds,
  confirming that top-flagged providers reflect genuine statistical outliers
  rather than sampling artifacts
- **New York and New Jersey dominate the top-flagged provider list**,
  consistent with both states' documented PIP fraud histories and the
  concentration of organized fraud rings in the New York metro area
- **Individual physicians, not organizations, constitute the majority
  of top-flagged providers** in this Medicare dataset — a finding that
  diverges from the conventional PIP mill narrative, which emphasizes
  clinic-level organized fraud, and warrants further investigation

---

## Limitations

**Medicare data is not PIP data.** Medicare Part B covers the elderly and
disabled population under federal fee schedules. PIP covers automobile accident
victims under state no-fault laws with different reimbursement structures.
The provider networks overlap substantially, and the billing patterns observed
in Medicare data are informative proxies for PIP billing behavior, but direct
inference requires caution. A provider flagged as anomalous in Medicare data
may bill differently under PIP fee schedules.

**No ground truth labels exist.** The absence of a validated fraud label
means model performance cannot be evaluated using standard classification
metrics such as precision, recall, or AUC. Validation relies on anomaly
score stability, cluster interpretability, and OIG cross-reference — all
of which are appropriate for unsupervised settings but are not substitutes
for labeled evaluation. Future work incorporating state insurance department
fraud referral data or civil litigation records would enable supervised
validation.

**Peer groups are granular but incomplete.** Z-scores are computed within
specialty × state × procedure code × place of service groups. This controls
for the most important sources of legitimate billing variation but does not
account for patient complexity differences, urban versus rural cost variation
within states, or practice-specific factors such as subspecialty focus.
A provider treating a disproportionately severe patient population may show
elevated services-per-patient for legitimate clinical reasons.

**Hawaii peer groups are underpowered.** With 958 rows across all specialties
in Hawaii, many peer groups contain fewer than five members — insufficient for
reliable z-score computation. Hawaii results are retained in the dataset with
a low-confidence flag but should be interpreted with caution and are excluded
from the primary anomaly detection analysis.

---

## Domain context

This analysis was developed by a credentialed no-fault claims examiner with
nine years of experience in New York PIP claims adjudication, including
complex coverage determinations, SIU referral assessment, and for-hire vehicle
fraud investigation. The feature engineering decisions — the specific procedure
codes selected, the clinical benchmarks used for services-per-patient thresholds,
the identification of Physical Medicine and Chiropractic as the highest-risk
specialty categories — reflect operational knowledge of how PIP fraud manifests
in actual claim files, not arbitrary analytical choices.

The billing patterns flagged by this model are not statistical abstractions.
A physical medicine provider billing 17 times the Medicare rate for therapeutic
exercises, averaging 34 sessions per patient, with 84% of total billing volume
concentrated in a single procedure code, matches the precise profile of a PIP
mill as it presents in claims examination: medically implausible treatment
duration, inflated charges submitted to the no-fault insurer, and a narrow
service menu consistent with a clinic organized around maximizing PIP payments
rather than providing individualized care.

The value of domain expertise in data science is not the ability to build
more sophisticated models — it is the ability to ask the right questions,
select the right features, and interpret statistical output in the context
of how the underlying system actually operates. This project is an attempt
to demonstrate that combination.

---

## Repository structure

```
cms_provider_anomalies/
├── notebook_01_data_features.ipynb     # DuckDB SQL filter, feature engineering
├── notebook_02_anomaly_detection.ipynb # Isolation Forest, clustering, validation
├── data/
│   ├── raw/                            # Raw CMS CSV — not committed to GitHub
│   └── processed/
│       └── cms_features.parquet        # Engineered provider features
├── outputs/
│   ├── figures/
│   │   ├── 01_billing_inflation_by_specialty.png
│   │   ├── 02_anomaly_scatter.png
│   │   └── 03_cluster_profiles.png
│   └── reports/
│       └── flagged_providers.csv
└── README.md
```

---

## Skills demonstrated

| Skill | Implementation |
|---|---|
| Large file handling | DuckDB SQL query on 3.5 GB CSV without loading into memory |
| Feature engineering | Domain-driven billing metrics and peer group z-scores |
| Unsupervised anomaly detection | Isolation Forest with contamination parameter tuning |
| Statistical peer comparison | Z-scores within specialty × state × procedure code groups |
| Cluster analysis | K-means with domain-labeled cluster typologies |
| Model validation | Stability analysis across random seeds + OIG cross-reference |
| No-fault insurance domain | Procedure code selection, clinical benchmarks, fraud pattern recognition |
| Scientific communication | Results framed for both technical and operational audiences |

---

## Data source

Centers for Medicare and Medicaid Services. *Medicare Physician and Other
Practitioners — by Provider and Service.* Calendar Year 2023. Available at
https://data.cms.gov. Public domain.

*Note: The raw CMS data file (~3.5 GB) is not committed to this repository.
Follow the download instructions in notebook_01_data_features.ipynb to
reproduce the analysis from source.*

---

*Analysis conducted using Python 3.10. Primary libraries: DuckDB 0.10,
pandas 2.0, scikit-learn 1.4, matplotlib 3.8.*
