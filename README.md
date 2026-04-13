# Medicare Provider Billing Anomalies in No-Fault PIP States

**Unsupervised anomaly detection on 38,816 Medicare providers across seven no-fault insurance states**

---

## What this project is about

Every time you get into a car accident in New York, New Jersey, Florida, or a handful of other
states, your own insurance company pays your medical bills, no matter who caused the crash.
That is what no-fault insurance means. You do not have to sue anyone. You just file a claim and
your insurer pays the doctors directly. This is called Personal Injury Protection, or PIP.
That sounds simple and fair. 

The problem is that some medical providers figured out they can
exploit this system. Medicare pays every provider the same fixed rate for the same procedure in the same state. 
Every physical medicine provider in New York gets paid exactly the same amount for a therapeutic
exercise session. The only variable a provider controls is what they submit. A provider billing $495
for a $28 service is not making a billing error — they are systematically inflating every charge.

They bill your insurer for treatments that never happened, inflate the cost of
treatments that did happen, or keep patients coming back for medically unnecessary visits
because every visit is another bill. This is called a PIP mill. A clinic organized around maximizing
insurance payments rather than treating patients.

The question this project asks is: can we find these providers using only publicly available billing
data, without anyone telling us in advance which ones are fraudulent?

---

## The data and why it works

The Centers for Medicare and Medicaid Services — CMS — publishes every year exactly how
much each doctor and clinic in America billed Medicare and how much Medicare actually paid.
The entire file is free, publicly available, and covers over nine million provider-procedure
combinations nationally. No registration is required to download it.
We filter the full nine million row dataset down to 85,862 rows covering 38,816 unique providers
across seven no-fault states — New York, New Jersey, Michigan, Florida, Massachusetts,
Pennsylvania, and Hawaii — and seven soft-tissue injury specialties including Physical Medicine
and Rehabilitation, Chiropractic, Neurology, Orthopedic Surgery, Internal Medicine, and Pain
Management.

This filtering happens using DuckDB — a tool that runs standard SQL queries directly on the raw
3.5 GB CSV file without loading it into memory. The same SQL language used in claims systems.
The 9.7 million row file never enters RAM — only the filtered results do. This is why the project
does not crash on a laptop.

Why these seven states: These are the only U.S. states with mandatory or elective no-fault PIP
laws and sufficient provider populations for statistically reliable peer group comparisons. The other
five no-fault states were excluded due to thin provider density in the target specialties.
Why Medicare data for a PIP problem: PIP insurers and Medicare share substantial overlap in
the provider networks billing for soft-tissue injury treatment. The procedure codes, provider types,
and billing patterns in PIP claims are the same ones visible in Medicare Part B data. Critically,
Medicare's fixed reimbursement rates create the analytical foundation that makes peer
comparison possible — a structural property private insurer data, which uses negotiated rates,
does not provide.

Hawaii note: Hawaii providers are retained but excluded from modeling. With only 958 rows
across all specialties, peer groups are too small to produce reliable z-scores. All Hawaii results are
flagged as low-confidence.

![Billing inflation ratio by specialty](outputs/figures/01_billing_inflation_by_specialty.png)
*Figure 1. Distribution of billing inflation ratios by specialty across 85,862 provider-procedure
combinations. The green dotted line marks the legitimate upper bound (~3x). The red dashed line
marks the anomaly threshold (8x). Physical Medicine and Chiropractic show the heaviest right tails —
consistent with their documented role in PIP mill billing.*

---

## The three features

### Step 1 — Feature engineering

Everything in the model is built from three numbers engineered from the raw CMS billing columns.
Each one captures a different dimension of suspicious behavior. None of them require a fraud
label to compute — they come directly from the billing data itself.

Feature 1 — Billing inflation ratio: The provider's average submitted charge divided by what
Medicare actually paid for that same service. Medicare payment is fixed per procedure code and
geography. Submitted charge is the only number the provider controls. A legitimate provider might 
bill 1.5x to 3x the Medicare rate to account for uninsured patients and administrative overhead. A
ratio above 8x is anomalous. Above 15x is consistent with organized billing fraud. This is the core
signal because the fixed denominator removes all legitimate sources of variation — you are only
seeing what the provider chose to charge.

Feature 2 — Services per patient: Total services billed divided by the number of distinct patients.
For soft-tissue injuries — the kind that follow car accidents — clinical guidelines support roughly 8
to 14 sessions. A provider averaging 34 sessions per patient for the same injuries is not treating a
more severely injured population. The clinical evidence does not support that explanation. What it
does support is a billing mill keeping patients in treatment far beyond what is medically warranted.

Feature 3 — Peer group z-score: This is where the statistics come in. A z-score measures how
far a value sits from the average of its group, expressed in standard deviations. The peer group is
strict: same specialty, same state, same procedure code, same place of service. A physical
medicine provider in New York billing CPT 97110 in an office setting is compared only to other
physical medicine providers in New York billing CPT 97110 in an office setting — not to surgeons,
not to providers in Texas.

z = (x - mean) / standard deviation
x = provider billing ratio | mean = peer group average | standard deviation = spread of the peer group

A z-score of 0 means exactly average. A z-score of 2 means in the top 2.3% of peers. A z-score of
3 means in the top 0.1% — three standard deviations above the mean is a level of deviation that
occurs by chance less than one time in a thousand. When a provider sits that far from their peers
on billing inflation, chance is not a convincing explanation.

### Step 2 — How Isolation Forest works

Isolation Forest is the algorithm that scores every provider by how anomalous they are. It requires
no fraud labels — it finds anomalies purely by measuring how different a provider looks from the
rest of the group. The intuition is elegant. Imagine you have a room full of 38,000 providers and you are blindfolded.
You randomly pick a dividing line through the room — "is your billing inflation ratio above 4.7?" —
and split everyone into two groups. Then you pick another random line and split again. You keep
dividing until each provider stands alone in their own section.
The providers sitting far from the crowd — billing 17 times the Medicare rate, averaging 34
sessions per patient — get separated very quickly. After just a few random questions they are
already alone because they answer every question differently from everyone else. The providers
buried in the normal cluster take many more questions to separate because they are surrounded
by similar providers on all sides.
That is exactly what Isolation Forest does, except instead of one person asking questions it builds
100 randomly constructed decision trees simultaneously, and averages the results. 
The anomaly score is:

s(x, n) = 2 ^ ( -average_path_length / c(n) )
average_path_length = how many splits to isolate this provider across 100 trees | c(n) = normalizing constant
| Score near 1.0 = anomalous | Score near 0.5 = normal

The model is set with contamination = 0.05, meaning we tell it to treat the top 5% most isolated
providers as anomalous. This is a conservative threshold — it surfaces the clearest cases without
over-flagging legitimate providers.

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

![Cluster profiles](outputs/figures/03_cluster_profiles.png?raw=true) 
*Figure 2. Normalized average feature values per cluster. Each bar group represents one
of the four features for a given cluster. The height shows the relative level of that
feature compared to the other clusters — revealing what makes each fraud typology distinct.*

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

![Provider anomaly map](outputs/figures/02_anomaly_scatter.png)
*Figure 3. Billing inflation ratio vs median services per patient for 38,816 providers.
Each point is one provider, colored by K-means cluster. Black dots mark the top 2% most
anomalous providers by Isolation Forest score. Green dotted lines mark legitimate practice
thresholds. Providers in the top-right quadrant — high inflation and high treatment volume
simultaneously — represent the highest investigation priority.*

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
