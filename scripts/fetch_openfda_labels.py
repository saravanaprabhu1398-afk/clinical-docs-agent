#!/usr/bin/env python3
"""Download public FDA drug labels (openFDA) and turn each into one Markdown document.

Usage: python scripts/fetch_openfda_labels.py [--limit N]

Source: https://open.fda.gov/apis/drug/label/ (public domain, no API key needed
for small pulls). These are published product labels, so there is no patient data.
Writes data/raw/<generic>.json (the API record) and data/docs/<generic>.md
(front matter + one "##" heading per label section, ready for chunking).
"""
import argparse, json, pathlib, re, time, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "https://api.fda.gov/drug/label.json"

# Commonly prescribed generics across major therapeutic areas.
DRUGS = """
metformin glipizide glimepiride pioglitazone sitagliptin empagliflozin dapagliflozin insulin glargine liraglutide semaglutide
lisinopril enalapril ramipril losartan valsartan olmesartan amlodipine nifedipine diltiazem verapamil
metoprolol atenolol carvedilol propranolol bisoprolol hydrochlorothiazide chlorthalidone furosemide spironolactone torsemide
atorvastatin rosuvastatin simvastatin pravastatin ezetimibe warfarin apixaban rivaroxaban dabigatran clopidogrel
aspirin digoxin amiodarone nitroglycerin hydralazine sertraline fluoxetine escitalopram citalopram paroxetine
venlafaxine duloxetine bupropion mirtazapine trazodone amitriptyline nortriptyline quetiapine olanzapine risperidone
aripiprazole haloperidol lithium lamotrigine valproate levetiracetam carbamazepine phenytoin topiramate gabapentin
pregabalin alprazolam lorazepam clonazepam diazepam zolpidem buspirone methylphenidate atomoxetine donepezil
memantine levodopa ropinirole sumatriptan tramadol oxycodone morphine hydrocodone buprenorphine naloxone
ibuprofen naproxen celecoxib acetaminophen meloxicam prednisone methylprednisolone dexamethasone hydrocortisone budesonide
albuterol fluticasone montelukast tiotropium salmeterol cetirizine loratadine fexofenadine diphenhydramine omeprazole
pantoprazole esomeprazole famotidine ondansetron metoclopramide loperamide amoxicillin azithromycin doxycycline ciprofloxacin
levofloxacin cephalexin ceftriaxone clindamycin nitrofurantoin vancomycin metronidazole fluconazole acyclovir valacyclovir
oseltamivir levothyroxine methimazole alendronate tamsulosin finasteride sildenafil oxybutynin allopurinol colchicine
methotrexate hydroxychloroquine adalimumab tamoxifen letrozole anastrozole estradiol medroxyprogesterone ferrous potassium
""".split()

def fetch(generic):
    q = urllib.parse.urlencode({"search": f'openfda.generic_name:"{generic}"', "limit": 20})
    with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
        results = json.load(r).get("results", [])
    # prefer a single-ingredient label for this generic, then the newest
    def score(rec):
        names = rec.get("openfda", {}).get("generic_name", [""])
        single = all(" and " not in n.lower() and "," not in n for n in names)
        return (single, "indications_and_usage" in rec, rec.get("effective_time", ""))
    return max(results, key=score) if results else None

SECTIONS = [
    ("boxed_warning", "Boxed warning"), ("indications_and_usage", "Indications and usage"),
    ("dosage_and_administration", "Dosage and administration"), ("dosage_forms_and_strengths", "Dosage forms and strengths"),
    ("contraindications", "Contraindications"), ("warnings_and_cautions", "Warnings and precautions"),
    ("warnings", "Warnings"), ("precautions", "Precautions"), ("adverse_reactions", "Adverse reactions"),
    ("drug_interactions", "Drug interactions"), ("use_in_specific_populations", "Use in specific populations"),
    ("pregnancy", "Pregnancy"), ("pediatric_use", "Pediatric use"), ("geriatric_use", "Geriatric use"),
    ("overdosage", "Overdosage"), ("mechanism_of_action", "Mechanism of action"), ("pharmacokinetics", "Pharmacokinetics"),
]

def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^\d+(\.\d+)*\s+[A-Z &/]+\s", "", text)  # drop "5 WARNINGS AND PRECAUTIONS" prefix

def to_markdown(generic, rec):
    of = rec.get("openfda", {})
    meta = {
        "doc_id": f"fda-label-{generic}", "generic_name": "; ".join(of.get("generic_name", [generic])),
        "brand_name": "; ".join(of.get("brand_name", [])), "manufacturer": "; ".join(of.get("manufacturer_name", [])),
        "route": "; ".join(of.get("route", [])), "pharm_class": "; ".join(of.get("pharm_class_epc", [])),
        "set_id": rec.get("set_id", ""), "effective_date": rec.get("effective_time", ""),
        "source": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={rec.get('set_id', '')}",
    }
    out = ["---"] + [f"{k}: {json.dumps(v)}" for k, v in meta.items()] + ["---", "",
           f"# {meta['brand_name'] or generic.title()} ({meta['generic_name'].lower()})", ""]
    for key, title in SECTIONS:
        if rec.get(key):
            out += [f"## {title}", "", clean(" ".join(rec[key])), ""]
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=len(DRUGS)); a = ap.parse_args()
    ok, missing = 0, []
    for g in DRUGS[: a.limit]:
        try:
            rec = fetch(g)
        except Exception as e:  # 404 = no match on openFDA
            rec = None
        if not rec or not rec.get("indications_and_usage"):
            missing.append(g); continue
        (ROOT / "data/raw" / f"{g}.json").write_text(json.dumps(rec, indent=1))
        (ROOT / "data/docs" / f"{g}.md").write_text(to_markdown(g, rec))
        ok += 1; time.sleep(0.3)  # stay well under openFDA's 240 req/min
    print(f"saved {ok} documents; no usable label for: {', '.join(missing) or 'none'}")

if __name__ == "__main__":
    main()
