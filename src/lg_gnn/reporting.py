"""Source-tagged records and deliberately constrained, auditable reporting.

The checker accepts canonical structured claims, not arbitrary prose semantics.
"""
from dataclasses import asdict, dataclass
import json
import math


def category(value):
    if not math.isfinite(value):
        return 'not_assessed'
    return 'reduced' if value <= -1.5 else 'not_reduced' if value >= -1.0 else 'uncertain'


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    region: str
    laterality: str
    modality: str
    value: float | None
    unit: str
    source: str
    quality: str

    def __post_init__(self):
        if self.source not in ('measured', 'predicted') or self.modality not in ('MRI', 'PET'):
            raise ValueError("Explicit measured/predicted source and MRI/PET modality required")
        if self.laterality not in ('L', 'R', 'B', 'NA') or self.unit != 'z_score':
            raise ValueError("Explicit laterality and z_score unit required")
        if self.quality not in ('pass', 'not_assessed', 'failed'):
            raise ValueError("Unknown quality flag")
        if self.quality == 'pass' and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("Passing evidence must have a finite value")


def build_records(regions, measured_z, predicted_z, covered, qc):
    records = []
    for r, region in enumerate(regions):
        for m, modality in enumerate(('MRI', 'PET')):
            for source, values in (('measured', measured_z), ('predicted', predicted_z)):
                quality = ('not_assessed' if not covered[r] else
                           'failed' if (source == 'measured' and not qc[r, m]) or not math.isfinite(float(values[r, m]))
                           else 'pass')
                value = float(values[r, m]) if quality == 'pass' else None
                records.append(EvidenceRecord(f'{region["id"]}:{modality}:{source}', region['name'],
                    region['laterality'], modality, value, 'z_score', source, quality))
    return records


def canonical_finding(record):
    if record.quality != 'pass':
        raise ValueError("Cannot assert a finding from unassessed/failed evidence")
    cat = category(record.value)
    source = 'Measured' if record.source == 'measured' else 'Predicted estimate of'
    statement = (f'{source} {record.modality} {record.region} ({record.laterality}): '
                 f'{cat.replace("_", " ")} (z={record.value:.2f}) [{record.id}].')
    return {'statement': statement, 'evidence_ids': [record.id], 'source': record.source,
            'region': record.region, 'laterality': record.laterality, 'modality': record.modality,
            'value': record.value, 'unit': record.unit, 'category': cat}


def validate_finding(finding, records):
    """All cited IDs must support all fields. Prose must equal canonical rendering."""
    by_id = {record.id: record for record in records}
    ids = finding.get('evidence_ids')
    if not isinstance(ids, list) or len(ids) != 1 or any(not isinstance(i, str) or i not in by_id for i in ids):
        return False
    for evidence_id in ids:
        record = by_id[evidence_id]
        if record.quality != 'pass':
            return False
        expected = canonical_finding(record)
        if set(finding) != set(expected):
            return False
        if any(finding.get(key) != expected[key] for key in expected if key != 'evidence_ids'):
            return False
    return True


def modality_agreement(records):
    passing = {(r.region, r.laterality, r.modality): r for r in records
               if r.source == 'measured' and r.quality == 'pass'}
    discordant = []
    for region, side in sorted({(key[0], key[1]) for key in passing}):
        mri, pet = passing.get((region, side, 'MRI')), passing.get((region, side, 'PET'))
        if mri and pet and {category(mri.value), category(pet.value)} == {'reduced', 'not_reduced'}:
            discordant.append({'region': region, 'laterality': side, 'evidence_ids': [mri.id, pet.id]})
    return {'discordant_measured_regions': discordant,
            'interpretation': 'Measurement disagreement only; no cause or pathology inferred.'}


def checked_summary(candidate, records, probabilities):
    if len({r.id for r in records}) != len(records):
        raise ValueError("Duplicate evidence IDs")
    probabilities = {key: float(probabilities[key]) for key in ('NC', 'MCI', 'AD')}
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
        raise ValueError("Calibrated probabilities must be supplied externally")
    findings = candidate.get('findings', [])
    if not isinstance(findings, list):
        findings = []
    accepted = [f for f in findings if isinstance(f, dict) and validate_finding(f, records)]
    limitations = sorted({f'{r.region} ({r.laterality}) {r.modality}: {r.quality}'
                          for r in records if r.quality != 'pass'})
    return {'findings': accepted, 'modality_agreement': modality_agreement(records),
            'limitations': limitations, 'probabilities': probabilities,
            'rejected_findings': len(findings) - len(accepted),
            'status': 'verified_structured_output' if accepted else 'insufficient_evidence'}


def template_summary(records, probabilities, reduced_only=False):
    findings = [canonical_finding(r) for r in records if r.quality == 'pass'
                and (not reduced_only or category(r.value) == 'reduced')]
    return checked_summary({'findings': findings}, records, probabilities)


def reporting_prompt(records, probabilities):
    return ('Select relevant findings from the canonical finding objects below. Return JSON '
            'with a findings list. Copy selected objects exactly, including evidence IDs and '
            'source tags; predicted values are estimates. Do not add diagnoses or new claims. '
            'Unassessed regions cannot be called normal. The application copies probabilities '
            'and computes limitations and measured modality agreement.\n' + json.dumps({
                'canonical_findings': [canonical_finding(r) for r in records if r.quality == 'pass'],
                'evidence': [asdict(r) for r in records], 'probabilities': probabilities}, allow_nan=False))


def generate_checked(generator, records, probabilities):
    """Generator callable(prompt)->JSON text; at most one regeneration, then filter."""
    prompt = reporting_prompt(records, probabilities)
    candidate = {}
    for attempt in range(2):
        try:
            parsed = json.loads(generator(prompt))
            candidate = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            candidate = {}
        result = checked_summary(candidate, records, probabilities)
        if result['status'] == 'verified_structured_output' and not result['rejected_findings']:
            return result
        prompt += '\nInvalid output. Return only the requested JSON and copy canonical findings exactly.'
    return checked_summary(candidate, records, probabilities)
