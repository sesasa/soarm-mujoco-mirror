# SOARM–MuJoCoMirror project report

- `SOARM_MuJoCoMirror_Thesis_Report.pdf`: 24-page thesis-style report.
- `report.md`: editable report text.
- `build_report.py`: PDF and diagram generator (ReportLab and Pillow).
- `assets/`: copied historical figures and explanatory diagrams.
- `evidence_manifest.json`: source hashes, package versions, configuration snapshot and image origins.
- `test_results.txt`: nine hardware-free regression tests, all passed.
- `document_checks.json`: final PDF layout checks.

Rebuild from the project root:

```bash
.venv/bin/python reports/soarm_thesis/build_report.py
```

Rebuilding refreshes the evidence manifest from the then-current workspace. Preserve the existing folder if its snapshot needs to remain immutable. No robot or camera modules are imported by the report builder.
