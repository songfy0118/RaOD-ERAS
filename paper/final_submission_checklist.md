# Final Submission Checklist

## Required Before Final Upload

```text
1. Replace placeholder author names.
2. Replace placeholder institution.
3. Replace placeholder email.
4. Add acknowledgements/funding if required by the conference.
5. Rebuild PDF.
6. Rebuild CCIS submission package.
7. Run final_submission_check.py.
8. Copy title, abstract, keywords, and statements into the conference submission form.
```

## Commands

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json
```

Draft build with placeholders, if needed:

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json --allow-placeholders
```

## Current Expected Failure

```text
author_metadata_replaced will fail until real author, institution, and email are provided.
```

## Submission Form Text

```text
Use paper/submission_form_fields.md for title, abstract, keywords, contribution, dataset statement, and code availability.
Use paper/statements.md for data availability, ethics, conflict of interest, funding, acknowledgements, and limitations.
Use paper/最后提交怎么做.md for the Chinese final operation guide.
Use paper/author_metadata_examples.md for author JSON examples.
```
