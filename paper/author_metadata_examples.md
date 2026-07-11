# Author Metadata Examples

## Three Student Authors, One Institution

```json
{
  "authors_latex": "Feiyang Song\\inst{1} \\and Student A\\inst{1} \\and Student B\\inst{1}",
  "authorrunning": "F. Song et al.",
  "institute": "School of ..., ... University, City, China",
  "email": "your_email@example.com",
  "acknowledgements": ""
}
```

## Students plus Advisor, One Institution

```json
{
  "authors_latex": "Feiyang Song\\inst{1} \\and Student A\\inst{1} \\and Advisor Name\\inst{1}",
  "authorrunning": "F. Song et al.",
  "institute": "School of ..., ... University, City, China",
  "email": "advisor_or_corresponding@example.com",
  "acknowledgements": ""
}
```

## Multiple Institutions

```json
{
  "authors_latex": "Feiyang Song\\inst{1} \\and Collaborator A\\inst{2} \\and Advisor Name\\inst{1}",
  "authorrunning": "F. Song et al.",
  "institute": "School of ..., University A, City, China \\and Department of ..., University B, City, China",
  "email": "corresponding@example.com",
  "acknowledgements": ""
}
```

## With Funding or Acknowledgements

```json
{
  "authors_latex": "Feiyang Song\\inst{1} \\and Student A\\inst{1}",
  "authorrunning": "F. Song et al.",
  "institute": "School of ..., ... University, City, China",
  "email": "your_email@example.com",
  "acknowledgements": "This work was supported by ... ."
}
```

## Common Mistakes

```text
Do not delete \\inst{1}.
Do not write Chinese punctuation inside authors_latex.
Use \\and between authors.
Use a real email address before final submission.
Keep authorrunning short, for example: F. Song et al.
```
