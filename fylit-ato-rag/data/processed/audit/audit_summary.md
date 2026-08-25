# ATO corpus audit summary

Total files audited: **5588**

## Header compliance
- OK (all required fields present): 5588 (100.0%)
- Missing-field counts: {'description': 340}

## Encoding
- Files with non-clean UTF-8: 0

## Nav boilerplate consistency
- Full nav block present: 5588 (100.0%)
- Partial nav block: 0
- No nav block at all: 0

## QC footer code shape
- {'expected_pair': 5579, 'more_than_two': 3, 'missing': 6}

## Possible duplicate list-item artifact
- Files affected: 750 (13.42%)

## Markdown tables
- Files containing at least one table: 1119 (20.03%)

## Financial-year mention location (preview for Step 3)
- {'none_found': 4047, 'in_body_only': 1236, 'in_heading': 78, 'in_url': 215, 'in_title': 12}

## Coverage vs menu_tree.json
- 100.0% of scraped source_urls are known keys in menu_tree.json
- Files missing from menu_tree.json: 0

## Coverage vs visited_urls.json
- Visited but never scraped into a file: 836
- Scraped but not present in visited_urls.json: 0

## Documents by category
- {'businesses-and-organisations': 2750, 'individuals-and-families': 2407, 'root': 1, 'tax-and-super-professionals': 430}

Full details (including example file paths per finding) are in audit_report.json.