import re, os
p = "docs/midterm_report.md"
txt = open(p, encoding="utf-8").read()
links = re.findall(r"\]\(([^)]+)\)", txt)
bad = []
for l in links:
    if l.startswith(("http://","https://","#","mailto:")):
        continue
    f = os.path.normpath(os.path.join("docs", l))
    if not os.path.exists(f):
        bad.append((l, f))
print("internal links:", len([l for l in links if not l.startswith(('http','#'))]))
print("broken:", bad if bad else "none")
# 结构自检
import subprocess
print("headings:", len(re.findall(r"^#{1,3} ", txt, re.M)))
print("tables:", txt.count("\n|---"))
print("words(zh chars):", len(re.findall(r"[\u4e00-\u9fff]", txt)))
