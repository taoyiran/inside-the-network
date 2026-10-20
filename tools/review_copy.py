# Build single-file review copies of chapter pages: the shared assets and the chapter's own data files inlined,
# and the document wrapper dropped (a hosted review page supplies its own).
# python3 tools/review_copy.py <out dir> [words/index.html ...]   (default: every chapter)
import os, re, sys
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1]
pages = sys.argv[2:] or ["touch/index.html", "loss-and-gradients/index.html", "backprop/index.html", "going-deeper/index.html", "seeing/index.html", "listening/index.html", "words/index.html"]
read = lambda *p: open(os.path.join(repo, *p), encoding="utf8").read()
css, ag, common = read("assets/site.css"), read("assets/autograd.js"), read("assets/common.js")
os.makedirs(out, exist_ok=True)
for src in pages:
    s = read(src)
    s, n1 = re.subn(r'<link rel="stylesheet" href="(?:\.\./)?assets/site\.css">', lambda m: "<style>\n" + css + "</style>", s)
    s, n2 = re.subn(r'<script id="autograd" src="(?:\.\./)?assets/autograd\.js"></script>', lambda m: '<script id="autograd">\n' + ag + "</script>", s)
    s, n3 = re.subn(r'<script src="(?:\.\./)?assets/common\.js"></script>', lambda m: "<script>\n" + common + "</script>", s)
    for f in ("digits.js", "sounds.js", "stories.js", "weights.js"):
        s = re.sub(r'<script src="%s"></script>' % re.escape(f), lambda m: "<script>\n" + read(os.path.dirname(src), f) + "</script>", s)
    assert n1 == 1 and n3 == 1, (src, n1, n3)
    head = s[s.index("<title>"):s.index("</head>")].rstrip()
    body = s[s.index("<body>") + len("<body>"):s.index("</body>")].strip()
    name = (os.path.dirname(src) or "overview") + ".html"
    open(os.path.join(out, name), "w", encoding="utf8").write(head + "\n\n" + body + "\n")
    print(name, len(head) + len(body), "bytes")
