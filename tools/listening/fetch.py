# Stream the Speech Commands v0.02 tarball and keep only the ten digit folders and the official split lists.
import sys, tarfile, urllib.request, os, time
out = sys.argv[1]
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
keep_dirs = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
keep_files = {"testing_list.txt", "validation_list.txt", "LICENSE", "README.md"}
t0 = time.time(); n = 0
with urllib.request.urlopen(url) as r:
    with tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            name = m.name[2:] if m.name.startswith("./") else m.name
            top = name.split("/")[0]
            if (top in keep_dirs and m.isfile() and name.endswith(".wav")) or name in keep_files:
                m.name = name
                tf.extract(m, out, filter="data")
                n += 1
                if n % 2000 == 0: print(n, "files", round(time.time() - t0), "s", flush=True)
print("done", n, "files in", round(time.time() - t0), "s", flush=True)
