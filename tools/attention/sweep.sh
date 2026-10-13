# The settings sweep behind docs/chapter-7-training.md ("How the settings were chosen"): 2,000-step runs (4,000 for
# batches of 4), measured on validation stories 1,001 to 1,300. About 4 minutes each with 16 workers.
cd "$(dirname "$0")"
mkdir -p work/runs
run() { name=$1; shift; node ptrain.js every=1000 val=300 save=$name "$@" > work/runs/$name.txt 2>&1; }
run s_h8 steps=2000 H=8 dk=8 dv=16 lr=0.002
run s_h4 steps=2000 H=4 dk=8 dv=32 lr=0.002
run s_h16 steps=2000 H=16 dk=8 dv=8 lr=0.002
run s_h8lr1 steps=2000 lr=0.001
run s_h8lr4 steps=2000 lr=0.004
run s_h8T16 steps=2000 lr=0.002 T=16
run s_lr6 steps=2000 lr=0.006
run s_lr8 steps=2000 lr=0.008
run s_lr12 steps=2000 lr=0.012
run s_lr16 steps=2000 lr=0.016
run s_B4lr4 steps=4000 B=4 lr=0.004 every=2000
run s_h16lr6 steps=2000 H=16 dv=8 lr=0.006
run s_T16lr6 steps=2000 T=16 lr=0.006
grep -H "^step 2000 \|^step 4000 " work/runs/s_*.txt
