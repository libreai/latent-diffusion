#!/home/vedax/.pyenv/shims/python
# -*- coding: utf-8 -*-

import sys


def main(argv):
    prompts = argv[1]
    with open(prompts, "r") as fin, open("paint.sh", "w") as fout:
        for p in fin:
            p = p.strip()
            fout.write(f"""python scripts/txt2img.py --prompt "{p}" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50 --plms""")
            fout.write("\n")


if __name__ == "__main__":
       main(sys.argv)
