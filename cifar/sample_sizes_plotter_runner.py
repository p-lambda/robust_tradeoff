import subprocess
import shlex

CODALAB_WKSHT="YOUR_WKSHT_HERE"
names = [f'sample-sizes-stats-{i}' for i in range(1, 11)]
deps = ':' + ' :'.join(names)
name_list = ' '.join(names)
cmd = f"cl run -m -n sample-sizes-plot -w {CODALAB_WKSHT} --request-queue tag=nlp --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-memory 4g --request-cpus 1 {deps} :sample_sizes_plotter.py --- python sample_sizes_plotter.py --paths {name_list}"
print(cmd)
subprocess.run(shlex.split(cmd))
