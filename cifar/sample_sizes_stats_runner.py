import subprocess
import shlex

CODALAB_WKSHT = "YOUR_WKSHT_HERE"
i=0
for trial in [1, 2]:
    for take_fraction in [0.1, 0.125, 0.2, 0.5, 1.0]:
        i += 1
        names = []
        clean_bundle_name = f"sample-sizes-clean-labeledproportion{take_fraction}-trial{trial}"
        names.append(f"{clean_bundle_name}")
        for epsilon in [0.0039, 0.0078, 0.0118, 0.0157]:
            adv_bundle_name = f"sample-sizes-eps{epsilon}-labeledproportion{take_fraction}-trial{trial}"
            names.append(f"{adv_bundle_name}")

        deps = ':' + ' :'.join(names)
        name_list = ' '.join(names)
        cmd = f"cl run -m -n sample-sizes-stats-{i} -w {CODALAB_WKSHT} --request-queue tag=nlp --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-memory 4g --request-cpus 1 {deps} :sample_sizes_stats.py --- python sample_sizes_stats.py -p {name_list}"
        print(cmd)
        subprocess.run(shlex.split(cmd))
