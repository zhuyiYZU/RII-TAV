import logging
import subprocess
import time
from itertools import product
from datetime import datetime

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)  # 配置日志记录器

    l = ['rec-new']
    batch_sizes = {32}
    learning_rates = {'4e-5'}
    shots = {50}
    seeds = [i for i in range(100,150)]
    template_id = {0}
    verbalizer = {'manual'}
    max_epochs = {15}

    total_start_time = time.time()
    total_commands = len(list(product(l, template_id, seeds, batch_sizes, learning_rates, shots, verbalizer)))
    completed_commands = 0

    for n, t, j, i, k, m, v,  e in product(l, template_id, seeds, batch_sizes, learning_rates, shots, verbalizer,  max_epochs):
        cmd = (
            f"python fewshot0.py --result_file ./result/rec-new-pt.txt "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} --learning_rate {k} --verbalizer {v} --max_epochs {e} "
        )
        print(cmd)

        logging.info(f"Executing command: {cmd}")
        print(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing command ({completed_commands + 1}/{total_commands}): {cmd}")

        command_start_time = time.time()
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

        command_time = time.time() - command_start_time
        completed_commands += 1

        # Calculate estimated remaining time
        if completed_commands > 0:
            avg_time = (time.time() - total_start_time) / completed_commands
            remaining_time = avg_time * (total_commands - completed_commands)
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        else:
            remaining_str = "calculating..."

        print(f"Command completed in {time.strftime('%H:%M:%S', time.gmtime(command_time))}")
        print(
            f"Progress: {completed_commands}/{total_commands} | Avg time: {time.strftime('%H:%M:%S', time.gmtime(avg_time))} | Est. remaining: {remaining_str}")

        time.sleep(2)

    total_time = time.time() - total_start_time
    print(f"\nAll commands completed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")