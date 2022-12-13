from tabulate import tabulate
import torch.multiprocessing as mp
import psutil
import GPUtil
import signal
import os


def get_size(n_bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if n_bytes < factor:
            return f"{n_bytes:.2f}{unit}{suffix}"
        n_bytes /= factor


class MemoryTracker(mp.Process):
    def __init__(self, pids):
        super(MemoryTracker, self).__init__()
        self.daemon = True
        self.main_pid = os.getpid()
        self.pids = pids  # test, n_plots, memory, learner, n_agents if not eval else test, n_plots

    def run(self):
        self.print_info()
        print('pids', self.pids)
        while True:
            if psutil.virtual_memory().percent > 95:
                print()
                print('!' * 70, 'Excessive RAM Consumption', '!' * 70)
                self.print_info()
                [os.kill(pid, signal.SIGTERM) for pid in self.pids]

    @ staticmethod
    def print_info():
        print()
        # CPU Information
        print("=" * 40, "CPU Info", "=" * 40)
        print(f"Total CPU Usage: {psutil.cpu_percent()}%")

        # Memory Information
        print("=" * 40, "Memory Information", "=" * 40)
        # get the memory details
        svmem = psutil.virtual_memory()
        print(f"Total: {get_size(svmem.total)}")
        print(f"Available: {get_size(svmem.available)}")
        print(f"Used: {get_size(svmem.used)}")
        print(f"Percentage: {svmem.percent}%")

        # GPU information
        print("=" * 40, "GPU Details", "=" * 40)
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = gpu.id
            # name of GPU
            gpu_name = gpu.name
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load * 100}%"
            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = gpu.uuid
            list_gpus.append((
                gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
                gpu_total_memory, gpu_temperature, gpu_uuid
            ))

        print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                           "temperature", "uuid")))
        print()
