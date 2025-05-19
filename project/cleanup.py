import os
import psutil
import torch
import gc
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor
import time


def kill_python_processes():
    """Kill all Python processes except the current one"""
    current_pid = os.getpid()
    processes_to_kill = []

    # Collect processes to kill first
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            if proc.info["name"] == "python" and proc.info["pid"] != current_pid:
                # Only kill processes using significant CPU
                if proc.info["cpu_percent"] > 1.0:
                    processes_to_kill.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Kill processes in parallel
    if processes_to_kill:
        with ThreadPoolExecutor(max_workers=min(8, len(processes_to_kill))) as executor:
            for pid in processes_to_kill:
                executor.submit(os.kill, pid, signal.SIGTERM)


def clear_memory():
    """Clear unused memory"""
    # Clear Python garbage collector
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clear system cache (works on Unix-like systems)
    if os.name == "posix":  # Linux/Unix/MacOS
        try:
            # Only clear page cache (1) instead of all caches (3)
            subprocess.run(["sync"], check=True)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("1")
        except:
            pass


def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated() / 1024**2
        cuda_reserved = torch.cuda.memory_reserved() / 1024**2
        print(
            f"CUDA Memory: {cuda_allocated:.1f}MB allocated, {cuda_reserved:.1f}MB reserved"
        )

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Process Memory: {memory_info.rss / 1024**2:.1f}MB")

    system_memory = psutil.virtual_memory()
    print(
        f"System Memory: {system_memory.used / 1024**2:.1f}MB used, {system_memory.available / 1024**2:.1f}MB available"
    )


def main():
    start_time = time.time()
    print("Starting memory cleanup...")

    # Kill unnecessary Python processes
    print("\nKilling unnecessary Python processes...")
    kill_python_processes()

    # Clear memory
    print("\nClearing memory...")
    clear_memory()

    # Show memory usage
    print("\nCurrent memory usage:")
    get_memory_usage()

    end_time = time.time()
    print(f"\nCleanup completed in {end_time - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()
