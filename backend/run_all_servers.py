import subprocess
import os
import sys
import platform

# Define service folders and their main scripts
services = {
    "server_1": "app.py",
    "server_2": "app.py",
    "server_3": "app.py",
    "helpers": "segmentation_server.py"
}

def check_and_install_requirements(folder_path):
    """Checks if requirements are met and installs them if not."""
    req_file = os.path.join(folder_path, "requirements.txt")
    if not os.path.exists(req_file):
        print(f"[*] No requirements.txt found in {folder_path} — skipping.")
        return

    print(f"[+] Checking dependencies in: {req_file}")

    check_process = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=folder_path,
        capture_output=True,
        text=True
    )

    print(f"[*] Attempting to install dependencies from {req_file}")
    install_process = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        cwd=folder_path,
        capture_output=True,
        text=True
    )

    if install_process.returncode == 0:
        print(f"[✓] Dependencies are up to date for {folder_path}")
    else:
        print(f"[!] Failed to install dependencies for {folder_path}")
        print(f"[!] STDERR: {install_process.stderr}")
        print(f"[!] STDOUT: {install_process.stdout}")


def launch_servers():
    """Launches all defined services in new terminal windows."""
    current_os = platform.system()

    for folder, script in services.items():
        folder_path = os.path.abspath(folder)
        script_path = os.path.join(folder_path, script)

        if not os.path.exists(script_path):
            print(f"[!] Script not found: {script_path} — skipping.")
            continue

        check_and_install_requirements(folder_path)

        command = f"cd /d {folder_path} && {sys.executable} {script}" if current_os == "Windows" else f"cd {folder_path} && {sys.executable} {script}"

        print(f"[→] Launching '{script}' in a new terminal...")

        try:
            if current_os == "Windows":
                subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", command])
            elif current_os == "Darwin":  # macOS
                subprocess.Popen([
                    'osascript',
                    '-e', f'tell app "Terminal" to do script "{command}"'
                ])
            elif current_os == "Linux":
                subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"])
            else:
                print(f"[!] Unsupported OS: {current_os}")
                continue
        except Exception as e:
            print(f"[!] Failed to launch terminal for {folder}: {e}")


if __name__ == "__main__":
    launch_servers()