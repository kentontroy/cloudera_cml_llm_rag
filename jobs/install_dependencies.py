import subprocess

print(subprocess.run(["sh jobs/install_dependencies.sh"], shell=True))
