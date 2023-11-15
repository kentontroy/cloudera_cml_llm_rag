import subprocess

print(subprocess.run(["sh jobs/download_models.sh"], shell=True))
