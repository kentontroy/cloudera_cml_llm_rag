import subprocess

print(subprocess.run(["sh jobs/vectorstore_insert.sh"], shell=True))
