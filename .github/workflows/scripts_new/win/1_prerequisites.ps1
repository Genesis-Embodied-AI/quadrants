$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

pip install -U pip
pip install --group dev

# Set up the build toolchain, then exit WITHOUT building (-w writes env vars and stops before
# the wheel build). With a pre-installed Visual Studio (discovered via vswhere) this is a no-op
# for the compiler; only if no VS is found does build.py install Build Tools and exit non-zero
# (intentional, ignored by SilentlyContinue). This avoids building the wheel twice.
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "build.py","-w","prereq-env.ps1" -ErrorAction SilentlyContinue -Wait
