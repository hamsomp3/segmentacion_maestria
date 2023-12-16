# segmentacion_maestria


For window users, please install the following packages:

[Chocolatey](https://chocolatey.org/install)



execute the following command in the cmd.exe as administrator to install the packages:

```sh
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

```sh
choco install make
```

```sh
make local-setup
```