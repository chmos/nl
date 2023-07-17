# Natural Language Processing and ODE
Current active server is **lithium-miner**, not **cesium-miner**

## Setup server and access
There are 2 ways to access the server:

1. remote desktop (RDP)
  After disconnect from remote desktop, your apps are still running
   
2. ssh
  ```batch
  ssh gauss@cesium-miner
  sftp gauss@cesium-miner
  ```

  After disconnect, the processes you start in the SSH terminal will terminate

## Jupyterlab
You can start Jupyter lab server either through remote desktop or ssh
```shell
# start the server and people can access remotely (from another computer)
# note the port number. It typically is 8888, but can be 8889, ...,
# if another server has been running
jpt20 remote

# start the server and access it locally
jpt20
```

## Change sleep mode
After you login the server (ssh or RDP), typing the following command in a terminal:
```shell
sleep20 [x]
```
The PC will go to sleep after idling for x minutes. E.g.
```shell
sleep 20 45
```
will make the PC go to sleep if idling for 45 minutes.
```shell
sleep20 0
```
will make the PC NEVER sleep

The utility `sleep20.bat` is located in `C:\workcc\usaco\utils`

