# natural language to plot

## Setup server and access
There are 2 ways to access the server:

1. remote desktop
  After disconnect from remote desktop, your apps are still running
   
2. ssh gauss@cesium-miner
   sftp gauss@cesium-miner

  After disconnect, the processes you start in the SSH terminal will terminate

## Jupyterlab
You can start Jupyter lab server either through remote desktop or ssh
```batch
# start the server and people can access remotely (from another computer)
# note the port number. It typically is 8888, but can be 8889, ...,
# if another server has been running
jpt20 remote

# start the server and access it locally
jpt20
```
 

