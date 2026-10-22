# TB_Prediction_Model

Basic running instructions:

Requires make. If not installed, run "sudo apt install make" (for linux) (Restart terminal and continue)

1. cd src
2. make install (this will install all the dependencies needed)
3. make xyz ARGS="arg1 arg2 arg3" (where xyz is the name of the job you want to run. run "make help" to see list of possible jobs)


NOTE: Process_raw_images can take a large amount of ram. Set add -t x flag to the end of the command line call where x is the number of threads to use if memory problems arise (12 threads can peak at up to ~8GB of RAM for a short time. If memory problems, reduce number of threads. Leaving the -t flag out will default to the maximum number of available threads). It can also be a bit strange with the multiprocessing so if RAM is not an issue but its still failing, try running it again. Otherwise, set threads to 1 or 2 less than the max and try again.