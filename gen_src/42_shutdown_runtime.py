#@title Shutdown runtime 
#@markdown Useful with the new Colab policy.\
#@markdown If on, shuts down the runtime after every cell has been run successfully.

shut_down_after_run_all = False #@param {'type':'boolean'}
if shut_down_after_run_all and is_colab:
  from google.colab import runtime
  runtime.unassign()