import sys
std_out=sys.stdout
std_err= open('stderr', 'w')
sys.stdout=std_err

sys.stdout=std_out