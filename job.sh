#!/bin/bash
#YBATCH -r dgx-a100_1
#SBATCH -N 1
#SBATCH -o /home/szhang94/TensorCoreQR/testy%j.out
#SBATCH --time=72:00:00
#SBATCH -J s_chol
#SBATCH --error /home/szhang94/TensorCoreQR/test%j.err


./tc_dgeqrf 131072 1024 128




