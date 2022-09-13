#PBS -l ncpus=48
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l ngpus=4
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/path/to/gdata
#PBS -l wd

project_dir='AGCDM'

module load python3/3.9.2
cd /path/to/project/

python3 script/experiment1.py --train_data_file=dataset/FrcSub --output_dir=./results/logs/FrcSub.log  --meta=True --do_train --do_test